//! # Kentro - High-Performance K-Means Clustering Library
//!
//! A Rust implementation of efficient K-Means clustering with support for:
//! - Standard and spherical K-Means
//! - Balanced K-Means clustering
//! - Parallel processing
//! - Euclidean and cosine similarity metrics

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::prelude::*;
use rand_distr::Uniform;
use thiserror::Error;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Constants for the balanced K-Means algorithm
const EPS: f32 = 1.0 / 1024.0;
const PARTLY_REMAINING_FACTOR: f32 = 0.15;
const PENALTY_FACTOR: f32 = 2.5;

/// Errors that can occur during K-Means clustering
#[derive(Error, Debug)]
pub enum KMeansError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Clustering has already been trained")]
    AlreadyTrained,
    #[error("Clustering has not been trained yet")]
    NotTrained,
    #[error("Number of points ({0}) must be at least as large as number of clusters ({1})")]
    InsufficientPoints(usize, usize),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

type Result<T> = std::result::Result<T, KMeansError>;

/// High-performance K-Means clustering implementation
///
/// This implementation provides both standard and balanced K-Means clustering
/// with support for Euclidean and cosine similarity metrics.
///
/// # Examples
///
/// ```rust
/// use kentro::KMeans;
/// use ndarray::Array2;
///
/// // Create sample data
/// let data = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f32).collect()).unwrap();
///
/// // Create and train K-Means
/// let mut kmeans = KMeans::new(3).with_iterations(50).with_verbose(true);
/// let clusters = kmeans.train(data.view(), None).unwrap();
///
/// println!("Found {} clusters", clusters.len());
/// ```
pub struct KMeans {
    n_clusters: usize,
    iters: usize,
    euclidean: bool,
    balanced: bool,
    max_balance_diff: usize,
    verbose: bool,
    trained: bool,

    // Internal state
    centroids: Option<Array2<f32>>,
    assignments: Vec<usize>,
    cluster_sizes: Array1<f32>,
}

impl KMeans {
    /// Create a new K-Means instance
    ///
    /// # Arguments
    ///
    /// * `n_clusters` - The number of clusters (k)
    ///
    /// # Panics
    ///
    /// Panics if `n_clusters` is 0
    pub fn new(n_clusters: usize) -> Self {
        if n_clusters == 0 {
            panic!("Number of clusters must be positive");
        }

        Self {
            n_clusters,
            iters: 25,
            euclidean: false,
            balanced: false,
            max_balance_diff: 16,
            verbose: false,
            trained: false,
            centroids: None,
            assignments: Vec::new(),
            cluster_sizes: Array1::zeros(n_clusters),
        }
    }

    /// Set the number of iterations (default: 25)
    pub fn with_iterations(mut self, iters: usize) -> Self {
        if iters == 0 {
            panic!("Number of iterations must be positive");
        }
        self.iters = iters;
        self
    }

    /// Use Euclidean distance instead of cosine similarity (default: false)
    pub fn with_euclidean(mut self, euclidean: bool) -> Self {
        self.euclidean = euclidean;
        self
    }

    /// Enable balanced K-Means clustering (default: false)
    pub fn with_balanced(mut self, balanced: bool) -> Self {
        self.balanced = balanced;
        self
    }

    /// Set maximum balance difference for balanced clustering (default: 16)
    pub fn with_max_balance_diff(mut self, max_balance_diff: usize) -> Self {
        if max_balance_diff == 0 {
            panic!("Max balance difference must be positive");
        }
        self.max_balance_diff = max_balance_diff;
        self
    }

    /// Enable verbose output (default: false)
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Perform K-Means clustering on the provided data
    ///
    /// # Arguments
    ///
    /// * `data` - The data matrix (n_points × n_dimensions)
    /// * `num_threads` - Number of threads to use (None for automatic)
    ///
    /// # Returns
    ///
    /// A vector of vectors where each inner vector contains the indices of points
    /// assigned to the corresponding cluster
    pub fn train(
        &mut self,
        data: ArrayView2<f32>,
        num_threads: Option<usize>,
    ) -> Result<Vec<Vec<usize>>> {
        let (n, _) = data.dim();

        if self.trained {
            return Err(KMeansError::AlreadyTrained);
        }

        if n < self.n_clusters {
            return Err(KMeansError::InsufficientPoints(n, self.n_clusters));
        }

        // Set up parallel processing
        #[cfg(feature = "parallel")]
        if let Some(threads) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .map_err(|e| KMeansError::InvalidParameter(format!("Thread setup failed: {e}")))?;
        }

        // Initialize
        self.assignments = vec![0; n];
        self.cluster_sizes = Array1::zeros(self.n_clusters);

        // Compute data norms for Euclidean distance
        let data_norms = if self.euclidean {
            Some(data.map_axis(Axis(1), |row| row.dot(&row)))
        } else {
            None
        };

        // Initialize centroids by sampling random points
        self.centroids = Some(self.sample_rows(&data));
        self.postprocess_centroids();

        // Main K-Means iterations
        for iter in 0..self.iters {
            self.assign_clusters(&data, data_norms.as_ref());
            self.update_centroids(&data);
            self.split_clusters(&data);
            self.postprocess_centroids();

            if self.verbose {
                let cost = self.compute_cost(&data);
                println!("Iteration {}/{} | Cost: {:.6}", iter + 1, self.iters, cost);
            }
        }

        // Final assignment
        self.assign_clusters(&data, data_norms.as_ref());

        // Balance clusters if requested
        if self.balanced {
            if self.verbose {
                println!("Balancing clusters...");
            }
            self.balance_clusters(&data, data_norms.as_ref())?;
        }

        // Convert assignments to cluster vectors
        let mut result = vec![Vec::new(); self.n_clusters];
        for (point_idx, &cluster_idx) in self.assignments.iter().enumerate() {
            result[cluster_idx].push(point_idx);
        }

        self.trained = true;
        Ok(result)
    }

    /// Assign data points to their k nearest clusters
    ///
    /// # Arguments
    ///
    /// * `data` - The data matrix (n_points × n_dimensions)
    /// * `k` - Number of nearest clusters to assign each point to
    ///
    /// # Returns
    ///
    /// A vector of vectors where each inner vector contains the indices of points
    /// assigned to the corresponding cluster
    pub fn assign(&self, data: ArrayView2<f32>, k: usize) -> Result<Vec<Vec<usize>>> {
        if !self.trained {
            return Err(KMeansError::NotTrained);
        }

        if k == 0 {
            return Err(KMeansError::InvalidParameter(
                "k must be positive".to_string(),
            ));
        }

        let (n, m) = data.dim();
        let centroids = self.centroids.as_ref().unwrap();

        if m != centroids.ncols() {
            return Err(KMeansError::DimensionMismatch {
                expected: centroids.ncols(),
                actual: m,
            });
        }

        let centroid_norms = if self.euclidean {
            Some(centroids.map_axis(Axis(1), |row| row.dot(&row)))
        } else {
            None
        };

        let mut result = vec![Vec::new(); self.n_clusters];

        for i in 0..n {
            let point = data.row(i);
            let dots = centroids.dot(&point);

            let distances = if self.euclidean {
                let point_norm = point.dot(&point);
                let centroid_norms = centroid_norms.as_ref().unwrap();
                centroid_norms - &(&dots * 2.0) + point_norm
            } else {
                -dots
            };

            // Find k nearest clusters
            let mut indices: Vec<usize> = (0..self.n_clusters).collect();
            indices.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap());

            for &cluster_idx in indices.iter().take(k) {
                result[cluster_idx].push(i);
            }
        }

        Ok(result)
    }

    /// Get the number of clusters
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Get the number of iterations
    pub fn iterations(&self) -> usize {
        self.iters
    }

    /// Check if using Euclidean distance
    pub fn is_euclidean(&self) -> bool {
        self.euclidean
    }

    /// Check if using balanced clustering
    pub fn is_balanced(&self) -> bool {
        self.balanced
    }

    /// Get the cluster centroids
    ///
    /// Returns None if the model hasn't been trained yet
    pub fn centroids(&self) -> Option<ArrayView2<f32>> {
        self.centroids.as_ref().map(|c| c.view())
    }

    /// Check if the model has been trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

// Private implementation methods
impl KMeans {
    fn sample_rows(&self, data: &ArrayView2<f32>) -> Array2<f32> {
        let mut rng = thread_rng();
        let n = data.nrows();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut centroids = Array2::zeros((self.n_clusters, data.ncols()));
        for (i, &idx) in indices.iter().take(self.n_clusters).enumerate() {
            centroids.row_mut(i).assign(&data.row(idx));
        }

        centroids
    }

    fn assign_clusters(&mut self, data: &ArrayView2<f32>, data_norms: Option<&Array1<f32>>) {
        let centroids = self.centroids.as_ref().unwrap();
        let n = data.nrows();

        if self.euclidean {
            let centroid_norms = centroids.map_axis(Axis(1), |row| row.dot(&row));
            let data_norms = data_norms.unwrap();

            #[cfg(feature = "parallel")]
            let iter = (0..n).into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = 0..n;

            let assignments: Vec<usize> = iter
                .map(|i| {
                    let point = data.row(i);
                    let dots = centroids.dot(&point);
                    let distances = &centroid_norms - &(&dots * 2.0) + data_norms[i];

                    distances
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap()
                })
                .collect();

            self.assignments = assignments;
        } else {
            #[cfg(feature = "parallel")]
            let iter = (0..n).into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = 0..n;

            let assignments: Vec<usize> = iter
                .map(|i| {
                    let point = data.row(i);
                    let similarities = centroids.dot(&point);

                    similarities
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap()
                })
                .collect();

            self.assignments = assignments;
        }

        // Update cluster sizes
        self.cluster_sizes.fill(0.0);
        for &assignment in &self.assignments {
            self.cluster_sizes[assignment] += 1.0;
        }
    }

    fn update_centroids(&mut self, data: &ArrayView2<f32>) {
        let mut centroids = Array2::zeros((self.n_clusters, data.ncols()));

        for (i, point) in data.outer_iter().enumerate() {
            let cluster = self.assignments[i];
            centroids.row_mut(cluster).scaled_add(1.0, &point);
        }

        for j in 0..self.n_clusters {
            if self.cluster_sizes[j] > 0.0 {
                centroids
                    .row_mut(j)
                    .mapv_inplace(|x| x / self.cluster_sizes[j]);
            }
        }

        self.centroids = Some(centroids);
    }

    fn postprocess_centroids(&mut self) {
        if !self.euclidean {
            // Normalize centroids for spherical k-means
            if let Some(ref mut centroids) = self.centroids {
                for mut row in centroids.outer_iter_mut() {
                    let norm = (row.dot(&row)).sqrt();
                    if norm > 0.0 {
                        row.mapv_inplace(|x| x / norm);
                    }
                }
            }
        }
    }

    fn split_clusters(&mut self, data: &ArrayView2<f32>) {
        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0, 1.0);

        for i in 0..self.n_clusters {
            if self.cluster_sizes[i] == 0.0 {
                // Find cluster to split
                let mut j = 0;
                loop {
                    let p = (self.cluster_sizes[j] - 1.0)
                        / (data.nrows() as f32 - self.n_clusters as f32);
                    let r: f32 = rng.sample(uniform);
                    if r < p {
                        break;
                    }
                    j = (j + 1) % self.n_clusters;
                }

                // Split cluster j
                if let Some(ref mut centroids) = self.centroids {
                    let centroid_j = centroids.row(j).to_owned();
                    centroids.row_mut(i).assign(&centroid_j);

                    // Apply small symmetric perturbation
                    for k in 0..data.ncols() {
                        if k % 2 == 0 {
                            centroids[[i, k]] *= 1.0 + EPS;
                            centroids[[j, k]] *= 1.0 - EPS;
                        } else {
                            centroids[[i, k]] *= 1.0 - EPS;
                            centroids[[j, k]] *= 1.0 + EPS;
                        }
                    }
                }

                // Split cluster sizes evenly
                self.cluster_sizes[i] = self.cluster_sizes[j] / 2.0;
                self.cluster_sizes[j] -= self.cluster_sizes[i];
            }
        }
    }

    fn compute_cost(&self, data: &ArrayView2<f32>) -> f32 {
        let centroids = self.centroids.as_ref().unwrap();
        let mut total_cost = 0.0;

        if self.euclidean {
            for (i, point) in data.outer_iter().enumerate() {
                let centroid = centroids.row(self.assignments[i]);
                let diff = &point - &centroid;
                total_cost += diff.dot(&diff);
            }
        } else {
            for (i, point) in data.outer_iter().enumerate() {
                let centroid = centroids.row(self.assignments[i]);
                total_cost += point.dot(&centroid);
            }
        }

        total_cost / data.nrows() as f32
    }

    fn balance_clusters(
        &mut self,
        data: &ArrayView2<f32>,
        data_norms: Option<&Array1<f32>>,
    ) -> Result<()> {
        let centroids = self.centroids.as_ref().unwrap();
        let mut unnormalized_centroids = Array2::zeros(centroids.dim());

        // Compute unnormalized centroids
        for (i, point) in data.outer_iter().enumerate() {
            let cluster = self.assignments[i];
            unnormalized_centroids
                .row_mut(cluster)
                .scaled_add(1.0, &point);
        }

        let mut n_min = self
            .cluster_sizes
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let mut n_max = self
            .cluster_sizes
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut iters = 0;
        let mut p_now = 0.0;
        let mut p_next = f32::INFINITY;

        while n_max - n_min > 0.5 + self.max_balance_diff as f32 {
            for i in 0..data.nrows() {
                let old_cluster = self.assignments[i];
                let n_old = self.cluster_sizes[old_cluster];
                let point = data.row(i);

                // Remove point from old cluster
                unnormalized_centroids
                    .row_mut(old_cluster)
                    .scaled_add(-1.0, &point);

                // Update old centroid
                if n_old > 0.0 {
                    let new_centroid = &unnormalized_centroids.row(old_cluster) / (n_old - 1.0);
                    if let Some(ref mut centroids) = self.centroids {
                        centroids.row_mut(old_cluster).assign(&new_centroid);
                        if !self.euclidean {
                            let norm =
                                (centroids.row(old_cluster).dot(&centroids.row(old_cluster)))
                                    .sqrt();
                            if norm > 0.0 {
                                centroids.row_mut(old_cluster).mapv_inplace(|x| x / norm);
                            }
                        }
                    }
                }

                self.cluster_sizes[old_cluster] =
                    self.cluster_sizes[old_cluster] - 1.0 + PARTLY_REMAINING_FACTOR;

                // Compute distances and costs
                let centroids = self.centroids.as_ref().unwrap();
                let distances = if self.euclidean {
                    let dots = centroids.dot(&point);
                    let centroid_norms = centroids.map_axis(Axis(1), |row| row.dot(&row));
                    let point_norm = data_norms.unwrap()[i];
                    &centroid_norms - &(&dots * 2.0) + point_norm
                } else {
                    -centroids.dot(&point)
                };

                let costs = &distances + &(&self.cluster_sizes * p_now);
                let min_cluster = costs
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                // Update penalties
                let penalties_1 = &distances - distances[old_cluster];
                let penalties_2 = self.cluster_sizes[old_cluster] - &self.cluster_sizes;

                let mut min_p_value = f32::INFINITY;
                for p in 0..self.n_clusters {
                    if self.cluster_sizes[old_cluster] > self.cluster_sizes[p]
                        && penalties_2[p] != 0.0
                    {
                        let penalty = penalties_1[p] / penalties_2[p];
                        if penalty < min_p_value {
                            min_p_value = penalty;
                        }
                    }
                }

                if p_now < min_p_value && min_p_value < p_next {
                    p_next = min_p_value;
                }

                // Assign to new cluster
                self.cluster_sizes[min_cluster] += 1.0;
                unnormalized_centroids
                    .row_mut(min_cluster)
                    .scaled_add(1.0, &point);

                // Update new centroid
                let new_centroid =
                    &unnormalized_centroids.row(min_cluster) / self.cluster_sizes[min_cluster];
                if let Some(ref mut centroids) = self.centroids {
                    centroids.row_mut(min_cluster).assign(&new_centroid);
                    if !self.euclidean {
                        let norm =
                            (centroids.row(min_cluster).dot(&centroids.row(min_cluster))).sqrt();
                        if norm > 0.0 {
                            centroids.row_mut(min_cluster).mapv_inplace(|x| x / norm);
                        }
                    }
                }

                self.cluster_sizes[old_cluster] -= PARTLY_REMAINING_FACTOR;
                self.assignments[i] = min_cluster;
            }

            n_min = self
                .cluster_sizes
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            n_max = self
                .cluster_sizes
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            p_now = PENALTY_FACTOR * p_next;
            p_next = f32::INFINITY;

            iters += 1;

            if self.verbose {
                let cost = self.compute_cost(data);
                println!(
                    "Balance iteration {} | Cost: {:.6} | Max diff: {:.2}",
                    iters,
                    cost,
                    n_max - n_min
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_basic_kmeans() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1, 9.0, 9.0, 9.1, 9.1],
        )
        .unwrap();

        let mut kmeans = KMeans::new(3);
        let clusters = kmeans.train(data.view(), None).unwrap();

        assert_eq!(clusters.len(), 3);
        assert!(kmeans.is_trained());
    }

    #[test]
    fn test_euclidean_kmeans() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut kmeans = KMeans::new(2).with_euclidean(true);
        let clusters = kmeans.train(data.view(), None).unwrap();

        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_balanced_kmeans() {
        let data = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2, 9.0, 9.0, 9.1, 9.1,
                9.2, 9.2, 13.0, 13.0,
            ],
        )
        .unwrap();

        let mut kmeans = KMeans::new(3).with_balanced(true).with_max_balance_diff(2);
        let clusters = kmeans.train(data.view(), None).unwrap();

        assert_eq!(clusters.len(), 3);

        // Check that clusters are reasonably balanced
        let sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
        let max_size = *sizes.iter().max().unwrap();
        let min_size = *sizes.iter().min().unwrap();
        assert!(max_size - min_size <= 2); // Should be reasonably balanced
    }

    #[test]
    fn test_assign() {
        let train_data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1, 9.0, 9.0, 9.1, 9.1],
        )
        .unwrap();

        let test_data = Array2::from_shape_vec((2, 2), vec![1.05, 1.05, 9.05, 9.05]).unwrap();

        let mut kmeans = KMeans::new(3);
        kmeans.train(train_data.view(), None).unwrap();

        let assignments = kmeans.assign(test_data.view(), 1).unwrap();
        assert_eq!(assignments.len(), 3);
    }

    #[test]
    #[should_panic(expected = "Number of clusters must be positive")]
    fn test_zero_clusters() {
        KMeans::new(0);
    }

    #[test]
    fn test_insufficient_points() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let mut kmeans = KMeans::new(3);

        match kmeans.train(data.view(), None) {
            Err(KMeansError::InsufficientPoints(2, 3)) => {}
            _ => panic!("Expected InsufficientPoints error"),
        }
    }
}
