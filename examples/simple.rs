use kentro::KMeans;
use ndarray::Array2;
use rand::prelude::*;
use rand_distr::Normal;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Kentro K-Means Clustering Library Demo");
    println!("=========================================\n");

    // Generate sample data with 3 distinct clusters
    let start = Instant::now();
    let data = generate_sample_data(3000000, 768, 300)?;
    let duration = start.elapsed();
    println!("📊 Generated {} data points with {} dimensions in {:.2?}", data.nrows(), data.ncols(), duration);

    // Example 1: Basic K-Means
    println!("\n🔍 Example 1: Basic K-Means Clustering");
    println!("--------------------------------------");
    
    let mut kmeans_basic = KMeans::new(3)
        .with_iterations(50)
        .with_verbose(true);
    
    let start = Instant::now();
    let clusters_basic = kmeans_basic.train(data.view(), None)?;
    let duration = start.elapsed();
    
    println!("✅ Basic K-Means completed in {:.2?}!", duration);
    print_cluster_info(&clusters_basic);

    // Example 2: Euclidean K-Means
    println!("\n🔍 Example 2: Euclidean K-Means Clustering");
    println!("------------------------------------------");
    
    let mut kmeans_euclidean = KMeans::new(3)
        .with_euclidean(true)
        .with_iterations(30)
        .with_verbose(true);
    
    let start = Instant::now();
    let clusters_euclidean = kmeans_euclidean.train(data.view(), None)?;
    let duration = start.elapsed();
    
    println!("✅ Euclidean K-Means completed in {:.2?}!", duration);
    print_cluster_info(&clusters_euclidean);

    // Example 3: Balanced K-Means
    println!("\n🔍 Example 3: Balanced K-Means Clustering");
    println!("-----------------------------------------");
    
    let mut kmeans_balanced = KMeans::new(3)
        .with_balanced(true)
        .with_max_balance_diff(10)
        .with_iterations(25)
        .with_verbose(true);
    
    let start = Instant::now();
    let clusters_balanced = kmeans_balanced.train(data.view(), None)?;
    let duration = start.elapsed();
    
    println!("✅ Balanced K-Means completed in {:.2?}!", duration);
    print_cluster_info(&clusters_balanced);

    // Example 4: Assignment of new data points
    println!("\n🔍 Example 4: Assigning New Data Points");
    println!("--------------------------------------");
    
    // Generate some test data
    let start = Instant::now();
    let test_data = generate_sample_data(50, 2, 3)?;
    let duration = start.elapsed();
    println!("Generated test data in {:.2?}", duration);
    
    // Assign test points to nearest clusters (k=1)
    let start = Instant::now();
    let assignments = kmeans_basic.assign(test_data.view(), 1)?;
    let duration = start.elapsed();
    
    println!("📍 Assigned {} test points to clusters in {:.2?}:", test_data.nrows(), duration);
    print_cluster_info(&assignments);

    // Example 5: Multi-assignment (k-nearest clusters)
    println!("\n🔍 Example 5: Multi-Assignment (k=2 nearest clusters)");
    println!("----------------------------------------------------");
    
    let start = Instant::now();
    let multi_assignments = kmeans_basic.assign(test_data.view(), 2)?;
    let duration = start.elapsed();
    
    println!("📍 Assigned {} test points to 2 nearest clusters each in {:.2?}:", test_data.nrows(), duration);
    print_cluster_info(&multi_assignments);

    // Display centroids
    if let Some(centroids) = kmeans_basic.centroids() {
        println!("\n🎯 Final Centroids:");
        println!("------------------");
        for (i, centroid) in centroids.outer_iter().enumerate() {
            println!("Cluster {}: [{:.3}, {:.3}]", i, centroid[0], centroid[1]);
        }
    }

    println!("\n🎉 All examples completed successfully!");
    
    Ok(())
}

fn generate_sample_data(n_points: usize, n_dims: usize, n_clusters: usize) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let mut data = Vec::with_capacity(n_points * n_dims);
    
    // Define cluster centers
    let centers = match n_clusters {
        3 => vec![
            vec![2.0, 2.0],
            vec![8.0, 2.0],
            vec![5.0, 8.0],
        ],
        _ => {
            // Generate random centers
            (0..n_clusters)
                .map(|_| (0..n_dims).map(|_| rng.gen_range(-5.0..15.0)).collect())
                .collect()
        }
    };
    
    let points_per_cluster = n_points / n_clusters;
    let normal = Normal::new(0.0, 1.0)?;
    
    for (cluster_idx, center) in centers.iter().enumerate() {
        let points_in_this_cluster = if cluster_idx == n_clusters - 1 {
            n_points - (points_per_cluster * (n_clusters - 1)) // Handle remainder
        } else {
            points_per_cluster
        };
        
        for _ in 0..points_in_this_cluster {
            for dim in 0..n_dims {
                let noise: f32 = rng.sample(normal);
                let value = center[dim] + noise;
                data.push(value);
            }
        }
    }
    
    Ok(Array2::from_shape_vec((n_points, n_dims), data)?)
}

fn print_cluster_info(clusters: &[Vec<usize>]) {
    for (i, cluster) in clusters.iter().enumerate() {
        println!("  Cluster {}: {} points", i, cluster.len());
    }
    
    let total_points: usize = clusters.iter().map(|c| c.len()).sum();
    let sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
    let max_size = *sizes.iter().max().unwrap_or(&0);
    let min_size = *sizes.iter().min().unwrap_or(&0);
    
    println!("  📈 Total points: {}", total_points);
    println!("  📊 Size range: {} - {} (diff: {})", min_size, max_size, max_size - min_size);
}
