use kentro::KMeans;
use ndarray::Array2;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 K-Medoids Clustering Demo");
    println!("============================\n");

    // Generate sample data with 3 distinct clusters
    let data = Array2::from_shape_vec(
        (12, 2),
        vec![
            // Cluster 1: around (1, 1)
            1.0, 1.0, 1.1, 1.1, 1.2, 1.0, 0.9, 1.1, // Cluster 2: around (5, 5)
            5.0, 5.0, 5.1, 5.1, 4.9, 5.0, 5.0, 4.9, // Cluster 3: around (9, 9)
            9.0, 9.0, 9.1, 9.1, 8.9, 9.0, 9.0, 8.9,
        ],
    )?;

    println!(
        "📊 Generated {} data points with {} dimensions",
        data.nrows(),
        data.ncols()
    );
    println!("Data points:");
    for (i, row) in data.outer_iter().enumerate() {
        println!("  Point {}: [{:.1}, {:.1}]", i, row[0], row[1]);
    }
    println!();

    // Compare K-Means vs K-Medoids
    println!("🔄 Comparing K-Means vs K-Medoids...\n");

    // Standard K-Means
    println!("📈 Standard K-Means (centroids):");
    let start = Instant::now();
    let mut kmeans = KMeans::new(3)
        .with_euclidean(true)
        .with_iterations(10)
        .with_verbose(false);
    let clusters_means = kmeans.train(data.view(), None)?;
    let duration = start.elapsed();

    println!("  Training time: {:.2?}", duration);
    if let Some(centroids) = kmeans.centroids() {
        println!("  Centroids:");
        for (i, centroid) in centroids.outer_iter().enumerate() {
            println!(
                "    Cluster {}: [{:.3}, {:.3}]",
                i, centroid[0], centroid[1]
            );
        }
    }
    println!("  Cluster assignments:");
    for (cluster_id, points) in clusters_means.iter().enumerate() {
        println!("    Cluster {}: {:?}", cluster_id, points);
    }
    println!();

    // K-Medoids
    println!("🎯 K-Medoids (actual data points):");
    let start = Instant::now();
    let mut kmedoids = KMeans::new(3)
        .with_use_medoids(true)
        .with_euclidean(true)
        .with_iterations(10)
        .with_verbose(false);
    let clusters_medoids = kmedoids.train(data.view(), None)?;
    let duration = start.elapsed();

    println!("  Training time: {:.2?}", duration);
    if let Some(medoid_indices) = kmedoids.medoid_indices() {
        println!("  Medoids (actual data points):");
        for (i, &medoid_idx) in medoid_indices.iter().enumerate() {
            let medoid_point = data.row(medoid_idx);
            println!(
                "    Cluster {}: Point {} [{:.1}, {:.1}]",
                i, medoid_idx, medoid_point[0], medoid_point[1]
            );
        }
    }
    println!("  Cluster assignments:");
    for (cluster_id, points) in clusters_medoids.iter().enumerate() {
        println!("    Cluster {}: {:?}", cluster_id, points);
    }
    println!();

    // Show the difference
    println!("💡 Key Differences:");
    println!("  • K-Means uses computed centroids (may not be actual data points)");
    println!("  • K-Medoids uses actual data points as cluster centers");
    println!("  • K-Medoids is more robust to outliers");
    println!("  • K-Medoids ensures cluster representatives are real observations");

    Ok(())
}
