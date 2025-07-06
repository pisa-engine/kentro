#!/usr/bin/env python3
"""
Example usage of the Kentro K-Means clustering library Python bindings.

This script demonstrates the main features of the library, including:
- Standard K-Means clustering
- Euclidean vs Cosine similarity
- Balanced K-Means clustering
- K-medoids clustering
- Cluster assignment

Run this script after installing the package:
pip install .
python examples/python_example.py
"""

import numpy as np
from kentro import KMeans

def create_sample_data():
    """Create sample data with 3 distinct clusters."""
    np.random.seed(42)
    
    # Create 3 clusters
    cluster1 = np.random.normal([2, 2], 0.5, (30, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (30, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (30, 2))
    
    # Combine clusters
    data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
    
    return data

def demonstrate_basic_kmeans():
    """Demonstrate basic K-Means clustering."""
    print("=== Basic K-Means Clustering ===")
    
    data = create_sample_data()
    print(f"Data shape: {data.shape}")
    
    # Create and train K-Means
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.train(data, num_threads=None)
    
    print(f"Found {len(clusters)} clusters")
    print(f"Cluster sizes: {[len(c) for c in clusters]}")
    print(f"Is trained: {kmeans.is_trained}")
    print(f"Centroids shape: {kmeans.centroids.shape if kmeans.centroids is not None else 'None'}")
    print()

def demonstrate_euclidean_kmeans():
    """Demonstrate K-Means with Euclidean distance."""
    print("=== K-Means with Euclidean Distance ===")
    
    data = create_sample_data()
    
    # Create and train K-Means with Euclidean distance
    kmeans = KMeans(n_clusters=3).with_euclidean(True)
    clusters = kmeans.train(data, num_threads=None)
    
    print(f"Using Euclidean distance: {kmeans.is_euclidean}")
    print(f"Cluster sizes: {[len(c) for c in clusters]}")
    print()

def demonstrate_balanced_kmeans():
    """Demonstrate balanced K-Means clustering."""
    print("=== Balanced K-Means Clustering ===")
    
    # Create imbalanced data
    np.random.seed(42)
    cluster1 = np.random.normal([2, 2], 0.5, (60, 2))  # Large cluster
    cluster2 = np.random.normal([6, 6], 0.5, (20, 2))  # Medium cluster
    cluster3 = np.random.normal([2, 6], 0.5, (10, 2))  # Small cluster
    data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
    
    # Regular K-Means
    kmeans_regular = KMeans(n_clusters=3)
    clusters_regular = kmeans_regular.train(data, num_threads=None)
    
    # Balanced K-Means
    kmeans_balanced = KMeans(n_clusters=3) \
        .with_balanced(True) \
        .with_max_balance_diff(10)
    clusters_balanced = kmeans_balanced.train(data, num_threads=None)
    
    print(f"Regular K-Means cluster sizes: {[len(c) for c in clusters_regular]}")
    print(f"Balanced K-Means cluster sizes: {[len(c) for c in clusters_balanced]}")
    print(f"Is balanced: {kmeans_balanced.is_balanced}")
    print()

def demonstrate_medoids_clustering():
    """Demonstrate K-medoids clustering."""
    print("=== K-Medoids Clustering ===")
    
    data = create_sample_data()
    
    # Create and train K-medoids
    kmeans = KMeans(n_clusters=3) \
        .with_use_medoids(True) \
        .with_euclidean(True) \
        .with_verbose(True)
    
    clusters = kmeans.train(data, num_threads=None)
    
    print(f"Using medoids: {kmeans.is_use_medoids}")
    print(f"Cluster sizes: {[len(c) for c in clusters]}")
    
    medoid_indices = kmeans.medoid_indices
    if medoid_indices is not None:
        print(f"Medoid indices: {medoid_indices}")
        print("Medoid points:")
        for i, medoid_idx in enumerate(medoid_indices):
            print(f"  Cluster {i}: {data[medoid_idx]}")
    print()

def demonstrate_cluster_assignment():
    """Demonstrate cluster assignment on new data."""
    print("=== Cluster Assignment ===")
    
    # Train on training data
    train_data = create_sample_data()
    kmeans = KMeans(n_clusters=3)
    kmeans.train(train_data, num_threads=None)
    
    # Create test data
    test_data = np.array([
        [2.1, 2.1],  # Should be close to cluster 1
        [6.1, 6.1],  # Should be close to cluster 2
        [2.1, 6.1],  # Should be close to cluster 3
    ]).astype(np.float32)
    
    # Assign to nearest cluster
    assignments = kmeans.assign(test_data, k=1)
    
    print(f"Test data shape: {test_data.shape}")
    print("Assignment results:")
    for i, assignment in enumerate(assignments):
        if assignment:
            print(f"  Cluster {i} contains test points: {assignment}")
    print()

def demonstrate_method_chaining():
    """Demonstrate method chaining for configuration."""
    print("=== Method Chaining Configuration ===")
    
    data = create_sample_data()
    
    # Chain multiple configuration methods
    kmeans = KMeans(n_clusters=3) \
        .with_iterations(50) \
        .with_euclidean(True) \
        .with_balanced(True) \
        .with_max_balance_diff(5) \
        .with_verbose(False)
    
    clusters = kmeans.train(data, num_threads=None)
    
    print(f"Configuration:")
    print(f"  n_clusters: {kmeans.n_clusters}")
    print(f"  iterations: {kmeans.iterations}")
    print(f"  is_euclidean: {kmeans.is_euclidean}")
    print(f"  is_balanced: {kmeans.is_balanced}")
    print(f"  is_use_medoids: {kmeans.is_use_medoids}")
    print(f"  is_trained: {kmeans.is_trained}")
    print(f"Cluster sizes: {[len(c) for c in clusters]}")
    print()

def demonstrate_repr():
    """Demonstrate string representation."""
    print("=== String Representation ===")
    
    # Before training
    kmeans = KMeans(n_clusters=3).with_euclidean(True).with_balanced(True)
    print(f"Before training: {kmeans}")
    
    # After training
    data = create_sample_data()
    kmeans.train(data, num_threads=None)
    print(f"After training: {kmeans}")
    print()

def main():
    """Run all demonstrations."""
    print("Kentro K-Means Clustering Python Bindings Demo")
    print("=" * 50)
    print()
    
    demonstrate_basic_kmeans()
    demonstrate_euclidean_kmeans()
    demonstrate_balanced_kmeans()
    demonstrate_medoids_clustering()
    demonstrate_cluster_assignment()
    demonstrate_method_chaining()
    demonstrate_repr()
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    main() 