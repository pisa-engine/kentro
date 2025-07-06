#!/usr/bin/env python3
"""
Test script for Kentro K-Means clustering Python bindings.

This script tests the main functionality to ensure the Python bindings work correctly.
"""

import numpy as np
import sys


def test_basic_functionality():
    """Test basic K-Means functionality."""
    try:
        from kentro import KMeans
        
        # Create simple test data
        np.random.seed(42)
        data = np.random.rand(20, 2).astype(np.float32)
        
        # Test basic K-Means
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.train(data, num_threads=None)
        
        assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}"
        assert kmeans.is_trained, "Model should be trained"
        assert kmeans.n_clusters == 3, "Should have 3 clusters"
        assert kmeans.iterations == 25, "Default iterations should be 25"
        assert not kmeans.is_euclidean, "Default should be cosine similarity"
        assert not kmeans.is_balanced, "Default should not be balanced"
        assert not kmeans.is_use_medoids, "Default should not use medoids"
        assert kmeans.centroids is not None, "Centroids should be available"
        assert kmeans.centroids.shape == (3, 2), f"Expected (3, 2) centroids, got {kmeans.centroids.shape}"
        
        print("✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_configuration_methods():
    """Test configuration methods."""
    try:
        from kentro import KMeans
        
        # Test method chaining
        kmeans = KMeans(n_clusters=2) \
            .with_iterations(10) \
            .with_euclidean(True) \
            .with_balanced(True) \
            .with_max_balance_diff(5) \
            .with_verbose(False) \
            .with_use_medoids(True)
        
        assert kmeans.n_clusters == 2, "Should have 2 clusters"
        assert kmeans.iterations == 10, "Should have 10 iterations"
        assert kmeans.is_euclidean, "Should use Euclidean distance"
        assert kmeans.is_balanced, "Should be balanced"
        assert kmeans.is_use_medoids, "Should use medoids"
        
        print("✓ Configuration methods test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration methods test failed: {e}")
        return False


def test_error_handling():
    """Test error handling."""
    try:
        from kentro import KMeans
        
        # Test invalid n_clusters
        try:
            KMeans(n_clusters=0)
            print("✗ Should have raised error for 0 clusters")
            return False
        except ValueError:
            pass  # Expected
        
        # Test invalid iterations
        try:
            kmeans = KMeans(n_clusters=2)
            kmeans.with_iterations(0)
            print("✗ Should have raised error for 0 iterations")
            return False
        except ValueError:
            pass  # Expected
        
        # Test invalid max_balance_diff
        try:
            kmeans = KMeans(n_clusters=2)
            kmeans.with_max_balance_diff(0)
            print("✗ Should have raised error for 0 max_balance_diff")
            return False
        except ValueError:
            pass  # Expected
        
        # Test training with insufficient data
        try:
            kmeans = KMeans(n_clusters=10)
            data = np.random.rand(5, 2).astype(np.float32)
            kmeans.train(data, num_threads=None)
            print("✗ Should have raised error for insufficient data")
            return False
        except ValueError:
            pass  # Expected
        
        # Test assignment without training
        try:
            kmeans = KMeans(n_clusters=3)
            data = np.random.rand(10, 2).astype(np.float32)
            kmeans.assign(data, k=1)
            print("✗ Should have raised error for assignment without training")
            return False
        except ValueError:
            pass  # Expected
        
        print("✓ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_medoids_functionality():
    """Test medoids-specific functionality."""
    try:
        from kentro import KMeans
        
        # Create test data
        np.random.seed(42)
        data = np.random.rand(20, 2).astype(np.float32)
        
        # Test K-medoids
        kmeans = KMeans(n_clusters=3).with_use_medoids(True).with_euclidean(True)
        clusters = kmeans.train(data, num_threads=None)
        
        assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}"
        assert kmeans.is_use_medoids, "Should use medoids"
        assert kmeans.medoid_indices is not None, "Medoid indices should be available"
        assert len(kmeans.medoid_indices) == 3, f"Expected 3 medoid indices, got {len(kmeans.medoid_indices)}"
        
        # Check that medoid indices are valid
        for idx in kmeans.medoid_indices:
            assert 0 <= idx < len(data), f"Invalid medoid index: {idx}"
        
        print("✓ Medoids functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Medoids functionality test failed: {e}")
        return False


def test_assignment_functionality():
    """Test cluster assignment functionality."""
    try:
        from kentro import KMeans
        
        # Create and train model
        np.random.seed(42)
        train_data = np.random.rand(20, 2).astype(np.float32)
        kmeans = KMeans(n_clusters=3)
        kmeans.train(train_data, num_threads=None)
        
        # Test assignment
        test_data = np.random.rand(10, 2).astype(np.float32)
        assignments = kmeans.assign(test_data, k=1)
        
        assert len(assignments) == 3, f"Expected 3 assignment groups, got {len(assignments)}"
        
        # Check that each test point is assigned to exactly one cluster
        total_assignments = sum(len(cluster) for cluster in assignments)
        assert total_assignments == len(test_data), f"Expected {len(test_data)} total assignments, got {total_assignments}"
        
        print("✓ Assignment functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Assignment functionality test failed: {e}")
        return False


def test_string_representation():
    """Test string representation."""
    try:
        from kentro import KMeans
        
        kmeans = KMeans(n_clusters=3).with_euclidean(True)
        repr_str = str(kmeans)
        
        assert "KMeans" in repr_str, "String representation should contain 'KMeans'"
        assert "n_clusters=3" in repr_str, "String representation should contain cluster count"
        assert "euclidean=true" in repr_str, "String representation should contain euclidean setting"
        
        print("✓ String representation test passed")
        return True
        
    except Exception as e:
        print(f"✗ String representation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Kentro K-Means Python Bindings")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_configuration_methods,
        test_error_handling,
        test_medoids_functionality,
        test_assignment_functionality,
        test_string_representation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)


if __name__ == "__main__":
    main() 