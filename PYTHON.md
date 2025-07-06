# Kentro Python Bindings

Python bindings for the Kentro high-performance K-Means clustering library, implemented in Rust using PyO3.

## Features

- **Identical API**: The Python bindings expose the exact same API as the Rust library
- **High Performance**: Leverages Rust's performance with Python's ease of use
- **NumPy Integration**: Seamless integration with NumPy arrays
- **Method Chaining**: Fluent API for easy configuration
- **Comprehensive Error Handling**: Proper Python exceptions for all error conditions

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (if building from source)
- NumPy

### Install from PyPI (when available)

```bash
pip install kentro
```

### Build from Source

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Install maturin**:
   ```bash
   pip install maturin[patchelf]
   ```

3. **Build and install**:
   ```bash
   # Development build
   maturin develop --features python
   
   # Or production build
   maturin build --release --features python
   pip install target/wheels/kentro-*.whl
   ```

## Version Management

The Python package version is automatically synchronized with the Rust crate version defined in `Cargo.toml`. This ensures that both the Rust library and Python bindings always have the same version number.

- **Single Source of Truth**: Version is defined only in `Cargo.toml`
- **Automatic Synchronization**: Python package version is extracted from `Cargo.toml` during build
- **Runtime Access**: Python version is available via `kentro.__version__`

```python
import kentro
print(f"Kentro version: {kentro.__version__}")
```

## Quick Start

```python
import numpy as np
from kentro import KMeans

# Create sample data
data = np.random.rand(100, 2).astype(np.float32)

# Create and train K-Means
kmeans = KMeans(n_clusters=3)
clusters = kmeans.train(data, num_threads=None)

print(f"Found {len(clusters)} clusters")
print(f"Cluster sizes: {[len(c) for c in clusters]}")
```

## API Reference

### KMeans Class

#### Constructor

```python
KMeans(n_clusters: int)
```

Create a new K-Means instance.

**Parameters:**
- `n_clusters`: Number of clusters (must be positive)

**Raises:**
- `ValueError`: If n_clusters is 0

#### Configuration Methods (Method Chaining)

```python
with_iterations(iters: int) -> KMeans
```
Set the number of iterations (default: 25).

```python
with_euclidean(euclidean: bool) -> KMeans
```
Use Euclidean distance instead of cosine similarity (default: False).

```python
with_balanced(balanced: bool) -> KMeans
```
Enable balanced K-Means clustering (default: False).

```python
with_max_balance_diff(max_balance_diff: int) -> KMeans
```
Set maximum balance difference for balanced clustering (default: 16).

```python
with_verbose(verbose: bool) -> KMeans
```
Enable verbose output (default: False).

```python
with_use_medoids(use_medoids: bool) -> KMeans
```
Enable K-medoids clustering (default: False).

#### Training and Prediction

```python
train(data: np.ndarray, num_threads: Optional[int] = None) -> List[List[int]]
```

Perform K-Means clustering on the provided data.

**Parameters:**
- `data`: Data matrix (n_points × n_dimensions) as float32
- `num_threads`: Number of threads to use (None for automatic)

**Returns:**
- List of lists where each inner list contains indices of points assigned to the corresponding cluster

**Raises:**
- `ValueError`: If already trained, insufficient points, or dimension mismatch

```python
assign(data: np.ndarray, k: int) -> List[List[int]]
```

Assign data points to their k nearest clusters.

**Parameters:**
- `data`: Data matrix (n_points × n_dimensions) as float32
- `k`: Number of nearest clusters to assign each point to

**Returns:**
- List of lists where each inner list contains indices of points assigned to the corresponding cluster

**Raises:**
- `ValueError`: If not trained, k is 0, or dimension mismatch

#### Properties

```python
n_clusters: int                    # Number of clusters
iterations: int                    # Number of iterations
is_euclidean: bool                # Whether using Euclidean distance
is_balanced: bool                 # Whether using balanced clustering
is_use_medoids: bool              # Whether using medoids clustering
centroids: Optional[np.ndarray]   # Cluster centroids (n_clusters × n_dimensions)
medoid_indices: Optional[List[int]] # Medoid point indices (if using medoids)
is_trained: bool                  # Whether model has been trained
```

## Examples

### Basic K-Means

```python
import numpy as np
from kentro import KMeans

# Create sample data
np.random.seed(42)
data = np.random.rand(100, 2).astype(np.float32)

# Create and train K-Means
kmeans = KMeans(n_clusters=3)
clusters = kmeans.train(data, num_threads=None)

print(f"Found {len(clusters)} clusters")
print(f"Cluster sizes: {[len(c) for c in clusters]}")
print(f"Centroids:\n{kmeans.centroids}")
```

### Method Chaining Configuration

```python
kmeans = KMeans(n_clusters=3) \
    .with_iterations(50) \
    .with_euclidean(True) \
    .with_verbose(True)

clusters = kmeans.train(data, num_threads=None)
```

### K-Medoids Clustering

```python
# K-medoids finds actual data points as cluster centers
kmeans = KMeans(n_clusters=3) \
    .with_use_medoids(True) \
    .with_euclidean(True)

clusters = kmeans.train(data, num_threads=None)

# Get medoid indices
medoid_indices = kmeans.medoid_indices
if medoid_indices:
    print(f"Medoid indices: {medoid_indices}")
    print("Medoid points:")
    for i, idx in enumerate(medoid_indices):
        print(f"  Cluster {i}: {data[idx]}")
```

### Balanced K-Means

```python
# Balanced K-Means ensures clusters have similar sizes
kmeans = KMeans(n_clusters=3) \
    .with_balanced(True) \
    .with_max_balance_diff(5)

clusters = kmeans.train(data, num_threads=None)
print(f"Balanced cluster sizes: {[len(c) for c in clusters]}")
```

### Cluster Assignment

```python
# Train on training data
train_data = np.random.rand(100, 2).astype(np.float32)
kmeans = KMeans(n_clusters=3)
kmeans.train(train_data, num_threads=None)

# Assign new data to clusters
test_data = np.random.rand(20, 2).astype(np.float32)
assignments = kmeans.assign(test_data, k=1)

print("Assignment results:")
for i, cluster_points in enumerate(assignments):
    if cluster_points:
        print(f"  Cluster {i}: {cluster_points}")
```

### Euclidean vs Cosine Similarity

```python
# Cosine similarity (default) - good for high-dimensional data
kmeans_cosine = KMeans(n_clusters=3)
clusters_cosine = kmeans_cosine.train(data, num_threads=None)

# Euclidean distance - good for low-dimensional data
kmeans_euclidean = KMeans(n_clusters=3).with_euclidean(True)
clusters_euclidean = kmeans_euclidean.train(data, num_threads=None)
```

## Error Handling

The Python bindings provide proper error handling with descriptive error messages:

```python
try:
    # This will raise ValueError
    kmeans = KMeans(n_clusters=0)
except ValueError as e:
    print(f"Error: {e}")

try:
    # This will raise ValueError if not enough data
    kmeans = KMeans(n_clusters=10)
    small_data = np.random.rand(5, 2).astype(np.float32)
    kmeans.train(small_data, num_threads=None)
except ValueError as e:
    print(f"Error: {e}")
```

## Testing

Run the test suite:

```bash
python test_python_bindings.py
```

Run the comprehensive example:

```bash
python examples/python_example.py
```

## Performance Notes

- Always use `float32` NumPy arrays for optimal performance
- For large datasets, consider using `num_threads` parameter to control parallelization
- K-medoids is slower than standard K-means but provides actual data points as cluster centers
- Balanced K-means adds computational overhead but ensures more even cluster sizes

## Comparison with Rust API

The Python bindings provide an identical API to the Rust library:

| Rust | Python |
|------|--------|
| `KMeans::new(n_clusters)` | `KMeans(n_clusters)` |
| `with_iterations(25)` | `with_iterations(25)` |
| `with_euclidean(true)` | `with_euclidean(True)` |
| `train(data.view(), None)` | `train(data, num_threads=None)` |
| `assign(data.view(), k)` | `assign(data, k)` |
| `centroids()` | `centroids` (property) |
| `medoid_indices()` | `medoid_indices` (property) |

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 