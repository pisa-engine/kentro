#!/bin/bash

# Build script for Kentro Python bindings

set -e

echo "Building Kentro Python bindings..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin[patchelf]
fi

# Development build
echo "Building development version..."
maturin develop --features python

# Test the bindings
echo "Running tests..."
python test_python_bindings.py

# Run example
echo "Running example..."
python examples/python_example.py

echo "Build completed successfully!"
echo ""
echo "To create a release build:"
echo "  maturin build --release --features python"
echo ""
echo "To install the wheel:"
echo "  pip install target/wheels/kentro-*.whl" 