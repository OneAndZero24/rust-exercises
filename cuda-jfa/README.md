# CUDA Jump Flood Algorithm (JFA)

GPU-accelerated implementation of the Jump Flood Algorithm for computing Voronoi diagrams using CUDA and Rust.

## Overview

The Jump Flood Algorithm is a parallel algorithm for computing approximate Voronoi diagrams and distance transforms on the GPU. This implementation uses:
- **cudarc** - Rust bindings for CUDA
- **Custom CUDA kernels** - Written in `src/kernel.cu`
- **Voronoi visualization** - Colorizes regions based on nearest seed point

## Requirements

- Rust toolchain
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.5+ installed

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

## Configuration

Edit `src/main.rs` to modify:
- `n` - Number of seed points (default: 10)
- `size` - Image dimensions in pixels (default: 512x512)
- Block/grid dimensions for kernel launches

## Algorithm

The Jump Flood Algorithm works by:
1. Initializing seed points with unique IDs
2. Iteratively propagating nearest seed information with step sizes: n/2, n/4, n/8, ..., 1
3. Each iteration checks 8 neighbors at the current step distance
4. Uses ping-pong buffers to avoid read-write conflicts

## Output

- `output.png` - RGB image where each pixel is colored based on its nearest seed point
- Colors generated via HSV color space for maximum distinction

## Custom Kernels

CUDA kernels are in `src/kernel.cu` and compiled at runtime. You can modify the JFA logic directly in the `.cu` file.