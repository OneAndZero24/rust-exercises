# Lorenz Attractor Simulator

A Rust implementation of the Lorenz attractor using the Runge-Kutta 4th order (RK4) numerical integration method with 3D visualization.

## Overview

The Lorenz system is a set of ordinary differential equations originally studied by Edward Lorenz as a simplified model of atmospheric convection:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

This implementation simulates the chaotic behavior of the system and generates 3D plots of the resulting attractor.


## Usage

### Basic usage

```bash
cargo run --release
```

This will generate `lorentz.png` with default parameters.

### Custom parameters

```bash
cargo run --release -- \
  -x 1.0 -y 1.0 -z 1.0 \
  --steps 10000 \
  --dt 0.01 \
  --sigma 10.0 \
  --r 28.0 \
  --b 2.667 \
  -o output.png
```

### CLI Arguments

- `-x`, `-y`, `-z`: Starting coordinates (default: 1.0, 1.0, 1.0)
- `--steps`: Number of integration steps (default: 10000)
- `--dt`: Time step size (default: 0.01)
- `--sigma`: Lorenz σ parameter (default: 10.0)
- `--r`: Lorenz ρ parameter (default: 28.0)
- `--b`: Lorenz β parameter (default: 8/3 ≈ 2.667)
- `-o`, `--output`: Output file path (default: "lorentz.png")

### Example configurations

**Classic chaotic attractor:**
```bash
cargo run --release -- --sigma 10 --r 28 --b 2.667
```

**Different initial conditions:**
```bash
cargo run --release -- -x 0.1 -y 0.0 -z 0.0
```

**Higher resolution:**
```bash
cargo run --release -- --steps 50000 --dt 0.005
```

## Building

```bash
cargo build --release
```