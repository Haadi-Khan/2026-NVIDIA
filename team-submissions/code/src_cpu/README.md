# PCE Solver - CPU Version

A pure NumPy implementation of the PCE (Pauli Correlation Encoding) solver for the LABS problem.

## Overview

This is a CPU-only version of the PCE solver that runs entirely on NumPy, without requiring CUDA-Q or CuPy. It's useful for:
- Running benchmarks on systems without GPU
- Comparing CPU vs GPU performance
- Development and debugging

## Key Differences from GPU Version

| Component | GPU Version | CPU Version |
|-----------|-------------|-------------|
| State simulation | CUDA-Q | Pure NumPy |
| Array operations | CuPy | NumPy |
| GPU_AVAILABLE | True | False |
| Performance | Fast (GPU accelerated) | Slower (CPU only) |

## Usage

```python
from src_cpu import PCESolver, get_optimal_energy, merit_factor

# Initialize solver (same API as GPU version)
solver = PCESolver(
    N=13,                      # Sequence length
    n_qubits=10,              # Number of qubits
    n_layers=5,               # Ansatz depth
    use_noncommuting=True     # Use non-commuting Paulis
)

# Run optimization
sequence, energy = solver.optimize(
    n_restarts=50,
    maxiter=100,
    method='COBYLA',
    use_tabu=True,
    verbose=True
)

# Check results
optimal = get_optimal_energy(13)
print(f"Energy: {energy} (optimal: {optimal})")
```

## Module Structure

- **`solver.py`** - Main `PCESolver` class with all optimization methods
- **`circuit.py`** - Pure NumPy statevector simulator
- **`paulis.py`** - Pauli string generation (numpy-only)
- **`energy.py`** - LABS energy computation
- **`search.py`** - Greedy and tabu search algorithms
- **`constants.py`** - Optimal LABS energies database

## Supported Optimization Methods

All methods from the GPU version are supported:
- **Derivative-free**: COBYLA, Powell, Nelder-Mead
- **Gradient-based**: L-BFGS-B, Adam
- **SPSA variants**: SPSA, SPSA-Adam, SPSA-CG
- **QNG**: Quantum Natural Gradient with diagonal metric

## Performance

The CPU version is significantly slower than GPU for large qubit counts:

| Qubits | GPU (ms/call) | CPU (ms/call) | Speedup |
|--------|---------------|---------------|---------|
| 6 | ~1 | ~5 | 5x |
| 8 | ~2 | ~20 | 10x |
| 10 | ~5 | ~100 | 20x |
| 12 | ~10 | ~500 | 50x |

*Approximate values, actual speedup depends on hardware*

## See Also

- [README_cpu_benchmark.md](../README_cpu_benchmark.md) - CPU vs GPU benchmarking
- [README_real_benchmark.md](../README_real_benchmark.md) - GPU benchmark
