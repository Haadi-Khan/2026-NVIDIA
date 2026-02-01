# benchmark_optimizers.py - Optimizer Comparison Tool

Benchmarks and compares different optimization methods for the PCE solver to identify the best optimizer for various problem sizes.

## Overview

This script systematically compares the performance of multiple optimization algorithms:

**Derivative-Free:**
- `COBYLA` - Constrained optimization baseline

**Gradient-Based (Parameter-Shift):**
- `L-BFGS-B` - Quasi-Newton method (O(2p) gradient cost)
- `Adam` - Adaptive momentum (O(2p) gradient cost)

**Gradient-Based (SPSA):**
- `SPSA` - Stochastic gradient estimation (O(2) gradient cost)
- `SPSA-Adam` - SPSA + Adam momentum
- `SPSA-CG` - SPSA + conjugate gradient
- `SPSA-CG-Wolfe` - SPSA + CG + Wolfe line search

**Quantum Natural Gradient:**
- `QNG` - Diagonal metric approximation (O(3p) gradient cost)

## Quick Start

```bash
# Compare COBYLA vs Adam on N=13
python benchmark_optimizers.py --methods COBYLA Adam --N 13

# Test multiple problem sizes
python benchmark_optimizers.py --methods COBYLA SPSA-Adam --N 10 15 20 --n-trials 3

# Comprehensive benchmark
python benchmark_optimizers.py --methods COBYLA Adam SPSA SPSA-Adam SPSA-CG \
  --N 10 15 20 25 --n-trials 5 --n-restarts 10 --verbose
```

## Command-Line Arguments

### Benchmark Configuration
- `--methods` - Optimizer methods to benchmark (default: `COBYLA Adam`)
  - Options: `COBYLA`, `Powell`, `Nelder-Mead`, `L-BFGS-B`, `Adam`, `QNG`, `SPSA`, `SPSA-Adam`, `SPSA-CG`, `SPSA-CG-Wolfe`
- `--N` - Sequence lengths to test (default: `[13]`)
- `--n-trials` - Number of trials for averaging (default: 1)

### Optimization Parameters
- `--n-restarts` - Number of restarts per trial (default: 5)
- `--maxiter` - Max iterations per restart (default: 50)

### Ansatz Configuration
- `--n-qubits` - Number of qubits (default: 8)
- `--n-layers` - Number of ansatz layers (default: 3)

### Output
- `-v, --verbose` - Print detailed trial-by-trial results

## Output Format

The script produces three types of output:

### 1. Per-Method Summary
```
COBYLA:
  Summary: time=12.45s, calls=2450, success=100%, best_E=8
```

### 2. Summary Table
```
   N |            COBYLA |              Adam |
----------------------------------------------
  10 |   5.2s 100%  1234 |   3.8s 100%  2468 |
  15 |  12.4s  80%  2456 |   8.9s  90%  4912 |
  20 |  25.1s  60%  3678 |  18.2s  70%  7356 |

Format: time success_rate calls
```

### 3. Speedup Analysis
```
SPEEDUP vs COBYLA (wall-clock time)

N = 10:
  COBYLA      : 1.00x
  Adam        : 1.37x
  SPSA-Adam   : 2.14x
```

## Usage Examples

### Quick Comparison
```bash
# Compare two optimizers on a single problem
python benchmark_optimizers.py --methods COBYLA Adam --N 13 --n-trials 3
```

### Scaling Study
```bash
# How do optimizers scale with problem size?
python benchmark_optimizers.py --methods COBYLA Adam SPSA-Adam \
  --N 10 15 20 25 30 --n-trials 3 --n-restarts 10
```

### Gradient Method Comparison
```bash
# Compare different gradient estimation methods
python benchmark_optimizers.py --methods Adam SPSA SPSA-Adam SPSA-CG \
  --N 15 --n-trials 5 --n-restarts 20 --verbose
```

### Large-Scale Benchmark
```bash
# Comprehensive comparison with statistical significance
python benchmark_optimizers.py \
  --methods COBYLA L-BFGS-B Adam QNG SPSA SPSA-Adam SPSA-CG SPSA-CG-Wolfe \
  --N 10 13 15 18 20 25 \
  --n-trials 10 \
  --n-restarts 20 \
  --maxiter 100 \
  --n-qubits 10 \
  --n-layers 5 \
  --verbose
```

## Interpreting Results

### Success Rate
- Percentage of trials that found the optimal energy
- Higher is better
- Indicates reliability of the optimizer

### Wall-Clock Time
- Total time including all restarts
- Lower is better
- Includes overhead (state preparation, gradient computation)

### Function Calls
- Number of objective function evaluations
- Lower is better (more sample-efficient)
- Gradient methods typically use more calls

### Speedup
- Relative to COBYLA baseline
- \>1.0x means faster than COBYLA
- Accounts for wall-clock time, not just function calls

## Optimization Method Trade-offs

| Method | Gradient Cost | Convergence | Sample Efficiency | Best For |
|--------|---------------|-------------|-------------------|----------|
| COBYLA | None | Moderate | High | Small problems, baseline |
| Adam | O(2p) | Fast | Low | When gradients are cheap |
| SPSA | O(2) | Moderate | Very High | Large parameter spaces |
| SPSA-Adam | O(2) | Fast | Very High | Best overall choice |
| SPSA-CG | O(2) | Fast | Very High | Smooth landscapes |
| QNG | O(3p) | Very Fast | Low | Quantum-specific problems |

**Key Insights:**
- **COBYLA**: Reliable but slow for large parameter spaces
- **Adam**: Fast convergence but expensive gradients (2p evaluations)
- **SPSA-Adam**: Best of both worlds - fast convergence + cheap gradients
- **QNG**: Excellent for quantum problems but highest per-step cost

## Performance Tips

1. **Start small**: Test on N=10-15 before scaling up
2. **Use multiple trials**: Variational optimization is stochastic
3. **Adjust restarts**: More restarts = higher success rate but longer time
4. **Match problem size to resources**: Larger N needs more qubits/layers

## Example Results

Typical speedups vs COBYLA (N=20, 10 qubits, 5 layers, 20 restarts):
- **Adam**: 1.2-1.5x (faster convergence, but expensive gradients)
- **SPSA**: 0.8-1.0x (cheap gradients, but slower convergence)
- **SPSA-Adam**: 1.5-2.0x (cheap gradients + fast convergence)
- **SPSA-CG**: 1.8-2.5x (best performance, but more complex)

## See Also

- [README_main.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_main.md) - Main CLI documentation
- [src/README.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/README.md) - PCE solver module documentation
- [README_plot_benchmark.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_plot_benchmark.md) - Large-scale parallel benchmarking
