# main.py - PCE LABS Solver CLI

Command-line interface for running the PCE (Pauli Correlation Encoding) solver for the Low Autocorrelation Binary Sequences (LABS) problem.

## Quick Start

```bash
# Basic usage - solve LABS for N=13
python main.py --N 13 --qubits 10 --layers 5 --restarts 10

# With verbose output
python main.py --N 20 --qubits 12 --layers 8 --restarts 50 --verbose

# Parallel optimization with 4 workers
python main.py --N 25 --restarts 100 --parallel 4

# Use processes instead of threads (bypasses GIL)
python main.py --N 30 --restarts 100 --parallel 8 --processes
```

## Command-Line Arguments

### Problem Parameters
- `--N` - Sequence length (default: 13)

### Ansatz Parameters
- `--qubits` - Number of qubits in the quantum circuit (default: 10)
- `--layers` - Number of ansatz layers (default: 5)

### Optimization Parameters
- `--restarts` - Number of random restarts (default: 10)
- `--maxiter` - Maximum iterations per restart (default: 100)
- `--method` - Optimization method (default: COBYLA)
  - Options: `COBYLA`, `Powell`, `Nelder-Mead`, `L-BFGS-B`, `Adam`, `QNG`, `SPSA`, `SPSA-Adam`, `SPSA-CG`

### Parallelization
- `--parallel` - Number of parallel workers for restarts (default: 1)
- `--processes` - Use multiprocessing instead of threading (bypasses Python GIL)

### Post-Processing
- `--no-tabu` - Disable tabu search refinement
- `--tabu-iters` - Maximum tabu search iterations (default: 20000)

### Pauli Set Selection
- `--commuting` - Use commuting k-body Paulis instead of non-commuting set

### Output
- `--verbose` - Show detailed progress output

## Usage Examples

### Small Problem (N ≤ 20)
```bash
python main.py --N 15 --qubits 8 --layers 3 --restarts 20 --verbose
```

### Medium Problem (N ≤ 40)
```bash
python main.py --N 30 --qubits 10 --layers 5 --restarts 50 --maxiter 200
```

### Large Problem (N ≤ 80)
```bash
python main.py --N 65 --qubits 14 --layers 10 --restarts 50 --maxiter 300 \
  --parallel 8 --processes --tabu-iters 20000
```

### Optimizer Comparison
```bash
# COBYLA (derivative-free, baseline)
python main.py --N 20 --method COBYLA --restarts 20

# Adam (gradient-based with parameter-shift)
python main.py --N 20 --method Adam --restarts 20

# SPSA-Adam (SPSA gradients + Adam momentum, O(2) cost)
python main.py --N 20 --method SPSA-Adam --restarts 20
```

### Disable Tabu Refinement
```bash
python main.py --N 25 --no-tabu
```

### Use Commuting Paulis
```bash
python main.py --N 20 --commuting
```

## Output Format

The solver prints:
1. **Configuration** - Problem size, GPU status, hyperparameters
2. **Progress** - Optimization progress (if `--verbose`)
3. **Timing** - Breakdown of computation time
4. **Results** - Final energy, optimal energy, merit factor, success status

Example output:
```
============================================================
PCE LABS Solver - N=13
============================================================
GPU: True
Qubits: 10, Layers: 5
Restarts: 10, MaxIter: 100, Method: COBYLA
Pauli set: Pi(NC) non-commuting
Tabu search: enabled
Params: 50, Alpha: 0.0, Beta: 15.0
State dim: 1024

[Optimization progress...]

Timing breakdown:
  State preparation: 2.34s (45.2%)
  Expectation values: 1.89s (36.5%)
  Loss computation: 0.95s (18.3%)

============================================================
Energy: 8 (optimal: 8)
Match: YES
Merit: 10.5625
Total time: 5.18s
============================================================
```

## Optimization Tips

### Choosing Qubits and Layers
- **More qubits** = larger state space = more expressivity
- **More layers** = more parameters = better optimization landscape
- **Trade-off**: Larger circuits are slower to simulate
- **Rule of thumb**: Start with `qubits ≈ log₂(N) + 2` and `layers ≈ N/5`

### Choosing Restarts
- More restarts increase chance of finding optimal solution
- Diminishing returns after ~50-100 restarts
- Use `--parallel` to speed up multiple restarts

### Choosing Optimization Method
- **COBYLA**: Reliable baseline, no gradient computation
- **Adam**: Fast convergence with gradients, but O(2p) cost per step
- **SPSA-Adam**: Best of both worlds - O(2) gradient cost with Adam momentum
- **QNG**: Quantum natural gradient, good for quantum landscapes

### Tabu Search
- Always keep enabled unless debugging
- Refines quantum solutions classically
- Typically improves energy by 5-20%

### Parallelization
- Use `--parallel` with `--processes` for CPU-bound workloads
- Threads work well for GPU-bound workloads (less overhead)
- Optimal workers ≈ number of CPU cores

## Performance Expectations

| N Range | Qubits | Layers | Restarts | Expected Time | Success Rate |
|---------|--------|--------|----------|---------------|--------------|
| 5-15    | 8      | 3      | 10-20    | 10-30s        | ~90%         |
| 16-30   | 10     | 5      | 20-50    | 1-3min        | ~70%         |
| 31-50   | 12     | 8      | 50-100   | 5-15min       | ~50%         |
| 51-80   | 14     | 10     | 50-100   | 15-45min      | ~30%         |

*Times are approximate and depend on GPU hardware*

## See Also

- [src/README.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/README.md) - PCE solver module documentation
- [README_benchmark_optimizers.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_benchmark_optimizers.md) - Optimizer benchmarking
- [README_plot_benchmark.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_plot_benchmark.md) - Large-scale benchmarking
