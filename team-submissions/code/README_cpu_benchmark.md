# cpu_benchmark.py - CPU vs GPU Comparison Benchmark

Benchmarks the CPU-based PCE solver and compares performance against GPU results, generating a comprehensive comparison plot.

## Overview

This script:
1. Runs the CPU solver on the same problem sizes as `real_benchmark.py`
2. Loads GPU timing results from `gpu_times_real.json`
3. Computes speedup factors (CPU time / GPU time)
4. Generates a dual-panel comparison plot

## Prerequisites

**Must run first:**
```bash
python real_benchmark.py  # Generates gpu_times_real.json
```

**Then run:**
```bash
python cpu_benchmark.py
```

## Quick Start

```bash
# Complete workflow
python real_benchmark.py   # GPU benchmark
python cpu_benchmark.py    # CPU benchmark + comparison
```

## What It Does

### 1. CPU Benchmarking
For each problem size N = 5, 10, 15, 20, 25, 30, 35, 40:
- Creates CPU solver with same configuration as GPU version
- Performs warmup call
- Times 3 objective function calls
- Reports runtime

### 2. Comparison Analysis
- Loads GPU results from `gpu_times_real.json`
- Computes speedup: `CPU_time / GPU_time`
- Prints comparison table
- Generates visualization

### 3. Visualization
Creates `gpu_vs_cpu_real.png` with two panels:
- **Left**: Runtime comparison (GPU vs CPU lines)
- **Right**: Speedup bar chart (how many times faster GPU is)

## Output

### Console Output
```
==================================================
CPU BENCHMARK
==================================================
Importing CPU solver...
N=5... 0.1234s (41.13ms/call)
N=10... 0.2456s (81.87ms/call)
N=15... 0.4567s (152.23ms/call)
N=20... 0.8901s (296.70ms/call)
N=25... 1.4567s (485.57ms/call)
N=30... 2.3456s (781.87ms/call)
N=35... 3.6789s (1226.30ms/call)
N=40... 5.4321s (1810.70ms/call)

CPU Results: {5: 0.1234, 10: 0.2456, ...}

==================================================
COMPARISON
==================================================
   N    GPU(s)    CPU(s)   Speedup
   5    0.0234    0.1234       5.3x
  10    0.0456    0.2456       5.4x
  15    0.0789    0.4567       5.8x
  20    0.1234    0.8901       7.2x
  25    0.1890    1.4567       7.7x
  30    0.2567    2.3456       9.1x
  35    0.3456    3.6789      10.6x
  40    0.4567    5.4321      11.9x

Plot saved to gpu_vs_cpu_real.png
```

### Generated Plot (`gpu_vs_cpu_real.png`)

**Left Panel - Runtime Comparison:**
- Blue line with circles: GPU runtime
- Red line with squares: CPU runtime
- Shows absolute performance difference

**Right Panel - Speedup Factor:**
- Green bars: Speedup ≥ 1x (GPU faster)
- Red bars: Speedup < 1x (GPU slower)
- Dashed line at 1x: Break-even point
- Labels show exact speedup (e.g., "7x")

## Interpreting Results

### Typical Speedup Patterns
- **Small N (5-15)**: 3-6x speedup
  - GPU overhead dominates
  - Less benefit from parallelization
  
- **Medium N (20-35)**: 7-10x speedup
  - Sweet spot for GPU acceleration
  - Good balance of computation vs overhead
  
- **Large N (40+)**: 10-15x speedup
  - Maximum GPU benefit
  - Computation dominates overhead

### When GPU Wins
- ✅ Large state spaces (many qubits)
- ✅ Many Pauli expectations to compute
- ✅ Repeated objective calls
- ✅ Parallel expectation value computation

### When CPU Competitive
- ⚠️ Very small problems (N < 10)
- ⚠️ Single objective calls (warmup overhead)
- ⚠️ Few qubits (< 6)

## Modifying the Benchmark

### Test Different Configurations
```python
# Increase qubit count
n_qubits = 10  # Instead of adaptive

# More layers
solver = CPUSolver(N=N, n_qubits=n_qubits, n_layers=5)

# Non-commuting Paulis
solver = CPUSolver(N=N, n_qubits=n_qubits, n_layers=2, use_noncommuting=True)
```

### Add More Problem Sizes
```python
sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
```

### Change Number of Timing Calls
```python
# Time 10 calls instead of 3
for _ in range(10):
    _ = solver.objective(np.random.uniform(-np.pi, np.pi, solver.n_params))
```

## Use Cases

1. **Hardware selection**: Decide if GPU investment is worthwhile
2. **Performance validation**: Verify GPU acceleration is working correctly
3. **Scaling analysis**: Understand at what problem size GPU becomes beneficial
4. **Documentation**: Generate performance plots for reports/papers

## Troubleshooting

### FileNotFoundError: gpu_times_real.json
**Solution**: Run `real_benchmark.py` first
```bash
python real_benchmark.py
python cpu_benchmark.py
```

### Import Error: src_cpu
**Solution**: Ensure CPU solver module exists
- Check for `src_cpu/` directory
- Verify it contains PCESolver implementation

### Slow CPU Performance
**Expected**: CPU is intentionally slower than GPU
- This is what we're measuring!
- Typical speedups: 5-15x

## Performance Tips

### For Faster Benchmarking
- Reduce problem sizes: `sizes = [5, 10, 15, 20]`
- Fewer timing calls: `range(1)` instead of `range(3)`
- Fewer layers: `n_layers=1`

### For More Accurate Results
- More timing calls: `range(10)`
- Repeat entire benchmark multiple times
- Average results across runs

## See Also

- [README_real_benchmark.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/team-submissions/code/README_real_benchmark.md) - GPU benchmark (run this first)
- [README_benchmark.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_benchmark.md) - Simple CUDA-Q benchmark
- [src/README.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/README.md) - PCE solver documentation
