# real_benchmark.py - GPU Performance Benchmark

Benchmarks the GPU-accelerated PCE solver across multiple LABS problem sizes and saves timing results to JSON.

## Overview

This script:
1. Tests the GPU solver on problem sizes N = 5, 10, 15, 20, 25, 30, 35, 40
2. Uses adaptive qubit counts based on problem size
3. Times 3 objective function calls per problem size
4. Saves results to `gpu_times_real.json` for comparison with CPU

## Quick Start

```bash
python real_benchmark.py
```

## What It Does

For each problem size N:
1. **Creates GPU solver** with adaptive configuration:
   - Qubits: `max(4, min(8, N//4 + 3))`
   - Layers: 2
   - Pauli set: Commuting k-body

2. **Warmup**: Single objective call to initialize GPU

3. **Benchmark**: Times 3 objective function calls

4. **Reports**: Runtime in seconds and ms/call

## Output

### Console Output
```
==================================================
GPU BENCHMARK
==================================================
Importing GPU solver...
N=5... 0.0234s (7.80ms/call)
N=10... 0.0456s (15.20ms/call)
N=15... 0.0789s (26.30ms/call)
N=20... 0.1234s (41.13ms/call)
N=25... 0.1890s (63.00ms/call)
N=30... 0.2567s (85.57ms/call)
N=35... 0.3456s (115.20ms/call)
N=40... 0.4567s (152.23ms/call)

GPU Results: {5: 0.0234, 10: 0.0456, ...}
Saved to gpu_times_real.json
```

### JSON Output (`gpu_times_real.json`)
```json
{
  "5": 0.0234,
  "10": 0.0456,
  "15": 0.0789,
  "20": 0.1234,
  "25": 0.1890,
  "30": 0.2567,
  "35": 0.3456,
  "40": 0.4567
}
```

## Adaptive Configuration

The script uses adaptive qubit counts to balance expressivity and performance:

| N | Qubits | Reasoning |
|---|--------|-----------|
| 5 | 4 | Minimum viable |
| 10 | 5 | N//4 + 3 = 5 |
| 15 | 6 | N//4 + 3 = 6 |
| 20 | 7 | N//4 + 3 = 8, capped at 8 |
| 25+ | 8 | Capped at maximum |

This ensures:
- Small problems don't waste resources
- Large problems don't exceed GPU memory
- Consistent layer count (2) for fair comparison

## Modifying the Benchmark

### Change Problem Sizes
```python
sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Add more sizes
```

### Change Solver Configuration
```python
solver = GPUSolver(
    N=N, 
    n_qubits=10,              # Fixed qubit count
    n_layers=5,               # More layers
    use_noncommuting=True     # Non-commuting Paulis
)
```

### Change Number of Calls
```python
# Time 10 calls instead of 3
for _ in range(10):
    _ = solver.objective(np.random.uniform(-np.pi, np.pi, solver.n_params))
```

## Use Cases

1. **GPU verification**: Ensure GPU acceleration is working
2. **Performance profiling**: Measure GPU performance across problem sizes
3. **Comparison baseline**: Generate data for CPU vs GPU comparison
4. **Scaling analysis**: Understand how GPU performance scales with N

## Next Steps

After running this script:
1. Run `cpu_benchmark.py` to generate CPU comparison
2. View the generated plot: `gpu_vs_cpu_real.png`
3. Analyze speedup factors

## See Also

- [README_cpu_benchmark.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/team-submissions/code/README_cpu_benchmark.md) - CPU comparison benchmark
- [README_benchmark.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_benchmark.md) - Simple CUDA-Q benchmark
- [src/README.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/README.md) - PCE solver documentation
