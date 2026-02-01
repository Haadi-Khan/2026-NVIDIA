# benchmark.py - CUDA-Q Performance Benchmark

Simple performance benchmark for CUDA-Q state vector simulation on GPU.

## Overview

This script measures the raw performance of CUDA-Q's `get_state()` function for a simple parameterized quantum circuit. It's useful for:
- Verifying GPU acceleration is working
- Measuring state preparation overhead
- Comparing performance across different hardware

## What It Does

1. Defines a simple ansatz with:
   - Hadamard gates on all qubits
   - RY rotation layers
   - CNOT entanglement layers

2. Warms up the GPU (3 calls)

3. Benchmarks `cudaq.get_state()` over 100 calls

4. Reports average time per call

## Quick Start

```bash
python benchmark.py
```

## Expected Output

```
Qubits: 10, Layers: 5, Params: 50
State dim: 1024
Warmup done
get_state: 2.34ms per call (100 calls)
```

## Modifying the Benchmark

Edit the script to change:

```python
# Circuit configuration
n_qubits = 10      # Number of qubits (state dim = 2^n_qubits)
n_layers = 5       # Number of ansatz layers
n_params = n_layers * n_qubits  # Total parameters

# Benchmark configuration
n_calls = 100      # Number of calls to average
```

## Performance Expectations

Typical `get_state()` times on NVIDIA GPUs:

| Qubits | State Dim | GPU (A100) | GPU (V100) | CPU |
|--------|-----------|------------|------------|-----|
| 8      | 256       | ~0.5ms     | ~1ms       | ~2ms |
| 10     | 1,024     | ~1ms       | ~2ms       | ~8ms |
| 12     | 4,096     | ~2ms       | ~4ms       | ~30ms |
| 14     | 16,384    | ~5ms       | ~10ms      | ~120ms |
| 16     | 65,536    | ~15ms      | ~30ms      | ~500ms |

*Times are approximate and depend on circuit depth and hardware*

## Troubleshooting

### GPU Not Detected
If you see slow performance, check:
```python
import cudaq
print(cudaq.get_target())  # Should show "nvidia" or "nvidia-mgpu"
```

### Out of Memory
For large qubit counts (≥18), you may hit GPU memory limits:
- Reduce `n_qubits`
- Use a GPU with more memory
- Consider using MPS (Matrix Product State) simulation

## Use Cases

1. **Hardware verification**: Ensure GPU acceleration is working
2. **Performance profiling**: Identify bottlenecks in your workflow
3. **Hardware comparison**: Compare different GPUs or CPU vs GPU
4. **Scaling analysis**: Measure how performance scales with qubit count

## See Also

- [README_main.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_main.md) - Main PCE solver CLI
- [src/README.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/README.md) - PCE solver module documentation
