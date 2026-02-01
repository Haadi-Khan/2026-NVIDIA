#!/usr/bin/env python3
"""Real GPU benchmark with larger sizes."""
import numpy as np
import time
import json

print("=" * 50)
print("GPU BENCHMARK")
print("=" * 50)

print("Importing GPU solver...")
from src_gpu import PCESolver as GPUSolver

gpu_times = {}
sizes = [5, 10, 15, 20, 25, 30, 35, 40]

for N in sizes:
    print(f"N={N}...", end=" ", flush=True)
    n_qubits = max(4, min(8, N//4 + 3))
    solver = GPUSolver(N=N, n_qubits=n_qubits, n_layers=2, use_noncommuting=False)
    
    # Warmup
    p = np.random.uniform(-np.pi, np.pi, solver.n_params)
    _ = solver.objective(p)
    
    # Time 3 calls
    start = time.perf_counter()
    for _ in range(3):
        _ = solver.objective(np.random.uniform(-np.pi, np.pi, solver.n_params))
    t = time.perf_counter() - start
    gpu_times[N] = t
    print(f"{t:.4f}s ({t/3*1000:.2f}ms/call)")

with open('gpu_times_real.json', 'w') as f:
    json.dump(gpu_times, f)

print(f"\nGPU Results: {gpu_times}")
print("Saved to gpu_times_real.json")
