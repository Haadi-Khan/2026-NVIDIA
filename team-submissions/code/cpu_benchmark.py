t#!/usr/bin/env python3
"""CPU benchmark."""
import numpy as np
import time
import json
import matplotlib.pyplot as plt

print("=" * 50)
print("CPU BENCHMARK")
print("=" * 50)

print("Importing CPU solver...")
from src_cpu import PCESolver as CPUSolver

cpu_times = {}
sizes = [5, 10, 15, 20, 25, 30, 35, 40]

for N in sizes:
    print(f"N={N}...", end=" ", flush=True)
    n_qubits = max(4, min(8, N//4 + 3))
    solver = CPUSolver(N=N, n_qubits=n_qubits, n_layers=2, use_noncommuting=False)
    
    # Warmup
    p = np.random.uniform(-np.pi, np.pi, solver.n_params)
    _ = solver.objective(p)
    
    # Time 3 calls
    start = time.perf_counter()
    for _ in range(3):
        _ = solver.objective(np.random.uniform(-np.pi, np.pi, solver.n_params))
    t = time.perf_counter() - start
    cpu_times[N] = t
    print(f"{t:.4f}s ({t/3*1000:.2f}ms/call)")

print(f"\nCPU Results: {cpu_times}")

# Load GPU results and create plot
with open('gpu_times_real.json', 'r') as f:
    gpu_times = json.load(f)

# Convert keys to int
gpu_times = {int(k): v for k, v in gpu_times.items()}

# Create comparison plot
sizes_list = sorted(gpu_times.keys())
gpu_vals = [gpu_times[n] for n in sizes_list]
cpu_vals = [cpu_times[n] for n in sizes_list]
speedups = [c/g for g, c in zip(gpu_vals, cpu_vals)]

print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"{'N':>4} {'GPU(s)':>10} {'CPU(s)':>10} {'Speedup':>10}")
for n, g, c, s in zip(sizes_list, gpu_vals, cpu_vals, speedups):
    print(f"{n:>4} {g:>10.4f} {c:>10.4f} {s:>9.1f}x")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Main title
fig.suptitle('PCE-MTS Solver: GPU vs CPU Performance', fontsize=13, fontweight='bold')

ax1.plot(sizes_list, gpu_vals, 'b-o', lw=2, ms=10, label='GPU')
ax1.plot(sizes_list, cpu_vals, 'r-s', lw=2, ms=10, label='CPU')
ax1.set_xlabel('LABS Instance Size (N)', fontsize=12)
ax1.set_ylabel('Runtime for 3 objective calls (s)', fontsize=12)
ax1.set_title('Runtime Comparison', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

colors = ['#27ae60' if s >= 1 else '#e74c3c' for s in speedups]
bars = ax2.bar(sizes_list, speedups, color=colors, alpha=0.8, width=3)
ax2.axhline(y=1, color='k', ls='--', lw=1)
ax2.set_xlabel('LABS Instance Size (N)', fontsize=12)
ax2.set_ylabel('GPU Speedup (CPU/GPU)', fontsize=12)
ax2.set_title('GPU Speedup Factor', fontsize=12)
for bar, s in zip(bars, speedups):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{s:.0f}x', ha='center', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gpu_vs_cpu_real.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to gpu_vs_cpu_real.png")
