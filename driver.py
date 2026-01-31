import argparse
import numpy as np
import time

from pce import (
    PCESolver, 
    OPTIMAL_ENERGIES, 
    labs_energy, 
    merit_factor,
    GPU_AVAILABLE,
    get_n_params
)


def run_single(N: int, n_qubits: int = 10, n_layers: int = 10, 
               n_restarts: int = 50, maxiter: int = 100, method: str = 'COBYLA',
               verbose: bool = True, use_optimized: bool = True):
    print("=" * 60)
    print(f"PCE LABS Solver - N={N}")
    print("=" * 60)
    print(f"GPU: {GPU_AVAILABLE}")
    print(f"Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"Restarts: {n_restarts}, MaxIter: {maxiter}, Method: {method}")
    print(f"Mode: {'SPARSE (optimized)' if use_optimized else 'DENSE (legacy)'}")
    
    solver = PCESolver(N=N, n_qubits=n_qubits, n_layers=n_layers, use_optimized=use_optimized)
    print(f"Params: {solver.n_params}, Alpha: {solver.alpha:.1f}, Beta: {solver.beta}")
    print(f"State dim: {2**n_qubits}")
    print()
    
    start = time.time()
    sequence, energy = solver.optimize(n_restarts=n_restarts, maxiter=maxiter, 
                                        method=method, verbose=verbose)
    elapsed = time.time() - start
    
    optimal = OPTIMAL_ENERGIES.get(N)
    
    print()
    solver.print_timing()
    print()
    print("=" * 60)
    print(f"Energy: {energy} (optimal: {optimal})")
    print(f"Match: {'YES' if energy == optimal else 'NO'}")
    if sequence is not None:
        print(f"Merit: {merit_factor(sequence):.4f}")
    print(f"Total time: {elapsed:.2f}s")
    print("=" * 60)
    
    return elapsed, solver.call_count


def run_benchmark(N: int, n_qubits: int = 10, n_layers: int = 5,
                  n_restarts: int = 5, maxiter: int = 50):
    """Run benchmark comparing optimized vs legacy implementations."""
    print("=" * 70)
    print(f"BENCHMARK: N={N}, qubits={n_qubits}, layers={n_layers}")
    print(f"Restarts: {n_restarts}, MaxIter: {maxiter}")
    print("=" * 70)
    print()
    
    # Run optimized version (sparse Pauli application)
    print(">>> Running SPARSE (optimized) version...")
    np.random.seed(42)
    solver_opt = PCESolver(N=N, n_qubits=n_qubits, n_layers=n_layers, use_optimized=True)
    
    t0 = time.time()
    seq_opt, energy_opt = solver_opt.optimize(n_restarts=n_restarts, maxiter=maxiter, 
                                               method='COBYLA', verbose=True)
    time_opt = time.time() - t0
    
    print()
    solver_opt.print_timing()
    print()
    
    # Run legacy version (dense matrices)
    print(">>> Running DENSE (legacy) version...")
    np.random.seed(42)
    solver_leg = PCESolver(N=N, n_qubits=n_qubits, n_layers=n_layers, use_optimized=False)
    
    t0 = time.time()
    seq_leg, energy_leg = solver_leg.optimize(n_restarts=n_restarts, maxiter=maxiter, 
                                               method='COBYLA', verbose=True)
    time_leg = time.time() - t0
    
    print()
    solver_leg.print_timing()
    print()
    
    # Summary
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Sparse:  {time_opt:.2f}s, {solver_opt.call_count} calls, energy={energy_opt}")
    print(f"Dense:   {time_leg:.2f}s, {solver_leg.call_count} calls, energy={energy_leg}")
    
    if time_opt > 0:
        speedup = time_leg / time_opt
        print(f"Speedup: {speedup:.2f}x {'(FASTER)' if speedup > 1 else '(SLOWER)'}")
    
    # Per-call comparison
    if solver_opt.call_count > 0 and solver_leg.call_count > 0:
        time_per_call_opt = 1000 * time_opt / solver_opt.call_count
        time_per_call_leg = 1000 * time_leg / solver_leg.call_count
        print(f"Per call: {time_per_call_opt:.2f}ms (sparse) vs {time_per_call_leg:.2f}ms (dense)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['single', 'verify', 'benchmark'])
    parser.add_argument('--N', type=int, default=7)
    parser.add_argument('--qubits', type=int, default=10)
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--restarts', type=int, default=10)
    parser.add_argument('--maxiter', type=int, default=100)
    parser.add_argument('--method', type=str, default='COBYLA')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--legacy', action='store_true', help='Use legacy dense matrix method')
    args = parser.parse_args()
    
    if args.command == 'single':
        run_single(args.N, args.qubits, args.layers, args.restarts, args.maxiter, 
                   args.method, args.verbose, use_optimized=not args.legacy)
    elif args.command == 'benchmark':
        run_benchmark(args.N, args.qubits, args.layers, args.restarts, args.maxiter)


if __name__ == '__main__':
    main()
