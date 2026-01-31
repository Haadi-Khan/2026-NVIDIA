import argparse
import numpy as np
import time

from pce import (
    PCESolver, 
    OPTIMAL_ENERGIES, 
    labs_energy, 
    merit_factor,
    GPU_AVAILABLE
)


def run_single(N: int, n_qubits: int = 10, n_layers: int = 10, 
               n_restarts: int = 50, maxiter: int = 100, method: str = 'COBYLA',
               verbose: bool = True, n_parallel: int = 1, use_processes: bool = False):
    print("=" * 60)
    if n_parallel > 1:
        parallel_type = "procs" if use_processes else "threads"
        parallel_str = f", {n_parallel} {parallel_type}"
    else:
        parallel_str = ""
    print(f"PCE LABS Solver - N={N}{parallel_str}")
    print("=" * 60)
    print(f"GPU: {GPU_AVAILABLE}")
    print(f"Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"Restarts: {n_restarts}, MaxIter: {maxiter}, Method: {method}")
    
    solver = PCESolver(N=N, n_qubits=n_qubits, n_layers=n_layers)
    
    print(f"Params: {solver.n_params}, Alpha: {solver.alpha:.1f}, Beta: {solver.beta}")
    print(f"State dim: {2**n_qubits}")
    print()
    
    start = time.time()
    sequence, energy = solver.optimize(n_restarts=n_restarts, maxiter=maxiter, 
                                        method=method, verbose=verbose, 
                                        n_parallel=n_parallel, use_processes=use_processes)
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
    
    return solver, elapsed, energy


def main():
    parser = argparse.ArgumentParser(description="PCE LABS Solver Driver")
    parser.add_argument('--N', type=int, default=13, help="Sequence length")
    parser.add_argument('--qubits', type=int, default=10, help="Number of qubits")
    parser.add_argument('--layers', type=int, default=5, help="Number of ansatz layers")
    parser.add_argument('--restarts', type=int, default=10, help="Number of optimization restarts")
    parser.add_argument('--maxiter', type=int, default=100, help="Max iterations per restart")
    parser.add_argument('--method', type=str, default='COBYLA', help="Optimization method")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--parallel', type=int, default=1, help="Number of parallel workers for restarts")
    parser.add_argument('--processes', action='store_true', help="Use processes instead of threads (bypasses GIL, higher GPU util)")
    args = parser.parse_args()
    
    run_single(args.N, args.qubits, args.layers, args.restarts, args.maxiter, 
               args.method, args.verbose, args.parallel, args.processes)


if __name__ == '__main__':
    main()
