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
               verbose: bool = True):
    print("=" * 60)
    print(f"PCE LABS Solver - N={N}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['single', 'verify'])
    parser.add_argument('--N', type=int, default=7)
    parser.add_argument('--qubits', type=int, default=10)
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--restarts', type=int, default=10)
    parser.add_argument('--maxiter', type=int, default=100)
    parser.add_argument('--method', type=str, default='COBYLA')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    if args.command == 'single':
        run_single(args.N, args.qubits, args.layers, args.restarts, args.maxiter, 
                   args.method, args.verbose)


if __name__ == '__main__':
    main()
