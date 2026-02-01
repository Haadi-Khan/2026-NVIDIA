#!/usr/bin/env python3
"""PCE LABS Solver - Command Line Interface.

This is the main entry point for running the PCE (Pauli Correlation Encoding)
solver for the Low Autocorrelation Binary Sequences (LABS) problem.

Usage:
    python main.py --N 13 --qubits 10 --layers 5 --restarts 10 --verbose
"""

import argparse
import time

from src import (
    PCESolver,
    get_optimal_energy,
    merit_factor,
    GPU_AVAILABLE,
)


def run_single(
    N: int,
    n_qubits: int = 10,
    n_layers: int = 10,
    n_restarts: int = 50,
    maxiter: int = 100,
    method: str = "COBYLA",
    verbose: bool = True,
    n_parallel: int = 1,
    use_processes: bool = False,
    use_tabu: bool = True,
    tabu_iterations: int = 20000,
    use_noncommuting: bool = True,
):
    """Run a single LABS optimization.

    Args:
        N: Sequence length
        n_qubits: Number of qubits in the ansatz
        n_layers: Number of ansatz layers
        n_restarts: Number of optimization restarts
        maxiter: Max iterations per restart
        method: Scipy optimization method
        verbose: Show progress output
        n_parallel: Number of parallel workers
        use_processes: Use processes instead of threads
        use_tabu: Enable tabu search refinement
        tabu_iterations: Max tabu search iterations
        use_noncommuting: Use non-commuting Pauli set

    Returns:
        Tuple of (solver, elapsed_time, best_energy)
    """
    # Print configuration
    print("=" * 60)
    parallel_str = ""
    if n_parallel > 1:
        parallel_type = "procs" if use_processes else "threads"
        parallel_str = f", {n_parallel} {parallel_type}"
    print(f"PCE LABS Solver - N={N}{parallel_str}")
    print("=" * 60)
    print(f"GPU: {GPU_AVAILABLE}")
    print(f"Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"Restarts: {n_restarts}, MaxIter: {maxiter}, Method: {method}")
    print(
        f"Pauli set: {'Pi(NC) non-commuting' if use_noncommuting else 'Pi(C) k-body'}"
    )
    print(f"Tabu search: {'enabled' if use_tabu else 'disabled'}")

    # Initialize solver
    solver = PCESolver(
        N=N,
        n_qubits=n_qubits,
        n_layers=n_layers,
        use_noncommuting=use_noncommuting,
    )

    print(f"Params: {solver.n_params}, Alpha: {solver.alpha:.1f}, Beta: {solver.beta}")
    print(f"State dim: {2**n_qubits}")
    print()

    # Run optimization
    start = time.time()
    sequence, energy = solver.optimize(
        n_restarts=n_restarts,
        maxiter=maxiter,
        method=method,
        verbose=verbose,
        n_parallel=n_parallel,
        use_processes=use_processes,
        use_tabu=use_tabu,
        tabu_iterations=tabu_iterations,
    )
    elapsed = time.time() - start

    # Print results
    optimal = get_optimal_energy(N)

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
    """Parse command line arguments and run the solver."""
    parser = argparse.ArgumentParser(
        description="PCE LABS Solver - Quantum-enhanced optimization for LABS problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem parameters
    parser.add_argument("--N", type=int, default=13, help="Sequence length")

    # Ansatz parameters
    parser.add_argument("--qubits", type=int, default=10, help="Number of qubits")
    parser.add_argument("--layers", type=int, default=5, help="Number of ansatz layers")

    # Optimization parameters
    parser.add_argument(
        "--restarts", type=int, default=10, help="Number of optimization restarts"
    )
    parser.add_argument(
        "--maxiter", type=int, default=100, help="Max iterations per restart"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="COBYLA",
        help="Optimization method (COBYLA, Powell, Nelder-Mead, etc.)",
    )

    # Parallelization
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for restarts",
    )
    parser.add_argument(
        "--processes",
        action="store_true",
        help="Use processes instead of threads (bypasses GIL)",
    )

    # Post-processing
    parser.add_argument(
        "--no-tabu", action="store_true", help="Disable tabu search refinement"
    )
    parser.add_argument(
        "--tabu-iters", type=int, default=20000, help="Max tabu search iterations"
    )

    # Pauli set selection
    parser.add_argument(
        "--commuting",
        action="store_true",
        help="Use commuting k-body Paulis instead of non-commuting",
    )

    # Output
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    run_single(
        N=args.N,
        n_qubits=args.qubits,
        n_layers=args.layers,
        n_restarts=args.restarts,
        maxiter=args.maxiter,
        method=args.method,
        verbose=args.verbose,
        n_parallel=args.parallel,
        use_processes=args.processes,
        use_tabu=not args.no_tabu,
        tabu_iterations=args.tabu_iters,
        use_noncommuting=not args.commuting,
    )


if __name__ == "__main__":
    main()
