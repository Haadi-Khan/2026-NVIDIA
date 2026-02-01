#!/usr/bin/env python3
"""Benchmark different optimizers for PCE solver.

Compares performance of:
- COBYLA (derivative-free, baseline)
- L-BFGS-B (gradient-based with parameter-shift, O(2p) per gradient)
- Adam (gradient-based with momentum, O(2p) per gradient)
- QNG (Quantum Natural Gradient with diagonal metric, O(3p) per step)
- SPSA (stochastic gradient estimation, O(2) per gradient)
- SPSA-Adam (SPSA + Adam momentum, O(2) per gradient)
- SPSA-CG (SPSA + Polak-Ribiere conjugate gradient, O(2) per gradient)
- SPSA-CG-Wolfe (SPSA + CG + Strong Wolfe line search, O(2+) per gradient)
"""

import time
import numpy as np
from src import PCESolver, get_optimal_energy


def benchmark_optimizer(
    N: int,
    method: str,
    n_restarts: int = 10,
    maxiter: int = 100,
    n_trials: int = 3,
    n_qubits: int = 10,
    n_layers: int = 5,
    verbose: bool = False,
) -> list:
    """Benchmark a single optimizer configuration.

    Args:
        N: Sequence length
        method: Optimization method ('COBYLA', 'L-BFGS-B', 'Adam', 'QNG')
        n_restarts: Number of random restarts per trial
        maxiter: Maximum iterations per restart
        n_trials: Number of trials for averaging
        n_qubits: Number of qubits in ansatz
        n_layers: Number of layers in ansatz
        verbose: Print progress

    Returns:
        List of result dictionaries
    """
    results = []

    for trial in range(n_trials):
        solver = PCESolver(N=N, n_qubits=n_qubits, n_layers=n_layers)

        start = time.time()
        sequence, energy = solver.optimize(
            n_restarts=n_restarts,
            maxiter=maxiter,
            method=method,
            verbose=False,
        )
        elapsed = time.time() - start

        optimal = get_optimal_energy(N)
        success = energy == optimal

        results.append(
            {
                "method": method,
                "N": N,
                "energy": energy,
                "optimal": optimal,
                "success": success,
                "time": elapsed,
                "calls": solver.call_count,
                "trial": trial,
            }
        )

        if verbose:
            status = "✓" if success else "✗"
            print(
                f"  Trial {trial + 1}: E={energy} (opt={optimal}) {status}, "
                f"time={elapsed:.2f}s, calls={solver.call_count}"
            )

    return results


def print_summary(all_results: list, methods: list, test_N: list):
    """Print summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'N':>4} | ", end="")
    for method in methods:
        print(f"{method:>18} |", end="")
    print()
    print("-" * (6 + 21 * len(methods)))

    for N in test_N:
        print(f"{N:>4} | ", end="")
        for method in methods:
            method_results = [
                r for r in all_results if r["N"] == N and r["method"] == method
            ]
            if method_results:
                avg_time = np.mean([r["time"] for r in method_results])
                avg_calls = np.mean([r["calls"] for r in method_results])
                success_rate = np.mean([r["success"] for r in method_results])
                print(
                    f"{avg_time:6.1f}s {success_rate:4.0%} {avg_calls:6.0f} |", end=""
                )
            else:
                print(f"{'---':>18} |", end="")
        print()

    print("\nFormat: time success_rate calls")


def main():
    """Run benchmark comparison of all optimizers."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark PCE solver optimizers")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["COBYLA", "Adam"],
        help="Optimizer methods to benchmark",
    )
    parser.add_argument(
        "--N",
        nargs="+",
        type=int,
        default=[13],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=5,
        help="Number of restarts per trial",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=50,
        help="Max iterations per restart",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of trials for averaging",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=8,
        help="Number of qubits",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of ansatz layers",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PCE Solver Optimizer Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Methods: {args.methods}")
    print(f"  N values: {args.N}")
    print(f"  Restarts: {args.n_restarts}")
    print(f"  Max iterations: {args.maxiter}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Qubits: {args.n_qubits}, Layers: {args.n_layers}")

    all_results = []

    for N in args.N:
        print(f"\n{'=' * 60}")
        print(f"N = {N} (optimal energy = {get_optimal_energy(N) or '?'})")
        print("=" * 60)

        for method in args.methods:
            print(f"\n{method}:")
            results = benchmark_optimizer(
                N,
                method,
                n_restarts=args.n_restarts,
                maxiter=args.maxiter,
                n_trials=args.n_trials,
                n_qubits=args.n_qubits,
                n_layers=args.n_layers,
                verbose=args.verbose,
            )
            all_results.extend(results)

            # Summary for this method
            avg_time = np.mean([r["time"] for r in results])
            avg_calls = np.mean([r["calls"] for r in results])
            success_rate = np.mean([r["success"] for r in results])
            best_energy = min(r["energy"] for r in results)

            print(
                f"  Summary: time={avg_time:.2f}s, calls={avg_calls:.0f}, "
                f"success={success_rate:.0%}, best_E={best_energy}"
            )

    # Print summary table
    print_summary(all_results, args.methods, args.N)

    # Speedup comparison relative to COBYLA
    print("\n" + "=" * 80)
    print("SPEEDUP vs COBYLA (wall-clock time)")
    print("=" * 80)

    for N in args.N:
        cobyla_results = [
            r for r in all_results if r["N"] == N and r["method"] == "COBYLA"
        ]
        if not cobyla_results:
            continue
        cobyla_time = np.mean([r["time"] for r in cobyla_results])

        print(f"\nN = {N}:")
        for method in args.methods:
            method_results = [
                r for r in all_results if r["N"] == N and r["method"] == method
            ]
            if method_results:
                method_time = np.mean([r["time"] for r in method_results])
                speedup = cobyla_time / method_time if method_time > 0 else 0
                print(f"  {method:12s}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
