#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

from src import PCESolver, merit_factor, ENERGY_DATA


SOLVER_CONFIG = {
    "n_qubits": 14,
    "n_layers": 10,
    "method": "COBYLA",
    "n_restarts": 50,
    "maxiter": 300,
    "use_tabu": True,
    "tabu_iterations": 20000,
}

DEFAULT_NS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 66, 70, 75, 80, 82]


def run_single_task(args):
    N, sample_id, config = args
    try:
        solver = PCESolver(
            N=N,
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
        )
        sequence, energy = solver.optimize(
            method=config["method"],
            n_restarts=config["n_restarts"],
            maxiter=config["maxiter"],
            use_tabu=config["use_tabu"],
            tabu_iterations=config["tabu_iterations"],
            verbose=False,
        )
        mf = N * N / (2 * energy) if energy > 0 else float("inf")
        return {
            "N": N,
            "sample": sample_id,
            "MF_PCE": mf,
            "energy": energy,
            "calls": solver.call_count,
        }
    except Exception as e:
        return {
            "N": N,
            "sample": sample_id,
            "MF_PCE": None,
            "energy": None,
            "calls": 0,
            "error": str(e),
        }


def run_parallel_benchmarks(n_range, n_samples, config, n_workers=16):
    tasks = [(N, s, config) for N in n_range for s in range(n_samples)]
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_task, t): t for t in tasks}
        for future in tqdm(
            as_completed(futures), total=len(tasks), desc="Benchmarking"
        ):
            result = future.result()
            results.append(result)

    return results


def create_results_dataframe(results, base_csv_path):
    base_df = pd.read_csv(base_csv_path)

    exhaustive_data = base_df.groupby("N")["MF_Exhaustive"].first().reset_index()

    pce_df = pd.DataFrame(results)
    pce_df = pce_df[pce_df["MF_PCE"].notna()]

    all_rows = []
    for N in sorted(pce_df["N"].unique()):
        exhaustive_mf = exhaustive_data[exhaustive_data["N"] == N][
            "MF_Exhaustive"
        ].values
        exhaustive_mf = exhaustive_mf[0] if len(exhaustive_mf) > 0 else None

        n_samples = pce_df[pce_df["N"] == N]
        for _, row in n_samples.iterrows():
            all_rows.append(
                {
                    "N": N,
                    "MF_Exhaustive": exhaustive_mf,
                    "MF_PCE": row["MF_PCE"],
                }
            )

    return pd.DataFrame(all_rows)


def plot_comparison(df, output_path="benchmark_plot.png"):
    fig, ax = plt.subplots(figsize=(12, 6))

    exhaustive = df.groupby("N")["MF_Exhaustive"].first().reset_index()
    ax.plot(
        exhaustive["N"],
        exhaustive["MF_Exhaustive"],
        "b-",
        linewidth=2,
        label="Benchmark Exhaustive Search",
    )

    pce_stats = df.groupby("N")["MF_PCE"].agg(["median", "min", "max"]).reset_index()

    ax.errorbar(
        pce_stats["N"],
        pce_stats["median"],
        yerr=[
            pce_stats["median"] - pce_stats["min"],
            pce_stats["max"] - pce_stats["median"],
        ],
        fmt="o",
        color="orange",
        markersize=6,
        capsize=3,
        capthick=1,
        elinewidth=1,
        label="PCE Solver (median ± min/max)",
    )

    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Merit Factor (higher is better)", fontsize=12)
    ax.set_title("Merit Factor vs N (200k Loss Function Calls Budget)", fontsize=14)
    ax.legend(loc="upper right", title="Method")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PCE Solver and generate plots"
    )
    parser.add_argument(
        "--ns", type=int, nargs="+", default=None, help="Specific N values to test"
    )
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per N")
    parser.add_argument("--n-workers", type=int, default=16, help="Parallel workers")
    parser.add_argument(
        "--output-csv", type=str, default="total_results.csv", help="Output CSV"
    )
    parser.add_argument(
        "--output-plot", type=str, default="benchmark_plot.png", help="Output plot"
    )
    parser.add_argument(
        "--base-csv", type=str, default="merged_results.csv", help="Base CSV"
    )
    parser.add_argument("--n-qubits", type=int, default=14, help="Number of qubits")
    parser.add_argument("--n-layers", type=int, default=10, help="Number of layers")
    parser.add_argument(
        "--n-restarts", type=int, default=50, help="Restarts per optimization"
    )
    parser.add_argument(
        "--maxiter", type=int, default=300, help="Max iterations per restart"
    )
    parser.add_argument(
        "--tabu-iterations", type=int, default=20000, help="Tabu search iterations"
    )
    args = parser.parse_args()

    config = {
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "method": "COBYLA",
        "n_restarts": args.n_restarts,
        "maxiter": args.maxiter,
        "use_tabu": True,
        "tabu_iterations": args.tabu_iterations,
    }

    n_values = args.ns if args.ns else DEFAULT_NS
    total_tasks = len(n_values) * args.n_samples

    print("=" * 60)
    print("PCE Solver Benchmark")
    print("=" * 60)
    print(f"N values: {n_values}")
    print(f"Samples per N: {args.n_samples}")
    print(f"Total tasks: {total_tasks}")
    print(f"Workers: {args.n_workers}")
    print(f"Config: {config}")
    print("=" * 60)

    start_time = time.time()
    results = run_parallel_benchmarks(n_values, args.n_samples, config, args.n_workers)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.1f}m)")

    df = create_results_dataframe(results, args.base_csv)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

    plot_comparison(df, args.output_plot)

    print("\nSummary statistics:")
    stats = df.groupby("N")["MF_PCE"].agg(["median", "min", "max", "count"])
    print(stats.to_string())


if __name__ == "__main__":
    main()
