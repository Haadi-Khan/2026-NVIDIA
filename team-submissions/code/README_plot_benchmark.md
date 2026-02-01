# plot_benchmark.py - Large-Scale Parallel Benchmarking

Runs large-scale parallel benchmarks of the PCE solver across multiple problem sizes and generates comparison plots against baseline methods.

## Overview

This script:
1. Runs the PCE solver on multiple N values in parallel
2. Collects multiple samples per N for statistical analysis
3. Compares results against baseline exhaustive search
4. Generates publication-quality plots showing merit factor vs N

## Quick Start

```bash
# Run benchmark on default N values with 5 samples each
python plot_benchmark.py --n-samples 5 --n-workers 16

# Custom N values
python plot_benchmark.py --ns 10 15 20 25 30 --n-samples 3

# Full benchmark with custom solver configuration
python plot_benchmark.py --ns 5 10 15 20 25 30 35 40 \
  --n-samples 10 \
  --n-qubits 14 \
  --n-layers 10 \
  --n-restarts 50 \
  --maxiter 300 \
  --n-workers 32
```

## Command-Line Arguments

### Benchmark Configuration
- `--ns` - Specific N values to test (default: `[5, 10, 15, 20, ..., 82]`)
- `--n-samples` - Number of samples per N (default: 5)
- `--n-workers` - Number of parallel workers (default: 16)

### Solver Configuration
- `--n-qubits` - Number of qubits (default: 14)
- `--n-layers` - Number of ansatz layers (default: 10)
- `--n-restarts` - Restarts per optimization (default: 50)
- `--maxiter` - Max iterations per restart (default: 300)
- `--tabu-iterations` - Tabu search iterations (default: 20000)

### Output Files
- `--output-csv` - Output CSV file (default: `total_results.csv`)
- `--output-plot` - Output plot file (default: `benchmark_plot.png`)
- `--base-csv` - Baseline CSV with exhaustive search results (default: `merged_results.csv`)

## Output Files

### 1. Results CSV (`total_results.csv`)
Contains all benchmark results:
```csv
N,MF_Exhaustive,MF_PCE
10,6.250,6.250
10,6.250,6.250
15,8.125,7.890
15,8.125,8.125
...
```

### 2. Comparison Plot (`benchmark_plot.png`)
Shows:
- Blue line: Benchmark exhaustive search (optimal)
- Orange points: PCE solver results (median ± min/max error bars)
- X-axis: Sequence length N
- Y-axis: Merit factor (higher is better)

## Usage Examples

### Quick Test
```bash
# Test on small problems
python plot_benchmark.py --ns 10 15 20 --n-samples 3 --n-workers 4
```

### Production Benchmark
```bash
# Comprehensive benchmark with high-quality solver configuration
python plot_benchmark.py \
  --ns 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 \
  --n-samples 10 \
  --n-qubits 14 \
  --n-layers 10 \
  --n-restarts 100 \
  --maxiter 500 \
  --tabu-iterations 50000 \
  --n-workers 64 \
  --output-csv production_results.csv \
  --output-plot production_plot.png
```

### Scaling Study
```bash
# How does performance scale with N?
python plot_benchmark.py \
  --ns 10 20 30 40 50 60 70 80 \
  --n-samples 20 \
  --n-workers 32
```

### Resource-Constrained
```bash
# Limited workers, fewer samples
python plot_benchmark.py --ns 10 15 20 25 --n-samples 3 --n-workers 4
```

## Parallelization

The script uses `ProcessPoolExecutor` to run multiple optimizations in parallel:
- Each (N, sample) pair is an independent task
- Tasks are distributed across `n_workers` processes
- Progress is shown via `tqdm` progress bar

**Example**: 
- N values: `[10, 15, 20]`
- Samples: 5
- Total tasks: 3 × 5 = 15
- Workers: 8
- Expected speedup: ~8x vs sequential

## Performance Expectations

### Typical Runtime
| N Range | Samples | Workers | Expected Time |
|---------|---------|---------|---------------|
| 10-30   | 5       | 16      | 10-30 min     |
| 10-50   | 10      | 32      | 1-2 hours     |
| 10-80   | 10      | 64      | 3-6 hours     |

*Times depend on solver configuration and hardware*

### Memory Requirements
- Each worker needs ~2-4 GB RAM
- GPU memory: ~2-8 GB depending on qubit count
- Total RAM: `n_workers × 4 GB`

## Output Interpretation

### Summary Statistics
The script prints per-N statistics:
```
Summary statistics:
     median   min   max  count
N                              
10    6.250  6.250  6.250     5
15    8.125  7.890  8.125     5
20    9.524  8.889  9.524     5
```

- **median**: Typical performance
- **min/max**: Range of results (indicates consistency)
- **count**: Number of samples

### Plot Interpretation
- **Points on blue line**: PCE matched optimal
- **Points below blue line**: PCE suboptimal
- **Large error bars**: High variance (need more samples or better solver config)
- **Small error bars**: Consistent performance

## Solver Configuration Tips

### For Speed
```bash
--n-qubits 10 --n-layers 5 --n-restarts 20 --maxiter 100
```
- Faster but lower success rate
- Good for quick tests

### For Quality
```bash
--n-qubits 14 --n-layers 10 --n-restarts 100 --maxiter 500
```
- Slower but higher success rate
- Good for production benchmarks

### Balanced
```bash
--n-qubits 12 --n-layers 8 --n-restarts 50 --maxiter 300
```
- Good trade-off between speed and quality

## Baseline CSV Format

The `--base-csv` file should contain exhaustive search results:
```csv
N,MF_Exhaustive
10,6.250
15,8.125
20,9.524
...
```

This is used to plot the optimal baseline for comparison.

## Troubleshooting

### Out of Memory
- Reduce `--n-workers`
- Reduce `--n-qubits`
- Use a machine with more RAM

### Slow Progress
- Increase `--n-workers` (if you have CPU/RAM capacity)
- Reduce `--n-restarts` or `--maxiter`
- Test on smaller N values first

### Poor Results
- Increase `--n-qubits` and `--n-layers`
- Increase `--n-restarts`
- Increase `--tabu-iterations`
- Collect more `--n-samples` for better statistics

## Example Workflow

```bash
# 1. Quick test on small problems
python plot_benchmark.py --ns 10 15 --n-samples 2 --n-workers 4

# 2. Review results
cat total_results.csv
# Check plot: benchmark_plot.png

# 3. If results look good, scale up
python plot_benchmark.py \
  --ns 10 15 20 25 30 35 40 \
  --n-samples 10 \
  --n-workers 32 \
  --output-csv full_results.csv \
  --output-plot full_plot.png

# 4. Analyze summary statistics
# (printed at end of run)
```

## See Also

- [README_main.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_main.md) - Main CLI for single runs
- [README_benchmark_optimizers.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/README_benchmark_optimizers.md) - Optimizer comparison
- [src/README.md](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/README.md) - PCE solver module documentation
