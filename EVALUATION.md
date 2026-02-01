# Evaluation

## Setup
This repository assumes you're on a system that supports Nvidia CUDA-Q and has the requisite CUDA tools configured.

Assuming you meet these requirements, install all the packages using `uv sync`

## Generating Energy Levels
Generate our optimizer's prediction for the optimal energy by running
`uv run main.py --N {value}`.

Example: `uv run main.py --N 5`

We validate predictions up to `N = 82`. Beyond that, our solutions do not display the optimal energy value.

To improve accuracy, you can configure the max iteration count using command line arguments.

To improve speed, you can enable parallel processes to maximize your CPU's ability to dispatch jobs.

## Testing

To test our code, run `uv run pytest`.

We have an auto-generated test suite from $N=2 \dots 11$. Tests usually take around 30s to complete