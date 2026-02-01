"""PCE (Pauli Correlation Encoding) Solver Package - CPU Version.

A quantum-enhanced optimization solver for the Low Autocorrelation Binary
Sequences (LABS) problem using variational quantum circuits.
This version runs entirely on CPU using NumPy for state vector simulation.
"""

from .constants import (
    ENERGY_DATA,
    GPU_AVAILABLE,
    get_optimal_energy,
    get_metric_factor,
)
from .energy import labs_energy, merit_factor
from .search import greedy_local_search, tabu_search
from .solver import PCESolver

__all__ = [
    # Main solver
    "PCESolver",
    # Energy utilities
    "labs_energy",
    "merit_factor",
    # Search algorithms
    "greedy_local_search",
    "tabu_search",
    # Constants and helpers
    "ENERGY_DATA",
    "GPU_AVAILABLE",
    "get_optimal_energy",
    "get_metric_factor",
]
