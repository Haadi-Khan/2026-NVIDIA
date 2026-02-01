"""PCE (Pauli Correlation Encoding) Solver Package.

A quantum-enhanced optimization solver for the Low Autocorrelation Binary
Sequences (LABS) problem using variational quantum circuits.
"""

from .constants import OPTIMAL_ENERGIES, GPU_AVAILABLE
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
    # Constants
    "OPTIMAL_ENERGIES",
    "GPU_AVAILABLE",
]
