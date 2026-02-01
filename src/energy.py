"""LABS energy computation utilities."""

import numpy as np


def labs_energy(s: np.ndarray) -> int:
    """Compute the LABS (Low Autocorrelation Binary Sequences) energy.

    The energy is defined as: E = sum_{k=1}^{N-1} C_k^2
    where C_k = sum_{i=0}^{N-k-1} s[i] * s[i+k] is the autocorrelation at lag k.

    Args:
        s: Binary sequence (+1/-1) of length N

    Returns:
        Integer energy value (lower is better)
    """
    N = len(s)
    s = np.asarray(s, dtype=np.int32)
    energy = 0
    for k in range(1, N):
        C_k = np.dot(s[: N - k], s[k:])
        energy += C_k * C_k
    return int(energy)


def merit_factor(s: np.ndarray) -> float:
    """Compute the merit factor of a binary sequence.

    Merit factor F = N^2 / (2 * E) where E is the LABS energy.
    Higher merit factor indicates better sequence quality.

    Args:
        s: Binary sequence (+1/-1) of length N

    Returns:
        Merit factor (float), or inf if energy is 0
    """
    N = len(s)
    E = labs_energy(s)
    if E == 0:
        return float("inf")
    return N * N / (2 * E)
