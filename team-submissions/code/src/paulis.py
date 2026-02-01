"""Pauli string construction utilities for quantum circuit initialization."""

import numpy as np
from itertools import combinations, product
import cudaq


def build_kbody_paulis(n: int, k: int, N: int) -> list:
    """Build k-body Pauli strings for n qubits.

    Generates Pauli strings where exactly k qubits have non-identity operators.
    Prioritizes single Pauli types (all X, all Y, all Z) before mixing.

    Args:
        n: Number of qubits
        k: Number of non-identity Paulis per string
        N: Number of Pauli strings needed

    Returns:
        List of N Pauli strings (e.g., ['XXIII', 'YYIII', ...])
    """
    paulis = []
    indices = list(combinations(range(n), k))

    for pauli_type in ["X", "Y", "Z"]:
        for idx_set in indices:
            pauli = ["I"] * n
            for i in idx_set:
                pauli[i] = pauli_type
            paulis.append("".join(pauli))
            if len(paulis) >= N:
                return paulis

    if len(paulis) < N:
        all_paulis = [
            "".join(combo)
            for combo in product("IXYZ", repeat=n)
            if combo != tuple("I" * n)
        ]
        for p in all_paulis:
            if p not in paulis:
                paulis.append(p)
                if len(paulis) >= N:
                    break

    return paulis[:N]


def pauli_anticommutes(p1: str, p2: str) -> bool:
    """Check if two Pauli strings anticommute.

    Two Pauli strings anticommute if they differ in an odd number of
    non-identity positions where both are non-identity and different.

    Args:
        p1: First Pauli string (e.g., "XXIZ")
        p2: Second Pauli string (e.g., "ZYXI")

    Returns:
        True if the Pauli strings anticommute
    """
    count = sum(1 for a, b in zip(p1, p2) if a != "I" and b != "I" and a != b)
    return count % 2 == 1


def build_noncommuting_paulis(n: int, N: int, max_candidates: int = 500) -> list:
    """Build maximally non-commuting Pauli set (Pi_NC).

    Implements an optimized version of the algorithm from the PCE paper (Appendix 6):
    Select Paulis that anti-commute with the maximum number of
    already-selected Paulis, providing mutually unbiased measurements.

    Uses random sampling of candidates for efficiency with large qubit counts.

    Args:
        n: Number of qubits
        N: Number of Pauli strings needed
        max_candidates: Max candidates to evaluate per selection (for speed)

    Returns:
        List of N Pauli strings maximizing non-commutativity
    """
    # For small n, we can enumerate all Paulis
    # For large n, we sample randomly
    total_paulis = 4**n - 1

    if total_paulis <= max_candidates * 2:
        # Small enough to enumerate
        all_paulis = ["".join(p) for p in product("IXYZ", repeat=n) if set(p) != {"I"}]
        candidate_pool = all_paulis
    else:
        # Generate a diverse candidate pool via random sampling
        candidate_pool = set()
        pauli_chars = ["I", "X", "Y", "Z"]

        # Add structured Paulis (single-qubit, two-qubit patterns)
        for i in range(n):
            for p in ["X", "Y", "Z"]:
                pauli = ["I"] * n
                pauli[i] = p
                candidate_pool.add("".join(pauli))

        for i in range(n):
            for j in range(i + 1, n):
                for p1 in ["X", "Y", "Z"]:
                    for p2 in ["X", "Y", "Z"]:
                        pauli = ["I"] * n
                        pauli[i] = p1
                        pauli[j] = p2
                        candidate_pool.add("".join(pauli))

        # Add random Paulis to fill pool
        while len(candidate_pool) < max_candidates * 3:
            pauli = "".join(np.random.choice(pauli_chars) for _ in range(n))
            if set(pauli) != {"I"}:
                candidate_pool.add(pauli)

        candidate_pool = list(candidate_pool)

    selected = []
    selected_set = set()

    for _ in range(N):
        # Sample candidates if pool is large
        if len(candidate_pool) > max_candidates:
            candidates = np.random.choice(
                candidate_pool,
                size=min(max_candidates, len(candidate_pool)),
                replace=False,
            )
        else:
            candidates = candidate_pool

        best_pauli, best_score = None, -1

        for p in candidates:
            if p in selected_set:
                continue
            # Count anti-commutations with selected set
            score = sum(1 for s in selected if pauli_anticommutes(p, s))
            if score > best_score:
                best_score = score
                best_pauli = p

        if best_pauli:
            selected.append(best_pauli)
            selected_set.add(best_pauli)
        else:
            # Fallback: generate a random Pauli not yet selected
            for _ in range(100):
                pauli = "".join(
                    np.random.choice(["I", "X", "Y", "Z"]) for _ in range(n)
                )
                if set(pauli) != {"I"} and pauli not in selected_set:
                    selected.append(pauli)
                    selected_set.add(pauli)
                    break
            else:
                break

    return selected


def pauli_string_to_spin_op(pauli_str: str) -> cudaq.SpinOperator:
    """Convert a Pauli string like 'XXIIZ' to a cudaq.SpinOperator.

    The string is read left-to-right as qubit 0, 1, 2, ...

    Args:
        pauli_str: Pauli string (e.g., "XYZII")

    Returns:
        cudaq.SpinOperator representing the Pauli string
    """
    n = len(pauli_str)
    op = None

    for qubit_idx, pauli in enumerate(pauli_str):
        if pauli == "I":
            term = cudaq.spin.i(qubit_idx)
        elif pauli == "X":
            term = cudaq.spin.x(qubit_idx)
        elif pauli == "Y":
            term = cudaq.spin.y(qubit_idx)
        elif pauli == "Z":
            term = cudaq.spin.z(qubit_idx)
        else:
            raise ValueError(f"Unknown Pauli: {pauli}")

        if op is None:
            op = term
        else:
            op = op * term

    return op


def build_spin_operators(pauli_strings: list) -> list:
    """Build a list of cudaq.SpinOperator from Pauli strings.

    Args:
        pauli_strings: List of Pauli strings

    Returns:
        List of cudaq.SpinOperator objects
    """
    return [pauli_string_to_spin_op(ps) for ps in pauli_strings]
