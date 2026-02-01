"""Local search algorithms for post-processing optimization."""

import numpy as np
from .energy import labs_energy


def greedy_local_search(sequence: np.ndarray, max_passes: int = 10) -> tuple:
    """Greedy bit-flip local search. Guaranteed to not increase energy.

    Iteratively flips each bit and keeps the flip if it reduces energy.
    Continues until no improvement is found or max_passes is reached.

    Args:
        sequence: Binary sequence (+1/-1) to refine
        max_passes: Maximum number of full passes over all bits

    Returns:
        Tuple of (refined_sequence, final_energy)
    """
    current = sequence.copy()
    current_energy = labs_energy(current)

    for _ in range(max_passes):
        improved = False
        for i in range(len(current)):
            current[i] *= -1
            new_energy = labs_energy(current)
            if new_energy < current_energy:
                current_energy = new_energy
                improved = True
            else:
                current[i] *= -1  # revert
        if not improved:
            break

    return current, current_energy


def tabu_search(
    sequence: np.ndarray, tenure: int = None, max_iterations: int = 5000
) -> tuple:
    """Tabu search with aspiration criterion.

    Uses a memory structure to prevent revisiting recent moves, enabling
    escape from local minima. Aspiration criterion allows tabu moves if
    they improve the global best.

    Args:
        sequence: Binary sequence (+1/-1) to refine
        tenure: Number of iterations a move stays tabu (default: max(7, N//4))
        max_iterations: Maximum number of iterations

    Returns:
        Tuple of (best_sequence, best_energy)
    """
    N = len(sequence)
    tenure = tenure or max(7, N // 4)

    current = sequence.copy()
    best = current.copy()
    best_energy = labs_energy(best)
    current_energy = best_energy
    tabu_list = {}  # position -> iteration when freed

    for iteration in range(max_iterations):
        best_move, best_move_energy = None, float("inf")

        for i in range(N):
            # Check tabu status (with aspiration)
            is_tabu = tabu_list.get(i, 0) > iteration
            current[i] *= -1
            new_energy = labs_energy(current)

            # Accept if: not tabu OR aspiration (beats global best)
            if not is_tabu or new_energy < best_energy:
                if new_energy < best_move_energy:
                    best_move, best_move_energy = i, new_energy
            current[i] *= -1

        if best_move is None:
            break

        # Make the move
        current[best_move] *= -1
        current_energy = best_move_energy
        tabu_list[best_move] = iteration + tenure

        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy

    return best, best_energy
