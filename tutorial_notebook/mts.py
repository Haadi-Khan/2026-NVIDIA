import numpy as np

class LabsEnergyCounter:
    """Callable LABS energy evaluator that counts how many times it's been called."""
    def __init__(self, warn_at: int | None = 200_000):
        self.count = 0
        self.warn_at = warn_at

    def reset(self) -> None:
        self.count = 0

    def __call__(self, s):
        """
        Compute the LABS energy function E(s) = sum_{k=1}^{N-1} C_k^2
        where C_k = sum_{i=1}^{N-k} s_i * s_{i+k}

        Args:
            s: Binary sequence with values in {-1, +1}

        Returns:
            Energy value (lower is better)
        """
        N = len(s)
        energy = 0
        for k in range(1, N):
            C_k = sum(s[i] * s[i + k] for i in range(N - k))
            energy += C_k ** 2

        self.count += 1
        if self.warn_at is not None and self.count == self.warn_at:
            print(f"Reached {self.warn_at:,} energy evaluations, exiting.")
        return energy

def merit_factor(s, E) -> float:
    """
    Compute the merit factor F = N^2 / (2 * E).
    """
    N = len(s)
    if E == 0:
        return float('inf')
    return N * N / (2 * E)

def combine(p1, p2):
    """
    Single-point crossover of two parent sequences.

    Args:
        p1: First parent sequence
        p2: Second parent sequence

    Returns:
        Child sequence combining p1[0:k] and p2[k:N]
    """
    N = len(p1)
    k = np.random.randint(1, N)  # cut point in {1, ..., N-1}
    return np.concatenate([p1[:k], p2[k:]])

def mutate(s, p_mut):
    """
    Probabilistic bit-flipping mutation.

    Args:
        s: Input sequence
        p_mut: Probability of flipping each bit

    Returns:
        Mutated sequence
    """
    s = s.copy()
    for i in range(len(s)):
        if np.random.random() < p_mut:
            s[i] = -s[i]  # flip +1 <-> -1
    return s

def tabu_search(s, energy_func, max_iter=100, tabu_tenure=7):
    """
    Tabu search: a modified greedy local search that maintains a tabu list
    to avoid cycling back to recently visited solutions.

    Args:
        s: Starting sequence
        energy_func: Callable to evaluate energy
        max_iter: Maximum number of iterations
        tabu_tenure: Number of iterations a move stays tabu

    Returns:
        best_s: Best sequence found
        best_energy: Energy of best sequence
    """
    s = s.copy()
    best_s = s.copy()
    best_energy = energy_func(s)
    tabu_list = {}  # position -> iteration when it becomes non-tabu

    for iteration in range(max_iter):
        # Find best non-tabu neighbor (single bit flip)
        best_neighbor = None
        best_neighbor_energy = float('inf')
        best_flip_pos = -1

        for i in range(len(s)):
            # Check if move is tabu (unless it leads to aspiration criterion)
            is_tabu = tabu_list.get(i, 0) > iteration

            # Compute neighbor energy
            neighbor = s.copy()
            neighbor[i] = -neighbor[i]
            neighbor_energy = energy_func(neighbor)

            # Accept if not tabu, or if it beats the best (aspiration criterion)
            if (not is_tabu) or (neighbor_energy < best_energy):
                if neighbor_energy < best_neighbor_energy:
                    best_neighbor = neighbor
                    best_neighbor_energy = neighbor_energy
                    best_flip_pos = i

        if best_neighbor is None:
            break

        # Move to best neighbor
        s = best_neighbor
        tabu_list[best_flip_pos] = iteration + tabu_tenure

        # Update global best if improved
        if best_neighbor_energy < best_energy:
            best_s = s.copy()
            best_energy = best_neighbor_energy

    return best_s, best_energy

def memetic_tabu_search(N, energy_func, pop_size=10, max_generations=100,
                        p_mut=0.1, p_combine=0.5, target_energy=None,
                        tabu_max_iter=100, tabu_tenure=7, verbose=False,
                        initial_population=None):
    """
    Memetic Tabu Search (MTS) for the LABS problem.

    Args:
        N: Sequence length
        energy_func: LabsEnergyCounter instance or similar callable
        pop_size: Population size
        max_generations: Maximum number of generations
        p_mut: Mutation probability per bit
        p_combine: Probability of combining vs sampling
        target_energy: Stop if this energy is reached (optional)
        tabu_max_iter: Max iterations for tabu search
        tabu_tenure: Tabu tenure for tabu search
        verbose: Print progress if True
        initial_population: Optional list of initial sequences

    Returns:
        best_s: Best sequence found
        best_energy: Energy of best sequence
        population: Final population
        energies: Energies of final population
    """
    # Initialize population with values in {-1, +1}
    if initial_population is not None:
        population = [np.array(s) for s in initial_population[:pop_size]]
        # Fill remaining slots with random if needed
        while len(population) < pop_size:
            population.append(np.random.choice([-1, 1], size=N))
    else:
        population = [np.random.choice([-1, 1], size=N) for _ in range(pop_size)]

    energies = [energy_func(s) for s in population]

    # Find best solution in initial population
    best_idx = np.argmin(energies)
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]

    if verbose:
        print(f"Initial best energy: {best_energy}")

    for gen in range(max_generations):
        # Check stopping criterion
        if target_energy is not None and best_energy <= target_energy:
            if verbose:
                print(f"Target energy reached at generation {gen}")
            break

        # Make child: combine or sample
        if np.random.random() < p_combine and pop_size >= 2:
            # Combine two parents
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            child = combine(population[idx1], population[idx2])
        else:
            # Sample from population
            idx = np.random.randint(pop_size)
            child = population[idx].copy()

        # Mutate child
        child = mutate(child, p_mut)

        # Run tabu search
        result, result_energy = tabu_search(child, energy_func, max_iter=tabu_max_iter,
                                             tabu_tenure=tabu_tenure)
        
        if hasattr(energy_func, 'count') and hasattr(energy_func, 'warn_at') and energy_func.warn_at is not None:
            if energy_func.count >= energy_func.warn_at:
                if verbose:
                    print("Reached maximum allowed energy evaluations, stopping.")
                break

        # Update global best if improved
        better = result_energy < best_energy
        if better:
            best_s = result.copy()
            best_energy = result_energy

        if better or verbose:
            print(
                f"Generation {gen}: New best energy = {best_energy} "
                f"New merit factor = {merit_factor(best_s, best_energy)}, "
                f"labs_count= {getattr(energy_func, 'count', 'N/A')}",
                flush=True
            )

        # Add result to population (replace worst member if result is better)
        worst_idx = np.argmax(energies)
        if result_energy < energies[worst_idx]:
            population[worst_idx] = result
            energies[worst_idx] = result_energy

    return best_s, best_energy, population, energies
