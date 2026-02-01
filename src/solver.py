"""PCE Solver - main optimization class."""

import numpy as np
from scipy.optimize import minimize
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import cudaq
import cupy as cp
from tqdm import tqdm

from .constants import OPTIMAL_ENERGIES
from .energy import labs_energy
from .search import greedy_local_search, tabu_search
from .paulis import build_kbody_paulis, build_noncommuting_paulis
from .circuit import ansatz, get_n_params


class PCESolver:
    """PCE Solver with GPU-native computation.

    Features:
    1. Keeps state on GPU - avoids np.array() conversion overhead
    2. Computes expectations entirely on GPU using CuPy
    3. Computes loss using hybrid GPU/CPU approach
    4. Supports parallel restarts for faster optimization
    """

    def __init__(
        self,
        N: int,
        n_qubits: int = 10,
        n_layers: int = 10,
        k: int = 2,
        alpha: float = 0.0,
        beta: float = 15.0,
        use_noncommuting: bool = True,
    ):
        """Initialize the PCE Solver.

        Args:
            N: Sequence length (number of Pauli expectations)
            n_qubits: Number of qubits in the ansatz
            n_layers: Number of layers in the ansatz
            k: k-body Pauli operators (only used if use_noncommuting=False)
            alpha: Scaling factor for tanh (default: 1.5 * n_qubits)
            beta: Regularization strength
            use_noncommuting: If True, use non-commuting Pauli set (Pi_NC) for
                              better expressivity. If False, use k-body commuting set.
        """
        self.N = N
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.k = k
        self.alpha = alpha if alpha > 0 else 1.5 * n_qubits
        self.beta = beta
        self.use_noncommuting = use_noncommuting
        self.n_params = get_n_params(n_qubits, n_layers)

        # Build Pauli strings
        if use_noncommuting:
            self.pauli_strings = build_noncommuting_paulis(n_qubits, N)
        else:
            self.pauli_strings = build_kbody_paulis(n_qubits, k, N)

        # Pre-compute Pauli action indices and phases on GPU
        self._precompute_pauli_actions_gpu()

        # Tracking best results
        self.best_params = None
        self.best_loss = float("inf")
        self.best_sequence = None
        self.best_energy = float("inf")

        # Performance tracking
        self.call_count = 0
        self.time_circuit = 0.0
        self.time_expectations = 0.0
        self.time_loss = 0.0

    def _precompute_pauli_actions_gpu(self):
        """Precompute Pauli action indices and phases directly on GPU."""
        n_paulis = len(self.pauli_strings)
        dim = 2**self.n_qubits

        # Compute on CPU first (small overhead, done once)
        target_indices = np.zeros((n_paulis, dim), dtype=np.int64)
        phases = np.zeros((n_paulis, dim), dtype=np.complex128)

        for p_idx, pauli_str in enumerate(self.pauli_strings):
            for j in range(dim):
                target, phase = self._apply_pauli_to_basis(pauli_str, j)
                target_indices[p_idx, j] = target
                phases[p_idx, j] = phase

        # Move to GPU and keep there
        self.target_indices_gpu = cp.asarray(target_indices)
        self.phases_gpu = cp.asarray(phases)

    def _apply_pauli_to_basis(self, pauli_str: str, basis_idx: int) -> tuple:
        """Apply Pauli string to a basis state: P|j⟩ = phase * |j'⟩."""
        target = basis_idx
        phase = 1.0 + 0.0j

        for qubit_idx, pauli in enumerate(pauli_str):
            bit = (basis_idx >> qubit_idx) & 1

            if pauli == "I":
                pass
            elif pauli == "X":
                target ^= 1 << qubit_idx
            elif pauli == "Y":
                target ^= 1 << qubit_idx
                phase *= 1j if bit == 0 else -1j
            elif pauli == "Z":
                if bit == 1:
                    phase *= -1

        return target, phase

    def get_state_gpu(self, params: np.ndarray):
        """Get state vector and immediately move to GPU."""
        state = cudaq.get_state(ansatz, self.n_qubits, list(params), self.n_layers)
        return cp.asarray(state, dtype=cp.complex128)

    def compute_expectations_gpu(self, state_gpu: cp.ndarray) -> cp.ndarray:
        """Compute all Pauli expectations entirely on GPU."""
        sv_conj = cp.conj(state_gpu)
        transformed = self.phases_gpu * state_gpu[self.target_indices_gpu]
        expectations = cp.real(cp.sum(sv_conj * transformed, axis=1))
        return expectations  # Keep on GPU!

    def compute_loss_gpu(self, expectations_gpu: cp.ndarray) -> float:
        """Compute loss - GPU tanh + CPU autocorrelation (hybrid approach).

        The LABS energy is: E = sum_{k=1}^{N-1} C_k^2
        where C_k = sum_{i=0}^{N-k-1} x[i] * x[i+k]

        For small-medium N (<500), CPU autocorrelation is faster than GPU
        due to kernel launch overhead. Only the N-element expectations array
        is transferred, which is negligible.
        """
        # Compute tanh on GPU
        x_tilde_gpu = cp.tanh(self.alpha * expectations_gpu)
        # Transfer small array (N elements) to CPU for autocorrelation
        x_tilde = cp.asnumpy(x_tilde_gpu)

        # CPU-based autocorrelation (fast for N < 500)
        loss = 0.0
        for l in range(1, self.N):
            C_l = np.dot(x_tilde[: self.N - l], x_tilde[l:])
            loss += C_l * C_l

        reg = -self.beta * np.dot(x_tilde, x_tilde)
        return loss + reg

    def objective(self, params: np.ndarray) -> float:
        """Compute the PCE loss function for given parameters."""
        self.call_count += 1

        # Get state (single circuit execution)
        t0 = time.perf_counter()
        state_gpu = self.get_state_gpu(params)
        self.time_circuit += time.perf_counter() - t0

        # Compute expectations on GPU
        t0 = time.perf_counter()
        expectations_gpu = self.compute_expectations_gpu(state_gpu)
        self.time_expectations += time.perf_counter() - t0

        # Compute loss on GPU
        t0 = time.perf_counter()
        loss = self.compute_loss_gpu(expectations_gpu)
        self.time_loss += time.perf_counter() - t0

        return loss

    def extract_sequence(self, params: np.ndarray, refine: bool = True) -> tuple:
        """Extract binary sequence from optimized parameters.

        Args:
            params: Optimized circuit parameters
            refine: If True, apply greedy local search to improve the sequence

        Returns:
            Tuple of (sequence, energy)
        """
        state_gpu = self.get_state_gpu(params)
        expectations_gpu = self.compute_expectations_gpu(state_gpu)
        expectations = cp.asnumpy(expectations_gpu)

        sequence = np.sign(expectations).astype(np.int8)
        sequence[sequence == 0] = 1

        if refine:
            sequence, energy = greedy_local_search(sequence)
            return sequence, energy
        return sequence, labs_energy(sequence)

    def print_timing(self):
        """Print timing breakdown of optimization calls."""
        total = self.time_circuit + self.time_expectations + self.time_loss
        if total == 0:
            print("No timing data available")
            return
        print(f"Timing breakdown ({self.call_count} calls):")
        print(
            f"  circuit:      {self.time_circuit:.2f}s ({100 * self.time_circuit / total:.1f}%)"
        )
        print(
            f"  expectations: {self.time_expectations:.2f}s ({100 * self.time_expectations / total:.1f}%)"
        )
        print(
            f"  loss:         {self.time_loss:.2f}s ({100 * self.time_loss / total:.1f}%)"
        )
        print(f"  per call:     {1000 * total / self.call_count:.2f}ms")

    def _single_restart(self, restart_id: int, maxiter: int, method: str) -> tuple:
        """Run a single optimization restart. Used for parallel execution."""
        params_init = np.random.uniform(-np.pi, np.pi, self.n_params)

        try:
            result = minimize(
                self.objective,
                params_init,
                method=method,
                options={"maxiter": maxiter, "disp": False},
            )
            sequence, energy = self.extract_sequence(result.x)
            return restart_id, sequence, energy, result.x, result.fun
        except Exception as e:
            return restart_id, None, float("inf"), None, float("inf")

    def optimize(
        self,
        n_restarts: int = 50,
        maxiter: int = 100,
        verbose: bool = False,
        method: str = "COBYLA",
        n_parallel: int = 1,
        use_processes: bool = False,
        use_tabu: bool = False,
        tabu_iterations: int = 5000,
    ) -> tuple:
        """Optimize with optional parallel restarts and tabu search refinement.

        Args:
            n_restarts: Number of random restarts
            maxiter: Max iterations per restart
            verbose: Show progress
            method: Optimization method (e.g., 'COBYLA', 'Powell', 'Nelder-Mead')
            n_parallel: Number of parallel restarts (1=sequential, >1=parallel)
            use_processes: Use ProcessPoolExecutor instead of threads (bypasses GIL)
            use_tabu: If True, apply tabu search refinement to best solution
            tabu_iterations: Maximum iterations for tabu search

        Returns:
            Tuple of (best_sequence, best_energy)
        """
        self._use_tabu = use_tabu
        self._tabu_iterations = tabu_iterations

        if n_parallel > 1:
            return self._optimize_parallel(
                n_restarts, maxiter, verbose, method, n_parallel, use_processes
            )

        # Sequential optimization
        pbar = (
            tqdm(range(n_restarts), desc="Restarts", unit="restart")
            if verbose
            else range(n_restarts)
        )

        for restart in pbar:
            params_init = np.random.uniform(-np.pi, np.pi, self.n_params)

            try:
                result = minimize(
                    self.objective,
                    params_init,
                    method=method,
                    options={"maxiter": maxiter, "disp": False},
                )

                sequence, energy = self.extract_sequence(result.x)

                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_sequence = sequence
                    self.best_params = result.x
                    self.best_loss = result.fun

                    if verbose:
                        pbar.set_postfix(
                            {"best_E": self.best_energy, "calls": self.call_count}
                        )

                    if energy == OPTIMAL_ENERGIES.get(self.N, -1):
                        if verbose:
                            pbar.set_description("Found optimal!")
                        break

            except Exception as e:
                continue

        # Apply tabu search refinement if requested
        if self._use_tabu and self.best_sequence is not None:
            if verbose:
                print(
                    f"Applying tabu search refinement (max {self._tabu_iterations} iterations)..."
                )
            pre_tabu_energy = self.best_energy
            self.best_sequence, self.best_energy = tabu_search(
                self.best_sequence, max_iterations=self._tabu_iterations
            )
            if verbose:
                print(f"Tabu search: {pre_tabu_energy} -> {self.best_energy}")

        return self.best_sequence, int(self.best_energy) if self.best_energy < float(
            "inf"
        ) else -1

    def _optimize_parallel(
        self,
        n_restarts: int,
        maxiter: int,
        verbose: bool,
        method: str,
        n_parallel: int,
        use_processes: bool = False,
    ) -> tuple:
        """Run parallel restarts using ThreadPool or ProcessPool."""

        completed = 0
        executor_type = "Procs" if use_processes else "Threads"
        pbar = (
            tqdm(
                total=n_restarts, desc=f"{executor_type}({n_parallel})", unit="restart"
            )
            if verbose
            else None
        )

        # Choose executor type
        Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        with Executor(max_workers=n_parallel) as executor:
            futures = []
            for i in range(n_restarts):
                future = executor.submit(self._single_restart, i, maxiter, method)
                futures.append(future)

            for future in as_completed(futures):
                restart_id, sequence, energy, params, loss = future.result()
                completed += 1

                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_sequence = sequence
                    self.best_params = params
                    self.best_loss = loss

                if verbose and pbar:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "best_E": self.best_energy
                            if self.best_energy < float("inf")
                            else "?",
                        }
                    )

        if pbar:
            pbar.close()

        # Apply tabu search refinement if requested
        if self._use_tabu and self.best_sequence is not None:
            if verbose:
                print(
                    f"Applying tabu search refinement (max {self._tabu_iterations} iterations)..."
                )
            pre_tabu_energy = self.best_energy
            self.best_sequence, self.best_energy = tabu_search(
                self.best_sequence, max_iterations=self._tabu_iterations
            )
            if verbose:
                print(f"Tabu search: {pre_tabu_energy} -> {self.best_energy}")

        return self.best_sequence, int(self.best_energy) if self.best_energy < float(
            "inf"
        ) else -1
