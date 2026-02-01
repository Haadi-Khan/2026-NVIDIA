"""PCE Solver - main optimization class - CPU Version.

This version uses pure NumPy for all computations, replacing CUDA-Q and CuPy
with a custom statevector simulator.
"""

import numpy as np
from scipy.optimize import minimize
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .constants import get_optimal_energy
from .energy import labs_energy
from .search import greedy_local_search, tabu_search
from .paulis import build_kbody_paulis, build_noncommuting_paulis
from .circuit import get_state, get_n_params


class PCESolver:
    """PCE Solver with CPU-native computation.

    Features:
    1. Pure NumPy statevector simulation
    2. Computes expectations using vectorized operations
    3. Computes loss using optimized autocorrelation
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

        # Pre-compute Pauli action indices and phases
        self._precompute_pauli_actions()

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

    def _precompute_pauli_actions(self):
        """Precompute Pauli action indices and phases."""
        n_paulis = len(self.pauli_strings)
        dim = 2**self.n_qubits

        self.target_indices = np.zeros((n_paulis, dim), dtype=np.int64)
        self.phases = np.zeros((n_paulis, dim), dtype=np.complex128)

        for p_idx, pauli_str in enumerate(self.pauli_strings):
            for j in range(dim):
                target, phase = self._apply_pauli_to_basis(pauli_str, j)
                self.target_indices[p_idx, j] = target
                self.phases[p_idx, j] = phase

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

    def get_state_cpu(self, params: np.ndarray) -> np.ndarray:
        """Get state vector using CPU simulator."""
        return get_state(self.n_qubits, list(params), self.n_layers)

    def compute_expectations(self, state: np.ndarray) -> np.ndarray:
        """Compute all Pauli expectations."""
        sv_conj = np.conj(state)
        transformed = self.phases * state[self.target_indices]
        expectations = np.real(np.sum(sv_conj * transformed, axis=1))
        return expectations

    def compute_loss(self, expectations: np.ndarray) -> float:
        """Compute loss from expectations.

        The LABS energy is: E = sum_{k=1}^{N-1} C_k^2
        where C_k = sum_{i=0}^{N-k-1} x[i] * x[i+k]
        """
        # Compute tanh
        x_tilde = np.tanh(self.alpha * expectations)

        # Autocorrelation-based loss
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
        state = self.get_state_cpu(params)
        self.time_circuit += time.perf_counter() - t0

        # Compute expectations
        t0 = time.perf_counter()
        expectations = self.compute_expectations(state)
        self.time_expectations += time.perf_counter() - t0

        # Compute loss
        t0 = time.perf_counter()
        loss = self.compute_loss(expectations)
        self.time_loss += time.perf_counter() - t0

        return loss

    # =====================================================================
    # Gradient-Based Optimization Methods
    # =====================================================================

    def compute_gradient_parameter_shift(self, params: np.ndarray) -> np.ndarray:
        """Compute gradient using the parameter-shift rule.

        For RY and RZ gates, the shift is π/2.
        Cost: 2 * n_params function evaluations.
        """
        grad = np.zeros(self.n_params)
        shift = np.pi / 2

        for j in range(self.n_params):
            # Forward shift
            params_plus = params.copy()
            params_plus[j] += shift
            loss_plus = self.objective(params_plus)

            # Backward shift
            params_minus = params.copy()
            params_minus[j] -= shift
            loss_minus = self.objective(params_minus)

            # Parameter-shift formula
            grad[j] = (loss_plus - loss_minus) / 2

        return grad

    def compute_gradient_spsa(
        self,
        params: np.ndarray,
        c: float = 0.1,
        seed: int = None,
    ) -> np.ndarray:
        """Estimate gradient using SPSA (2 function evaluations)."""
        rng = np.random.default_rng(seed)

        # Random perturbation direction (Rademacher distribution: ±1)
        delta = rng.choice([-1, 1], size=self.n_params).astype(np.float64)

        # Two function evaluations (regardless of n_params!)
        loss_plus = self.objective(params + c * delta)
        loss_minus = self.objective(params - c * delta)

        # SPSA gradient estimate
        grad = (loss_plus - loss_minus) / (2 * c * delta)

        return grad

    def optimize_spsa(
        self,
        params_init: np.ndarray,
        maxiter: int = 100,
        a: float = 0.1,
        c: float = 0.1,
        A: float = 10.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        verbose: bool = False,
    ) -> tuple:
        """Optimize using SPSA with decaying step sizes."""
        params = params_init.copy()
        best_loss = float("inf")
        best_params = params.copy()

        for k in range(1, maxiter + 1):
            a_k = a / (k + A) ** alpha
            c_k = c / k**gamma

            grad = self.compute_gradient_spsa(params, c=c_k)
            params = params - a_k * grad

            if k % 10 == 0:
                loss = self.objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                if verbose:
                    print(f"  SPSA iter {k}: loss = {loss:.6f}")

        final_loss = self.objective(params)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params.copy()

        return best_params, best_loss

    def optimize_spsa_adam(
        self,
        params_init: np.ndarray,
        maxiter: int = 100,
        learning_rate: float = 0.05,
        c: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        verbose: bool = False,
    ) -> tuple:
        """SPSA gradient estimation + Adam optimizer."""
        params = params_init.copy()
        m = np.zeros(self.n_params)
        v = np.zeros(self.n_params)

        best_loss = float("inf")
        best_params = params.copy()

        for t in range(1, maxiter + 1):
            grad = self.compute_gradient_spsa(params, c=c)

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            if t % 10 == 0:
                loss = self.objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                if verbose:
                    print(f"  SPSA-Adam iter {t}: loss = {loss:.6f}")

        final_loss = self.objective(params)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params.copy()

        return best_params, best_loss

    def optimize_spsa_cg(
        self,
        params_init: np.ndarray,
        maxiter: int = 100,
        learning_rate: float = 0.05,
        c: float = 0.1,
        restart_threshold: int = 0,
        verbose: bool = False,
    ) -> tuple:
        """SPSA with Polak-Ribiere conjugate gradient momentum."""
        params = params_init.copy()
        best_loss = float("inf")
        best_params = params.copy()

        grad_prev = None
        direction = None

        for t in range(1, maxiter + 1):
            grad = self.compute_gradient_spsa(params, c=c)

            if grad_prev is None or (
                restart_threshold > 0 and t % restart_threshold == 0
            ):
                direction = -grad
                beta = 0.0
            else:
                grad_diff = grad - grad_prev
                grad_prev_norm_sq = np.dot(grad_prev, grad_prev)

                if grad_prev_norm_sq > 1e-12:
                    beta = max(0.0, np.dot(grad, grad_diff) / grad_prev_norm_sq)
                else:
                    beta = 0.0

                direction = -grad + beta * direction

            params = params + learning_rate * direction
            grad_prev = grad.copy()

            if t % 10 == 0:
                loss = self.objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                if verbose:
                    print(f"  SPSA-CG iter {t}: loss = {loss:.6f}, β = {beta:.4f}")

        final_loss = self.objective(params)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params.copy()

        return best_params, best_loss

    def optimize_adam(
        self,
        params_init: np.ndarray,
        maxiter: int = 100,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        verbose: bool = False,
    ) -> tuple:
        """Optimize using Adam with parameter-shift gradients."""
        params = params_init.copy()
        m = np.zeros(self.n_params)
        v = np.zeros(self.n_params)

        best_loss = float("inf")
        best_params = params.copy()

        for t in range(1, maxiter + 1):
            grad = self.compute_gradient_parameter_shift(params)

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            loss = self.objective(params)
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            if verbose and t % 10 == 0:
                print(f"  Adam iter {t}: loss = {loss:.6f}")

        return best_params, best_loss

    def compute_state_fidelity(self, params1: np.ndarray, params2: np.ndarray) -> float:
        """Compute fidelity between two parameter configurations."""
        state1 = self.get_state_cpu(params1)
        state2 = self.get_state_cpu(params2)

        overlap = np.vdot(state1, state2)
        fidelity = float(np.abs(overlap) ** 2)

        return fidelity

    def compute_diagonal_metric(self, params: np.ndarray) -> np.ndarray:
        """Compute diagonal elements of the Fubini-Study metric."""
        g_diag = np.zeros(self.n_params)
        state_base = self.get_state_cpu(params)

        for j in range(self.n_params):
            params_shifted = params.copy()
            params_shifted[j] += np.pi

            state_shifted = self.get_state_cpu(params_shifted)
            overlap = np.vdot(state_base, state_shifted)
            fidelity = float(np.abs(overlap) ** 2)

            g_diag[j] = 0.25 * (1 - fidelity)

        return g_diag

    def optimize_qng(
        self,
        params_init: np.ndarray,
        maxiter: int = 100,
        learning_rate: float = 0.1,
        epsilon: float = 1e-6,
        verbose: bool = False,
    ) -> tuple:
        """Optimize using Quantum Natural Gradient with diagonal metric."""
        params = params_init.copy()
        best_loss = float("inf")
        best_params = params.copy()

        for t in range(1, maxiter + 1):
            grad = self.compute_gradient_parameter_shift(params)
            g_diag = self.compute_diagonal_metric(params)
            natural_grad = grad / (g_diag + epsilon)
            params = params - learning_rate * natural_grad

            loss = self.objective(params)
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            if verbose and t % 10 == 0:
                print(
                    f"  QNG iter {t}: loss = {loss:.6f}, |g|_min = {g_diag.min():.4f}"
                )

        return best_params, best_loss

    def extract_sequence(self, params: np.ndarray, refine: bool = True) -> tuple:
        """Extract binary sequence from optimized parameters."""
        state = self.get_state_cpu(params)
        expectations = self.compute_expectations(state)

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

    def _single_restart(
        self,
        restart_id: int,
        maxiter: int,
        method: str,
        learning_rate: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        qng_learning_rate: float = 0.1,
        spsa_a: float = 0.1,
        spsa_c: float = 0.1,
        spsa_A: float = 10.0,
        spsa_alpha: float = 0.602,
        spsa_gamma: float = 0.101,
    ) -> tuple:
        """Run a single optimization restart."""
        params_init = np.random.uniform(-np.pi, np.pi, self.n_params)

        try:
            if method == "SPSA":
                result_x, result_fun = self.optimize_spsa(
                    params_init,
                    maxiter=maxiter,
                    a=spsa_a,
                    c=spsa_c,
                    A=spsa_A,
                    alpha=spsa_alpha,
                    gamma=spsa_gamma,
                )
            elif method == "SPSA-Adam":
                result_x, result_fun = self.optimize_spsa_adam(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=learning_rate,
                    c=spsa_c,
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                )
            elif method == "SPSA-CG":
                result_x, result_fun = self.optimize_spsa_cg(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=learning_rate,
                    c=spsa_c,
                )
            elif method == "Adam":
                result_x, result_fun = self.optimize_adam(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                )
            elif method == "QNG":
                result_x, result_fun = self.optimize_qng(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=qng_learning_rate,
                )
            elif method == "L-BFGS-B":
                result = minimize(
                    self.objective,
                    params_init,
                    method="L-BFGS-B",
                    jac=self.compute_gradient_parameter_shift,
                    options={"maxiter": maxiter, "disp": False},
                )
                result_x, result_fun = result.x, result.fun
            else:
                # Derivative-free methods (COBYLA, Powell, Nelder-Mead)
                result = minimize(
                    self.objective,
                    params_init,
                    method=method,
                    options={"maxiter": maxiter, "disp": False},
                )
                result_x, result_fun = result.x, result.fun

            sequence, energy = self.extract_sequence(result_x)
            return restart_id, sequence, energy, result_x, result_fun
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
        learning_rate: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        qng_learning_rate: float = 0.1,
        spsa_a: float = 0.1,
        spsa_c: float = 0.1,
        spsa_A: float = 10.0,
        spsa_alpha: float = 0.602,
        spsa_gamma: float = 0.101,
    ) -> tuple:
        """Optimize with optional parallel restarts and tabu search refinement."""
        self._use_tabu = use_tabu
        self._tabu_iterations = tabu_iterations
        self._learning_rate = learning_rate
        self._adam_beta1 = adam_beta1
        self._adam_beta2 = adam_beta2
        self._qng_learning_rate = qng_learning_rate
        self._spsa_a = spsa_a
        self._spsa_c = spsa_c
        self._spsa_A = spsa_A
        self._spsa_alpha = spsa_alpha
        self._spsa_gamma = spsa_gamma

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
            try:
                _, sequence, energy, result_x, result_fun = self._single_restart(
                    restart,
                    maxiter,
                    method,
                    learning_rate=learning_rate,
                    adam_beta1=adam_beta1,
                    adam_beta2=adam_beta2,
                    qng_learning_rate=qng_learning_rate,
                    spsa_a=spsa_a,
                    spsa_c=spsa_c,
                    spsa_A=spsa_A,
                    spsa_alpha=spsa_alpha,
                    spsa_gamma=spsa_gamma,
                )

                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_sequence = sequence
                    self.best_params = result_x
                    self.best_loss = result_fun

                    if verbose:
                        pbar.set_postfix(
                            {"best_E": self.best_energy, "calls": self.call_count}
                        )

                    if energy == get_optimal_energy(self.N):
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

        Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        with Executor(max_workers=n_parallel) as executor:
            futures = []
            for i in range(n_restarts):
                future = executor.submit(
                    self._single_restart,
                    i,
                    maxiter,
                    method,
                    self._learning_rate,
                    self._adam_beta1,
                    self._adam_beta2,
                    self._qng_learning_rate,
                    self._spsa_a,
                    self._spsa_c,
                    self._spsa_A,
                    self._spsa_alpha,
                    self._spsa_gamma,
                )
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
