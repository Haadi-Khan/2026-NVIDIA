"""PCE Solver - main optimization class."""

import numpy as np
from scipy.optimize import minimize
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import cudaq
import cupy as cp
from tqdm import tqdm

from .constants import get_optimal_energy
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

    # =====================================================================
    # Phase 1: Gradient-Based Optimization Methods
    # =====================================================================

    def compute_gradient_parameter_shift(self, params: np.ndarray) -> np.ndarray:
        """Compute gradient using the parameter-shift rule.

        For RY and RZ gates, the shift is π/2.
        Cost: 2 * n_params function evaluations.

        Args:
            params: Current parameter vector

        Returns:
            Gradient vector of same shape as params
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

    # =====================================================================
    # Phase 3: SPSA Gradient Estimation (O(2) cost)
    # =====================================================================

    def compute_gradient_spsa(
        self,
        params: np.ndarray,
        c: float = 0.1,
        seed: int = None,
    ) -> np.ndarray:
        """Estimate gradient using SPSA (2 function evaluations).

        SPSA (Simultaneous Perturbation Stochastic Approximation) estimates
        the gradient with only 2 function evaluations, regardless of the
        number of parameters. This is O(2) vs O(2p) for parameter-shift.

        Args:
            params: Current parameter vector
            c: Perturbation magnitude
            seed: Random seed for reproducibility

        Returns:
            Gradient estimate vector
        """
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
        """Optimize using SPSA with decaying step sizes.

        Standard SPSA gain sequences (Spall, 1998):
        - a_k = a / (k + A)^alpha
        - c_k = c / k^gamma

        Args:
            params_init: Initial parameters
            maxiter: Maximum iterations
            a: Learning rate scale
            c: Perturbation magnitude
            A: Learning rate stabilization constant
            alpha: Learning rate decay exponent (standard: 0.602)
            gamma: Perturbation decay exponent (standard: 0.101)
            verbose: Print progress

        Returns:
            Tuple of (optimized_params, final_loss)
        """
        params = params_init.copy()
        best_loss = float("inf")
        best_params = params.copy()

        for k in range(1, maxiter + 1):
            # Decaying gain sequences
            a_k = a / (k + A) ** alpha
            c_k = c / k**gamma

            # SPSA gradient estimate (only 2 function evals!)
            grad = self.compute_gradient_spsa(params, c=c_k)

            # Update parameters
            params = params - a_k * grad

            # Track best (every 10 iterations to reduce overhead)
            if k % 10 == 0:
                loss = self.objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                if verbose:
                    print(f"  SPSA iter {k}: loss = {loss:.6f}")

        # Final evaluation
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
        """SPSA gradient estimation + Adam optimizer.

        Combines the best of both worlds:
        - SPSA: O(2) gradient cost (vs O(2p) for parameter-shift)
        - Adam: Momentum + adaptive learning rates for better convergence

        Args:
            params_init: Initial parameters
            maxiter: Maximum iterations
            learning_rate: Adam learning rate
            c: SPSA perturbation magnitude
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            verbose: Print progress

        Returns:
            Tuple of (optimized_params, final_loss)
        """
        params = params_init.copy()
        m = np.zeros(self.n_params)  # First moment
        v = np.zeros(self.n_params)  # Second moment

        best_loss = float("inf")
        best_params = params.copy()

        for t in range(1, maxiter + 1):
            # SPSA gradient (only 2 evals!)
            grad = self.compute_gradient_spsa(params, c=c)

            # Adam updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Track best (every 10 iterations to reduce overhead)
            if t % 10 == 0:
                loss = self.objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                if verbose:
                    print(f"  SPSA-Adam iter {t}: loss = {loss:.6f}")

        # Final evaluation
        final_loss = self.objective(params)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params.copy()

        return best_params, best_loss

    # =====================================================================
    # Phase 4: Conjugate Gradient + Wolfe Line Search
    # =====================================================================

    def optimize_spsa_cg(
        self,
        params_init: np.ndarray,
        maxiter: int = 100,
        learning_rate: float = 0.05,
        c: float = 0.1,
        restart_threshold: int = 0,
        verbose: bool = False,
    ) -> tuple:
        """SPSA with Polak-Ribiere conjugate gradient momentum.

        Combines SPSA gradient estimation with CG-style momentum that
        carries information from previous steps, inspired by EGT-CG from
        the geodesics paper.

        The Polak-Ribiere formula:
            β_t = max(0, grad_t · (grad_t - grad_{t-1}) / ||grad_{t-1}||²)
            direction_t = -grad_t + β_t * direction_{t-1}

        Args:
            params_init: Initial parameters
            maxiter: Maximum iterations
            learning_rate: Step size
            c: SPSA perturbation magnitude
            restart_threshold: Reset CG every N steps (0 = no restart)
            verbose: Print progress

        Returns:
            Tuple of (optimized_params, final_loss)
        """
        params = params_init.copy()
        best_loss = float("inf")
        best_params = params.copy()

        # Initialize CG state
        grad_prev = None
        direction = None

        for t in range(1, maxiter + 1):
            # SPSA gradient (only 2 evals!)
            grad = self.compute_gradient_spsa(params, c=c)

            if grad_prev is None or (restart_threshold > 0 and t % restart_threshold == 0):
                # First iteration or restart: steepest descent
                direction = -grad
                beta = 0.0
            else:
                # Polak-Ribiere formula with restart (max with 0)
                grad_diff = grad - grad_prev
                grad_prev_norm_sq = np.dot(grad_prev, grad_prev)

                if grad_prev_norm_sq > 1e-12:
                    beta = max(0.0, np.dot(grad, grad_diff) / grad_prev_norm_sq)
                else:
                    beta = 0.0

                # Conjugate direction
                direction = -grad + beta * direction

            # Update parameters
            params = params + learning_rate * direction

            # Store gradient for next iteration
            grad_prev = grad.copy()

            # Track best (every 10 iterations to reduce overhead)
            if t % 10 == 0:
                loss = self.objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

                if verbose:
                    print(f"  SPSA-CG iter {t}: loss = {loss:.6f}, β = {beta:.4f}")

        # Final evaluation
        final_loss = self.objective(params)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params.copy()

        return best_params, best_loss

    def wolfe_line_search(
        self,
        params: np.ndarray,
        direction: np.ndarray,
        grad: np.ndarray,
        loss: float,
        c: float = 0.1,
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_init: float = 1.0,
        alpha_max: float = 10.0,
        max_iter: int = 20,
    ) -> tuple:
        """Strong Wolfe line search for adaptive learning rate.

        Finds step size α satisfying strong Wolfe conditions:
        1. Sufficient decrease: L(θ + α*d) <= L(θ) + c1*α*(∇L·d)
        2. Curvature condition: |∇L(θ + α*d)·d| <= c2*|∇L·d|

        Uses bracketing and zoom algorithm (Nocedal & Wright, Chapter 3).

        Args:
            params: Current parameters
            direction: Search direction
            grad: Current gradient
            loss: Current loss value
            c: SPSA perturbation magnitude for gradient estimation
            c1: Sufficient decrease parameter (typically 1e-4)
            c2: Curvature parameter (typically 0.9 for CG)
            alpha_init: Initial step size guess
            alpha_max: Maximum step size
            max_iter: Maximum line search iterations

        Returns:
            Tuple of (alpha, new_loss, new_grad, success)
        """
        # Directional derivative at current point
        phi_0 = loss
        dphi_0 = np.dot(grad, direction)

        if dphi_0 >= 0:
            # Not a descent direction, return small step
            return 0.01, loss, grad, False

        alpha_prev = 0.0
        alpha = alpha_init
        phi_prev = phi_0

        for i in range(max_iter):
            # Evaluate at new point
            params_new = params + alpha * direction
            phi = self.objective(params_new)

            # Check Armijo condition (sufficient decrease)
            if phi > phi_0 + c1 * alpha * dphi_0 or (i > 0 and phi >= phi_prev):
                # Need to zoom between alpha_prev and alpha
                return self._wolfe_zoom(
                    params, direction, phi_0, dphi_0,
                    alpha_prev, alpha, phi_prev, phi, c, c1, c2
                )

            # Compute gradient at new point
            grad_new = self.compute_gradient_spsa(params_new, c=c)
            dphi = np.dot(grad_new, direction)

            # Check strong Wolfe curvature condition
            if abs(dphi) <= -c2 * dphi_0:
                return alpha, phi, grad_new, True

            # Check if we've gone too far
            if dphi >= 0:
                return self._wolfe_zoom(
                    params, direction, phi_0, dphi_0,
                    alpha, alpha_prev, phi, phi_prev, c, c1, c2
                )

            # Increase step size
            alpha_prev = alpha
            phi_prev = phi
            alpha = min(2 * alpha, alpha_max)

        # Max iterations reached, return best found
        return alpha, phi, grad_new, False

    def _wolfe_zoom(
        self,
        params: np.ndarray,
        direction: np.ndarray,
        phi_0: float,
        dphi_0: float,
        alpha_lo: float,
        alpha_hi: float,
        phi_lo: float,
        phi_hi: float,
        c: float,
        c1: float,
        c2: float,
        max_iter: int = 10,
    ) -> tuple:
        """Zoom phase of Wolfe line search (bisection)."""
        for _ in range(max_iter):
            # Bisection
            alpha = 0.5 * (alpha_lo + alpha_hi)

            params_new = params + alpha * direction
            phi = self.objective(params_new)

            if phi > phi_0 + c1 * alpha * dphi_0 or phi >= phi_lo:
                alpha_hi = alpha
                phi_hi = phi
            else:
                grad_new = self.compute_gradient_spsa(params_new, c=c)
                dphi = np.dot(grad_new, direction)

                if abs(dphi) <= -c2 * dphi_0:
                    return alpha, phi, grad_new, True

                if dphi * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                    phi_hi = phi_lo

                alpha_lo = alpha
                phi_lo = phi

        # Return best found
        params_new = params + alpha * direction
        grad_new = self.compute_gradient_spsa(params_new, c=c)
        return alpha, phi, grad_new, False

    def optimize_spsa_cg_wolfe(
        self,
        params_init: np.ndarray,
        maxiter: int = 100,
        c: float = 0.1,
        c1: float = 1e-4,
        c2: float = 0.4,
        restart_threshold: int = 0,
        verbose: bool = False,
    ) -> tuple:
        """SPSA-CG with strong Wolfe line search (inspired by EGT-CG).

        Combines:
        - SPSA: O(2) gradient estimation
        - Polak-Ribiere CG: Momentum with global convergence
        - Strong Wolfe: Adaptive learning rate with guarantees

        This is our adaptation of the EGT-CG algorithm from the geodesics
        paper, modified to work with our brickwork ansatz using SPSA
        gradients instead of exact geodesic transport.

        Args:
            params_init: Initial parameters
            maxiter: Maximum iterations
            c: SPSA perturbation magnitude
            c1: Wolfe sufficient decrease parameter
            c2: Wolfe curvature parameter (0.4 typical for CG)
            restart_threshold: Reset CG every N steps (0 = auto)
            verbose: Print progress

        Returns:
            Tuple of (optimized_params, final_loss)
        """
        params = params_init.copy()
        best_loss = float("inf")
        best_params = params.copy()

        # Initialize
        grad = self.compute_gradient_spsa(params, c=c)
        loss = self.objective(params)
        direction = -grad
        grad_prev = None

        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()

        for t in range(1, maxiter + 1):
            # Wolfe line search for optimal step size
            alpha, new_loss, new_grad, success = self.wolfe_line_search(
                params, direction, grad, loss, c=c, c1=c1, c2=c2
            )

            # Update parameters
            params = params + alpha * direction
            loss = new_loss
            grad_prev = grad
            grad = new_grad

            # Track best
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            # Check for restart conditions
            should_restart = False
            if restart_threshold > 0 and t % restart_threshold == 0:
                should_restart = True
            elif not success:
                should_restart = True

            if should_restart or grad_prev is None:
                # Restart: steepest descent
                direction = -grad
                beta = 0.0
            else:
                # Polak-Ribiere with automatic restart
                grad_diff = grad - grad_prev
                grad_prev_norm_sq = np.dot(grad_prev, grad_prev)

                if grad_prev_norm_sq > 1e-12:
                    beta = max(0.0, np.dot(grad, grad_diff) / grad_prev_norm_sq)
                else:
                    beta = 0.0

                direction = -grad + beta * direction

                # Check for loss of conjugacy (restart if direction not descent)
                if np.dot(direction, grad) > 0:
                    direction = -grad
                    beta = 0.0

            if verbose and t % 5 == 0:
                print(
                    f"  SPSA-CG-Wolfe iter {t}: loss = {loss:.6f}, "
                    f"α = {alpha:.4f}, β = {beta:.4f}"
                )

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
        """Optimize using Adam with parameter-shift gradients.

        Args:
            params_init: Initial parameters
            maxiter: Maximum iterations
            learning_rate: Step size (η)
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            verbose: Print progress

        Returns:
            Tuple of (optimized_params, final_loss)
        """
        params = params_init.copy()
        m = np.zeros(self.n_params)  # First moment
        v = np.zeros(self.n_params)  # Second moment

        best_loss = float("inf")
        best_params = params.copy()

        for t in range(1, maxiter + 1):
            # Compute gradient
            grad = self.compute_gradient_parameter_shift(params)

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad

            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (grad**2)

            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update parameters
            params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Track best
            loss = self.objective(params)
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            if verbose and t % 10 == 0:
                print(f"  Adam iter {t}: loss = {loss:.6f}")

        return best_params, best_loss

    # =====================================================================
    # Phase 2: Quantum Natural Gradient Methods
    # =====================================================================

    def compute_state_fidelity(
        self, params1: np.ndarray, params2: np.ndarray
    ) -> float:
        """Compute fidelity between two parameter configurations.

        Fidelity = |⟨ψ(params1)|ψ(params2)⟩|²

        Args:
            params1: First parameter vector
            params2: Second parameter vector

        Returns:
            Fidelity value in [0, 1]
        """
        state1 = self.get_state_gpu(params1)
        state2 = self.get_state_gpu(params2)

        # Inner product on GPU
        overlap = cp.vdot(state1, state2)
        fidelity = float(cp.abs(overlap) ** 2)

        return fidelity

    def compute_diagonal_metric(self, params: np.ndarray) -> np.ndarray:
        """Compute diagonal elements of the Fubini-Study metric.

        Uses Eq. (27) from geodesics paper:
        g_jj = 1/4 * (1 - |⟨ψ(θ)|ψ(θ + π·e_j)⟩|²)

        Cost: p state preparations (can be optimized with caching)

        Args:
            params: Current parameter vector

        Returns:
            Diagonal metric vector of shape (n_params,)
        """
        g_diag = np.zeros(self.n_params)

        # Get base state once
        state_base = self.get_state_gpu(params)

        for j in range(self.n_params):
            # Shifted parameters (full π shift for diagonal element)
            params_shifted = params.copy()
            params_shifted[j] += np.pi

            # Compute fidelity
            state_shifted = self.get_state_gpu(params_shifted)
            overlap = cp.vdot(state_base, state_shifted)
            fidelity = float(cp.abs(overlap) ** 2)

            # Diagonal metric element
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
        """Optimize using Quantum Natural Gradient with diagonal metric.

        QNG update: θ_{t+1} = θ_t - η * g^{-1}(θ_t) * ∇L(θ_t)

        With diagonal approximation:
        θ_j^{new} = θ_j - η * (∂L/∂θ_j) / (g_jj + ε)

        Args:
            params_init: Initial parameters
            maxiter: Maximum iterations
            learning_rate: Step size (η)
            epsilon: Regularization for metric inversion
            verbose: Print progress

        Returns:
            Tuple of (optimized_params, final_loss)
        """
        params = params_init.copy()
        best_loss = float("inf")
        best_params = params.copy()

        for t in range(1, maxiter + 1):
            # Compute gradient (2p function evals)
            grad = self.compute_gradient_parameter_shift(params)

            # Compute diagonal metric (p state preparations)
            g_diag = self.compute_diagonal_metric(params)

            # Natural gradient (element-wise)
            natural_grad = grad / (g_diag + epsilon)

            # Update parameters
            params = params - learning_rate * natural_grad

            # Track best
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

    def _single_restart(
        self,
        restart_id: int,
        maxiter: int,
        method: str,
        learning_rate: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        qng_learning_rate: float = 0.1,
        # SPSA hyperparameters
        spsa_a: float = 0.1,
        spsa_c: float = 0.1,
        spsa_A: float = 10.0,
        spsa_alpha: float = 0.602,
        spsa_gamma: float = 0.101,
    ) -> tuple:
        """Run a single optimization restart. Used for parallel execution.

        Supports methods:
        - Derivative-free: COBYLA, Powell, Nelder-Mead
        - Gradient-based: L-BFGS-B, Adam
        - QNG: QNG (Quantum Natural Gradient with diagonal metric)
        - SPSA: SPSA, SPSA-Adam, SPSA-CG, SPSA-CG-Wolfe
        """
        params_init = np.random.uniform(-np.pi, np.pi, self.n_params)

        try:
            if method == "SPSA":
                # Use SPSA optimizer (O(2) gradient cost)
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
                # Use SPSA + Adam optimizer (O(2) gradient cost + momentum)
                result_x, result_fun = self.optimize_spsa_adam(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=learning_rate,
                    c=spsa_c,
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                )
            elif method == "SPSA-CG":
                # Use SPSA + Conjugate Gradient (Polak-Ribiere momentum)
                result_x, result_fun = self.optimize_spsa_cg(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=learning_rate,
                    c=spsa_c,
                )
            elif method == "SPSA-CG-Wolfe":
                # Use SPSA + CG + Strong Wolfe line search (inspired by EGT-CG)
                result_x, result_fun = self.optimize_spsa_cg_wolfe(
                    params_init,
                    maxiter=maxiter,
                    c=spsa_c,
                )
            elif method == "Adam":
                # Use custom Adam optimizer (with parameter-shift gradients)
                result_x, result_fun = self.optimize_adam(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                )
            elif method == "QNG":
                # Use Quantum Natural Gradient optimizer
                result_x, result_fun = self.optimize_qng(
                    params_init,
                    maxiter=maxiter,
                    learning_rate=qng_learning_rate,
                )
            elif method == "L-BFGS-B":
                # Use scipy L-BFGS-B with gradients
                result = minimize(
                    self.objective,
                    params_init,
                    method="L-BFGS-B",
                    jac=self.compute_gradient_parameter_shift,
                    options={"maxiter": maxiter, "disp": False},
                )
                result_x, result_fun = result.x, result.fun
            else:
                # Original derivative-free methods (COBYLA, Powell, Nelder-Mead)
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
        # Parameters for gradient-based methods
        learning_rate: float = 0.01,  # For Adam and SPSA-Adam
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        qng_learning_rate: float = 0.1,  # For QNG
        # SPSA hyperparameters
        spsa_a: float = 0.1,  # Learning rate scale
        spsa_c: float = 0.1,  # Perturbation magnitude
        spsa_A: float = 10.0,  # Learning rate stabilization
        spsa_alpha: float = 0.602,  # Learning rate decay exponent
        spsa_gamma: float = 0.101,  # Perturbation decay exponent
    ) -> tuple:
        """Optimize with optional parallel restarts and tabu search refinement.

        Args:
            n_restarts: Number of random restarts
            maxiter: Max iterations per restart
            verbose: Show progress
            method: Optimization method. Supported:
                - Derivative-free: 'COBYLA', 'Powell', 'Nelder-Mead'
                - Gradient-based: 'L-BFGS-B', 'Adam'
                - Quantum Natural Gradient: 'QNG'
                - SPSA (O(2) gradient): 'SPSA', 'SPSA-Adam', 'SPSA-CG', 'SPSA-CG-Wolfe'
            n_parallel: Number of parallel restarts (1=sequential, >1=parallel)
            use_processes: Use ProcessPoolExecutor instead of threads (bypasses GIL)
            use_tabu: If True, apply tabu search refinement to best solution
            tabu_iterations: Maximum iterations for tabu search
            learning_rate: Learning rate for Adam/SPSA-Adam optimizer
            adam_beta1: First moment decay rate for Adam
            adam_beta2: Second moment decay rate for Adam
            qng_learning_rate: Learning rate for QNG optimizer
            spsa_a: SPSA learning rate scale
            spsa_c: SPSA perturbation magnitude
            spsa_A: SPSA learning rate stabilization constant
            spsa_alpha: SPSA learning rate decay exponent
            spsa_gamma: SPSA perturbation decay exponent

        Returns:
            Tuple of (best_sequence, best_energy)
        """
        self._use_tabu = use_tabu
        self._tabu_iterations = tabu_iterations
        self._learning_rate = learning_rate
        self._adam_beta1 = adam_beta1
        self._adam_beta2 = adam_beta2
        self._qng_learning_rate = qng_learning_rate
        # Store SPSA hyperparameters
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

        # Choose executor type
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
