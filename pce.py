import cudaq
from cudaq import spin
import numpy as np
from scipy.optimize import minimize
from itertools import combinations, product
from functools import reduce
import time

# Optional: keep cupy for fallback/comparison
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

GPU_AVAILABLE = True
cudaq.set_target('nvidia')

OPTIMAL_ENERGIES = {
    3: 1, 4: 2, 5: 2, 6: 7, 7: 3, 8: 8, 9: 12, 10: 13,
    11: 5, 12: 10, 13: 6, 14: 19, 15: 15, 16: 24, 17: 32,
    18: 25, 19: 29, 20: 26, 21: 26, 22: 39, 23: 47, 24: 36, 25: 36
}

PAULI_MATS = {
    'I': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)
}


def pauli_string_to_spin_op(pauli_str: str) -> cudaq.SpinOperator:
    """Convert a Pauli string like 'XZIY' to a cudaq.SpinOperator.
    
    Args:
        pauli_str: String of Pauli operators (I, X, Y, Z) from qubit 0 to n-1
        
    Returns:
        cudaq.SpinOperator representing the tensor product
    """
    n_qubits = len(pauli_str)
    
    # Start with coefficient 1
    result = None
    
    for i, p in enumerate(pauli_str):
        if p == 'I':
            # Identity doesn't contribute to the operator
            continue
        elif p == 'X':
            op = spin.x(i)
        elif p == 'Y':
            op = spin.y(i)
        elif p == 'Z':
            op = spin.z(i)
        else:
            raise ValueError(f"Unknown Pauli operator: {p}")
        
        if result is None:
            result = op
        else:
            result = result * op
    
    # If all identity, return identity operator
    if result is None:
        result = spin.i(0)
    
    return result


def build_spin_operators(pauli_strings: list) -> list:
    """Build a list of cudaq.SpinOperator from Pauli strings.
    
    Args:
        pauli_strings: List of Pauli strings
        
    Returns:
        List of cudaq.SpinOperator objects
    """
    return [pauli_string_to_spin_op(p) for p in pauli_strings]


def labs_energy(s: np.ndarray) -> int:
    N = len(s)
    s = np.asarray(s, dtype=np.int32)
    energy = 0
    for k in range(1, N):
        C_k = np.dot(s[:N-k], s[k:])
        energy += C_k * C_k
    return int(energy)


def merit_factor(s: np.ndarray) -> float:
    N = len(s)
    E = labs_energy(s)
    if E == 0:
        return float('inf')
    return N * N / (2 * E)


def build_kbody_paulis(n: int, k: int, N: int) -> list:
    paulis = []
    indices = list(combinations(range(n), k))
    
    for pauli_type in ['X', 'Y', 'Z']:
        for idx_set in indices:
            pauli = ['I'] * n
            for i in idx_set:
                pauli[i] = pauli_type
            paulis.append(''.join(pauli))
            if len(paulis) >= N:
                return paulis
    
    if len(paulis) < N:
        all_paulis = [''.join(combo) for combo in product('IXYZ', repeat=n) 
                      if combo != tuple('I' * n)]
        for p in all_paulis:
            if p not in paulis:
                paulis.append(p)
                if len(paulis) >= N:
                    break
    
    return paulis[:N]


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    mats = [PAULI_MATS[p] for p in pauli_str]
    return reduce(np.kron, mats)


class PauliExpectationCalculator:
    """Legacy expectation calculator using dense matrices and CuPy (slow)."""
    def __init__(self, pauli_strings: list, use_gpu: bool = True):
        self.pauli_strings = pauli_strings
        self.n_paulis = len(pauli_strings)
        self.n_qubits = len(pauli_strings[0])
        self.dim = 2 ** self.n_qubits
        self.use_gpu = use_gpu and GPU_AVAILABLE and CUPY_AVAILABLE
        
        pauli_mats = np.stack([pauli_string_to_matrix(p) for p in pauli_strings])
        
        if self.use_gpu:
            self.pauli_mats_gpu = cp.asarray(pauli_mats)
        else:
            self.pauli_mats_gpu = pauli_mats
    
    def compute_expectations(self, state_vector: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            sv = cp.asarray(state_vector)
            sv_conj = cp.conj(sv)
            transformed = cp.einsum('pij,j->pi', self.pauli_mats_gpu, sv)
            expectations = cp.real(cp.einsum('i,pi->p', sv_conj, transformed))
            return cp.asnumpy(expectations)
        else:
            sv = state_vector
            expectations = np.zeros(self.n_paulis, dtype=np.float64)
            for i in range(self.n_paulis):
                transformed = self.pauli_mats_gpu[i] @ sv
                expectations[i] = np.real(np.vdot(sv, transformed))
            return expectations


class CudaqExpectationCalculator:
    """Expectation calculator using cudaq.observe() - one circuit per Pauli (slower)."""
    
    def __init__(self, pauli_strings: list):
        """Initialize with Pauli strings and build SpinOperators.
        
        Args:
            pauli_strings: List of Pauli strings (e.g., ['XZII', 'IYZI', ...])
        """
        self.pauli_strings = pauli_strings
        self.n_paulis = len(pauli_strings)
        self.n_qubits = len(pauli_strings[0]) if pauli_strings else 0
        
        # Pre-build SpinOperators for each Pauli string
        self.spin_operators = build_spin_operators(pauli_strings)
    
    def compute_expectations_with_observe(self, kernel, n_qubits: int, 
                                          params: list, n_layers: int) -> np.ndarray:
        """Compute all Pauli expectations using cudaq.observe().
        
        Note: This runs the circuit once per Pauli, so it's slower for many Paulis.
        
        Args:
            kernel: The CUDA-Q kernel function
            n_qubits: Number of qubits
            params: List of circuit parameters
            n_layers: Number of ansatz layers
            
        Returns:
            numpy array of expectation values
        """
        expectations = np.zeros(self.n_paulis, dtype=np.float64)
        
        for i, spin_op in enumerate(self.spin_operators):
            result = cudaq.observe(kernel, spin_op, n_qubits, params, n_layers)
            expectations[i] = result.expectation()
        
        return expectations


class SparsePauliExpectationCalculator:
    """Fast expectation calculator using sparse Pauli application on GPU.
    
    This approach:
    1. Gets the state vector once (single circuit execution)
    2. Applies Pauli operators sparsely in O(2^n) instead of O(2^{2n}) for dense
    3. Batches all Pauli expectations on GPU using CuPy
    """
    
    def __init__(self, pauli_strings: list, use_gpu: bool = True):
        """Initialize with Pauli strings.
        
        Args:
            pauli_strings: List of Pauli strings (e.g., ['XZII', 'IYZI', ...])
            use_gpu: If True, use CuPy for GPU acceleration
        """
        self.pauli_strings = pauli_strings
        self.n_paulis = len(pauli_strings)
        self.n_qubits = len(pauli_strings[0]) if pauli_strings else 0
        self.dim = 2 ** self.n_qubits
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # Pre-compute Pauli action indices and phases for each Pauli string
        # For Pauli string P, P|j⟩ = phase[j] * |target[j]⟩
        self._precompute_pauli_actions()
    
    def _precompute_pauli_actions(self):
        """Precompute the target indices and phases for each Pauli string."""
        # For each Pauli string, we need:
        # - target_indices: where each basis state maps to
        # - phases: the phase factor for each basis state
        
        self.target_indices = np.zeros((self.n_paulis, self.dim), dtype=np.int64)
        self.phases = np.zeros((self.n_paulis, self.dim), dtype=np.complex128)
        
        for p_idx, pauli_str in enumerate(self.pauli_strings):
            for j in range(self.dim):
                target, phase = self._apply_pauli_to_basis(pauli_str, j)
                self.target_indices[p_idx, j] = target
                self.phases[p_idx, j] = phase
        
        # Move to GPU if available
        if self.use_gpu:
            self.target_indices_gpu = cp.asarray(self.target_indices)
            self.phases_gpu = cp.asarray(self.phases)
    
    def _apply_pauli_to_basis(self, pauli_str: str, basis_idx: int) -> tuple:
        """Apply Pauli string to a basis state.
        
        For P|j⟩ = phase * |j'⟩, returns (j', phase).
        
        Args:
            pauli_str: Pauli string like 'XZIY'
            basis_idx: Index of basis state |j⟩
            
        Returns:
            (target_idx, phase): The target basis index and phase
        """
        target = basis_idx
        phase = 1.0 + 0.0j
        
        for qubit_idx, pauli in enumerate(pauli_str):
            # Get the bit value of this qubit in basis_idx
            bit = (basis_idx >> qubit_idx) & 1
            
            if pauli == 'I':
                # Identity: no change
                pass
            elif pauli == 'X':
                # X flips the bit: |0⟩ ↔ |1⟩
                target ^= (1 << qubit_idx)
            elif pauli == 'Y':
                # Y = i * |1⟩⟨0| - i * |0⟩⟨1|
                # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                target ^= (1 << qubit_idx)
                if bit == 0:
                    phase *= 1j
                else:
                    phase *= -1j
            elif pauli == 'Z':
                # Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                if bit == 1:
                    phase *= -1
        
        return target, phase
    
    def compute_expectations(self, state_vector: np.ndarray) -> np.ndarray:
        """Compute all Pauli expectations efficiently.
        
        Uses the precomputed sparse Pauli actions to compute:
        ⟨ψ|P|ψ⟩ = Σ_j ψ_j* × phase[j] × ψ_{target[j]}
        
        Args:
            state_vector: The quantum state vector
            
        Returns:
            Array of expectation values
        """
        if self.use_gpu:
            sv = cp.asarray(state_vector)
            sv_conj = cp.conj(sv)
            
            # Gather transformed amplitudes for all Paulis at once
            # transformed[p, j] = phase[p, j] * sv[target[p, j]]
            transformed = self.phases_gpu * sv[self.target_indices_gpu]
            
            # Compute expectations: sum over j of sv_conj[j] * transformed[p, j]
            expectations = cp.real(cp.sum(sv_conj * transformed, axis=1))
            
            return cp.asnumpy(expectations)
        else:
            sv = state_vector
            sv_conj = np.conj(sv)
            
            transformed = self.phases * sv[self.target_indices]
            expectations = np.real(np.sum(sv_conj * transformed, axis=1))
            
            return expectations


@cudaq.kernel
def ansatz(n_qubits: int, params: list[float], n_layers: int):
    qubits = cudaq.qvector(n_qubits)
    
    for q in range(n_qubits):
        h(qubits[q])
    
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            ry(params[idx], qubits[q])
            rz(params[idx + 1], qubits[q])
            idx += 2
        
        for q in range(0, n_qubits - 1, 2):
            x.ctrl(qubits[q], qubits[q + 1])
        
        for q in range(1, n_qubits - 1, 2):
            x.ctrl(qubits[q], qubits[q + 1])


def get_n_params(n_qubits: int, n_layers: int) -> int:
    return n_layers * 2 * n_qubits


class PCESolver:
    def __init__(self, N: int, n_qubits: int = 10, n_layers: int = 10, 
                 k: int = 2, alpha: float = 0.0, beta: float = 15.0,
                 use_optimized: bool = True):
        """Initialize the PCE Solver.
        
        Args:
            N: Sequence length (number of Pauli expectations)
            n_qubits: Number of qubits in the ansatz
            n_layers: Number of layers in the ansatz
            k: k-body Pauli operators
            alpha: Scaling factor for tanh
            beta: Regularization strength
            use_optimized: If True, use sparse Pauli application (fast). 
                          If False, use legacy dense matrix method (slow).
        """
        self.N = N
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.k = k
        self.alpha = alpha if alpha > 0 else 1.5 * n_qubits
        self.beta = beta
        self.n_params = get_n_params(n_qubits, n_layers)
        self.use_optimized = use_optimized
        
        self.pauli_strings = build_kbody_paulis(n_qubits, k, N)
        
        # Use sparse calculator for optimized path (single circuit + sparse ops)
        if use_optimized:
            self.expectation_calc = SparsePauliExpectationCalculator(self.pauli_strings)
        else:
            self.expectation_calc = PauliExpectationCalculator(self.pauli_strings)
        
        self.best_params = None
        self.best_loss = float('inf')
        self.best_sequence = None
        self.best_energy = float('inf')
        self.call_count = 0
        self.time_get_state = 0.0
        self.time_expectations = 0.0
        self.time_loss = 0.0
    
    def get_state(self, params: np.ndarray) -> np.ndarray:
        """Get state vector from the ansatz circuit."""
        state = cudaq.get_state(ansatz, self.n_qubits, list(params), self.n_layers)
        return np.array(state, dtype=np.complex128)
    
    def compute_expectations(self, params: np.ndarray) -> np.ndarray:
        """Compute Pauli expectations using the appropriate method.
        
        Args:
            params: Circuit parameters
            
        Returns:
            Array of expectation values
        """
        # Both methods now use get_state + expectation calculation
        # The difference is sparse vs dense Pauli application
        state = self.get_state(params)
        return self.expectation_calc.compute_expectations(state)
    
    def objective(self, params: np.ndarray) -> float:
        self.call_count += 1
        
        # Get state vector (same for both methods)
        t0 = time.perf_counter()
        state = self.get_state(params)
        self.time_get_state += time.perf_counter() - t0
        
        # Compute expectations (sparse vs dense)
        t0 = time.perf_counter()
        expectations = self.expectation_calc.compute_expectations(state)
        self.time_expectations += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        x_tilde = np.tanh(self.alpha * expectations)
        loss = 0.0
        for l in range(1, self.N):
            C_l = np.dot(x_tilde[:self.N-l], x_tilde[l:])
            loss += C_l * C_l
        reg = -self.beta * np.dot(x_tilde, x_tilde)
        self.time_loss += time.perf_counter() - t0
        
        return loss + reg
    
    def extract_sequence(self, params: np.ndarray) -> tuple:
        """Extract binary sequence from optimized parameters."""
        expectations = self.compute_expectations(params)
        
        sequence = np.sign(expectations).astype(np.int8)
        sequence[sequence == 0] = 1
        
        return sequence, labs_energy(sequence)
    
    def print_timing(self):
        total = self.time_get_state + self.time_expectations + self.time_loss
        mode = "SPARSE" if self.use_optimized else "DENSE"
        print(f"Timing breakdown ({self.call_count} calls) [{mode}]:")
        print(f"  get_state:    {self.time_get_state:.2f}s ({100*self.time_get_state/total:.1f}%)")
        print(f"  expectations: {self.time_expectations:.2f}s ({100*self.time_expectations/total:.1f}%)")
        print(f"  loss:         {self.time_loss:.2f}s ({100*self.time_loss/total:.1f}%)")
        print(f"  per call:     {1000*total/self.call_count:.2f}ms")
    
    def optimize(self, n_restarts: int = 50, maxiter: int = 100, 
                 verbose: bool = False, method: str = 'COBYLA') -> tuple:
        
        for restart in range(n_restarts):
            params_init = np.random.uniform(-np.pi, np.pi, self.n_params)
            
            try:
                result = minimize(
                    self.objective,
                    params_init,
                    method=method,
                    options={'maxiter': maxiter, 'disp': False}
                )
                
                sequence, energy = self.extract_sequence(result.x)
                
                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_sequence = sequence
                    self.best_params = result.x
                    self.best_loss = result.fun
                    
                    if verbose:
                        print(f"Restart {restart + 1}/{n_restarts}: energy={energy}, calls={self.call_count}")
                    
                    if energy == OPTIMAL_ENERGIES.get(self.N, -1):
                        if verbose:
                            print(f"Found optimal!")
                        break
                
            except Exception as e:
                if verbose:
                    print(f"Restart {restart + 1} failed: {e}")
                continue
        
        return self.best_sequence, int(self.best_energy) if self.best_energy < float('inf') else -1
