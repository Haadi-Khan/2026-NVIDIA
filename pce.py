import cudaq
import numpy as np
from scipy.optimize import minimize
from itertools import combinations, product
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

GPU_AVAILABLE = True
cudaq.set_target('nvidia')

OPTIMAL_ENERGIES = {
    3: 1, 4: 2, 5: 2, 6: 7, 7: 3, 8: 8, 9: 12, 10: 13,
    11: 5, 12: 10, 13: 6, 14: 19, 15: 15, 16: 24, 17: 32,
    18: 25, 19: 29, 20: 26, 21: 26, 22: 39, 23: 47, 24: 36, 25: 36
}


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
    """Build k-body Pauli strings for n qubits."""
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


def pauli_string_to_spin_op(pauli_str: str) -> cudaq.SpinOperator:
    """Convert a Pauli string like 'XXIIZ' to a cudaq.SpinOperator.
    
    The string is read left-to-right as qubit 0, 1, 2, ...
    """
    n = len(pauli_str)
    op = None
    
    for qubit_idx, pauli in enumerate(pauli_str):
        if pauli == 'I':
            term = cudaq.spin.i(qubit_idx)
        elif pauli == 'X':
            term = cudaq.spin.x(qubit_idx)
        elif pauli == 'Y':
            term = cudaq.spin.y(qubit_idx)
        elif pauli == 'Z':
            term = cudaq.spin.z(qubit_idx)
        else:
            raise ValueError(f"Unknown Pauli: {pauli}")
        
        if op is None:
            op = term
        else:
            op = op * term
    
    return op


def build_spin_operators(pauli_strings: list) -> list:
    """Build a list of cudaq.SpinOperator from Pauli strings."""
    return [pauli_string_to_spin_op(ps) for ps in pauli_strings]


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
    """PCE Solver with GPU-native computation.
    
    Features:
    1. Keeps state on GPU - avoids np.array() conversion overhead
    2. Computes expectations entirely on GPU using CuPy
    3. Computes loss using hybrid GPU/CPU approach
    4. Supports parallel restarts for faster optimization
    """
    
    def __init__(self, N: int, n_qubits: int = 10, n_layers: int = 10, 
                 k: int = 2, alpha: float = 0.0, beta: float = 15.0):
        """Initialize the PCE Solver.
        
        Args:
            N: Sequence length (number of Pauli expectations)
            n_qubits: Number of qubits in the ansatz
            n_layers: Number of layers in the ansatz
            k: k-body Pauli operators
            alpha: Scaling factor for tanh
            beta: Regularization strength
        """
        self.N = N
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.k = k
        self.alpha = alpha if alpha > 0 else 1.5 * n_qubits
        self.beta = beta
        self.n_params = get_n_params(n_qubits, n_layers)
        
        # Build Pauli strings
        self.pauli_strings = build_kbody_paulis(n_qubits, k, N)
        
        # Use GPU if available
        self.use_gpu = CUPY_AVAILABLE
        
        # Pre-compute Pauli action indices and phases on GPU
        self._precompute_pauli_actions_gpu()
        
        self.best_params = None
        self.best_loss = float('inf')
        self.best_sequence = None
        self.best_energy = float('inf')
        self.call_count = 0
        self.time_circuit = 0.0
        self.time_expectations = 0.0
        self.time_loss = 0.0
    
    def _precompute_pauli_actions_gpu(self):
        """Precompute Pauli action indices and phases directly on GPU."""
        n_paulis = len(self.pauli_strings)
        dim = 2 ** self.n_qubits
        
        # Compute on CPU first (small overhead, done once)
        target_indices = np.zeros((n_paulis, dim), dtype=np.int64)
        phases = np.zeros((n_paulis, dim), dtype=np.complex128)
        
        for p_idx, pauli_str in enumerate(self.pauli_strings):
            for j in range(dim):
                target, phase = self._apply_pauli_to_basis(pauli_str, j)
                target_indices[p_idx, j] = target
                phases[p_idx, j] = phase
        
        # Move to GPU and keep there
        if self.use_gpu:
            self.target_indices_gpu = cp.asarray(target_indices)
            self.phases_gpu = cp.asarray(phases)
        else:
            self.target_indices_cpu = target_indices
            self.phases_cpu = phases
    
    def _apply_pauli_to_basis(self, pauli_str: str, basis_idx: int) -> tuple:
        """Apply Pauli string to a basis state: P|j⟩ = phase * |j'⟩."""
        target = basis_idx
        phase = 1.0 + 0.0j
        
        for qubit_idx, pauli in enumerate(pauli_str):
            bit = (basis_idx >> qubit_idx) & 1
            
            if pauli == 'I':
                pass
            elif pauli == 'X':
                target ^= (1 << qubit_idx)
            elif pauli == 'Y':
                target ^= (1 << qubit_idx)
                phase *= 1j if bit == 0 else -1j
            elif pauli == 'Z':
                if bit == 1:
                    phase *= -1
        
        return target, phase
    
    def get_state_gpu(self, params: np.ndarray):
        """Get state vector and immediately move to GPU."""
        state = cudaq.get_state(ansatz, self.n_qubits, list(params), self.n_layers)
        # Convert directly to CuPy array, avoiding intermediate numpy array
        if self.use_gpu:
            return cp.asarray(state, dtype=cp.complex128)
        else:
            return np.array(state, dtype=np.complex128)
    
    def compute_expectations_gpu(self, state_gpu) -> 'cp.ndarray or np.ndarray':
        """Compute all Pauli expectations entirely on GPU, return GPU array."""
        if self.use_gpu:
            sv_conj = cp.conj(state_gpu)
            transformed = self.phases_gpu * state_gpu[self.target_indices_gpu]
            expectations = cp.real(cp.sum(sv_conj * transformed, axis=1))
            return expectations  # Keep on GPU!
        else:
            sv_conj = np.conj(state_gpu)
            transformed = self.phases_cpu * state_gpu[self.target_indices_cpu]
            expectations = np.real(np.sum(sv_conj * transformed, axis=1))
            return expectations
    
    def compute_loss_gpu(self, expectations_gpu) -> float:
        """Compute loss - GPU tanh + CPU autocorrelation (hybrid approach).
        
        The LABS energy is: E = sum_{k=1}^{N-1} C_k^2
        where C_k = sum_{i=0}^{N-k-1} x[i] * x[i+k]
        
        For small-medium N (<500), CPU autocorrelation is faster than GPU
        due to kernel launch overhead. Only the N-element expectations array
        is transferred, which is negligible.
        """
        if self.use_gpu:
            # Compute tanh on GPU
            x_tilde_gpu = cp.tanh(self.alpha * expectations_gpu)
            # Transfer small array (N elements) to CPU for autocorrelation
            x_tilde = cp.asnumpy(x_tilde_gpu)
        else:
            x_tilde = np.tanh(self.alpha * expectations_gpu)
        
        # CPU-based autocorrelation (fast for N < 500)
        loss = 0.0
        for l in range(1, self.N):
            C_l = np.dot(x_tilde[:self.N-l], x_tilde[l:])
            loss += C_l * C_l
        
        reg = -self.beta * np.dot(x_tilde, x_tilde)
        return loss + reg
    
    def objective(self, params: np.ndarray) -> float:
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
    
    def extract_sequence(self, params: np.ndarray) -> tuple:
        """Extract binary sequence from optimized parameters."""
        state_gpu = self.get_state_gpu(params)
        expectations_gpu = self.compute_expectations_gpu(state_gpu)
        
        if self.use_gpu:
            expectations = cp.asnumpy(expectations_gpu)
        else:
            expectations = expectations_gpu
        
        sequence = np.sign(expectations).astype(np.int8)
        sequence[sequence == 0] = 1
        
        return sequence, labs_energy(sequence)
    
    def print_timing(self):
        total = self.time_circuit + self.time_expectations + self.time_loss
        if total == 0:
            print("No timing data available")
            return
        print(f"Timing breakdown ({self.call_count} calls):")
        print(f"  circuit:      {self.time_circuit:.2f}s ({100*self.time_circuit/total:.1f}%)")
        print(f"  expectations: {self.time_expectations:.2f}s ({100*self.time_expectations/total:.1f}%)")
        print(f"  loss:         {self.time_loss:.2f}s ({100*self.time_loss/total:.1f}%)")
        print(f"  per call:     {1000*total/self.call_count:.2f}ms")
    
    def _single_restart(self, restart_id: int, maxiter: int, method: str) -> tuple:
        """Run a single optimization restart. Used for parallel execution."""
        params_init = np.random.uniform(-np.pi, np.pi, self.n_params)
        
        try:
            result = minimize(
                self.objective,
                params_init,
                method=method,
                options={'maxiter': maxiter, 'disp': False}
            )
            sequence, energy = self.extract_sequence(result.x)
            return restart_id, sequence, energy, result.x, result.fun
        except Exception as e:
            return restart_id, None, float('inf'), None, float('inf')
    
    def optimize(self, n_restarts: int = 50, maxiter: int = 100, 
                 verbose: bool = False, method: str = 'COBYLA',
                 n_parallel: int = 1, use_processes: bool = False) -> tuple:
        """Optimize with optional parallel restarts.
        
        Args:
            n_restarts: Number of random restarts
            maxiter: Max iterations per restart
            verbose: Show progress
            method: Optimization method
            n_parallel: Number of parallel restarts (1=sequential, >1=parallel)
            use_processes: Use ProcessPoolExecutor instead of threads (bypasses GIL)
        """
        
        if n_parallel > 1:
            return self._optimize_parallel(n_restarts, maxiter, verbose, method, n_parallel, use_processes)
        
        # Sequential optimization (original behavior)
        if verbose and TQDM_AVAILABLE:
            pbar = tqdm(range(n_restarts), desc="Restarts", unit="restart")
        else:
            pbar = range(n_restarts)
        
        for restart in pbar:
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
                    
                    if verbose and TQDM_AVAILABLE:
                        pbar.set_postfix({
                            'best_E': self.best_energy,
                            'calls': self.call_count
                        })
                    elif verbose:
                        print(f"Restart {restart + 1}/{n_restarts}: energy={energy}, calls={self.call_count}")
                    
                    if energy == OPTIMAL_ENERGIES.get(self.N, -1):
                        if verbose:
                            if TQDM_AVAILABLE:
                                pbar.set_description("Found optimal!")
                            else:
                                print(f"Found optimal!")
                        break
                
            except Exception as e:
                if verbose and not TQDM_AVAILABLE:
                    print(f"Restart {restart + 1} failed: {e}")
                continue
        
        return self.best_sequence, int(self.best_energy) if self.best_energy < float('inf') else -1
    
    def _optimize_parallel(self, n_restarts: int, maxiter: int, 
                          verbose: bool, method: str, n_parallel: int,
                          use_processes: bool = False) -> tuple:
        """Run parallel restarts using ThreadPool or ProcessPool."""
        
        completed = 0
        executor_type = "Procs" if use_processes else "Threads"
        if verbose and TQDM_AVAILABLE:
            pbar = tqdm(total=n_restarts, desc=f"{executor_type}({n_parallel})", unit="restart")
        
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
                
                if verbose and TQDM_AVAILABLE:
                    pbar.update(1)
                    pbar.set_postfix({
                        'best_E': self.best_energy if self.best_energy < float('inf') else '?',
                    })
                elif verbose:
                    print(f"Completed {completed}/{n_restarts}, best_E={self.best_energy}")
        
        if verbose and TQDM_AVAILABLE:
            pbar.close()
        
        return self.best_sequence, int(self.best_energy) if self.best_energy < float('inf') else -1
