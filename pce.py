import cudaq
import numpy as np
from scipy.optimize import minimize
from itertools import combinations, product
from functools import reduce
import time
import cupy as cp

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
    def __init__(self, pauli_strings: list, use_gpu: bool = True):
        self.pauli_strings = pauli_strings
        self.n_paulis = len(pauli_strings)
        self.n_qubits = len(pauli_strings[0])
        self.dim = 2 ** self.n_qubits
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
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
                 k: int = 2, alpha: float = 0.0, beta: float = 15.0):
        self.N = N
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.k = k
        self.alpha = alpha if alpha > 0 else 1.5 * n_qubits
        self.beta = beta
        self.n_params = get_n_params(n_qubits, n_layers)
        
        self.pauli_strings = build_kbody_paulis(n_qubits, k, N)
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
        state = cudaq.get_state(ansatz, self.n_qubits, list(params), self.n_layers)
        return np.array(state, dtype=np.complex128)
    
    def objective(self, params: np.ndarray) -> float:
        self.call_count += 1
        
        t0 = time.perf_counter()
        state = self.get_state(params)
        self.time_get_state += time.perf_counter() - t0
        
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
        state = self.get_state(params)
        expectations = self.expectation_calc.compute_expectations(state)
        
        sequence = np.sign(expectations).astype(np.int8)
        sequence[sequence == 0] = 1
        
        return sequence, labs_energy(sequence)
    
    def print_timing(self):
        total = self.time_get_state + self.time_expectations + self.time_loss
        print(f"Timing breakdown ({self.call_count} calls):")
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
