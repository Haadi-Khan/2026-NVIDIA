"""Quantum circuit simulation for the PCE solver - CPU Version.

This module provides a pure NumPy statevector simulator that replicates
the functionality of CUDA-Q for the hardware-efficient ansatz.
"""

import numpy as np


def get_n_params(n_qubits: int, n_layers: int) -> int:
    """Calculate the number of parameters for the ansatz.

    Each layer has 2 parameters (RY, RZ) per qubit.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        Total number of parameters
    """
    return n_layers * 2 * n_qubits


class StatevectorSimulator:
    """Pure NumPy statevector simulator for the hardware-efficient ansatz.
    
    Implements the same circuit structure as the CUDA-Q version:
    - Initial Hadamard gates on all qubits
    - Per-layer: RY and RZ rotations, followed by brick-pattern CNOTs
    """

    def __init__(self, n_qubits: int):
        """Initialize the simulator.
        
        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
        # Precompute single-qubit gate matrices
        self._h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
    def _apply_single_qubit_gate(
        self, state: np.ndarray, gate: np.ndarray, qubit: int
    ) -> np.ndarray:
        """Apply a single-qubit gate to the statevector.
        
        Args:
            state: Current statevector (dim,)
            gate: 2x2 gate matrix
            qubit: Target qubit index
            
        Returns:
            Updated statevector
        """
        # Reshape to tensor form for efficient einsum
        n = self.n_qubits
        shape = [2] * n
        state_tensor = state.reshape(shape)
        
        # Apply gate using einsum
        # The qubit index in the tensor is n - 1 - qubit (reverse order)
        axes = list(range(n))
        axes[qubit], axes[-1] = axes[-1], axes[qubit]
        
        state_tensor = np.transpose(state_tensor, axes)
        original_shape = state_tensor.shape
        state_tensor = state_tensor.reshape(-1, 2)
        state_tensor = state_tensor @ gate.T
        state_tensor = state_tensor.reshape(original_shape)
        state_tensor = np.transpose(state_tensor, axes)
        
        return state_tensor.reshape(-1)
    
    def _ry(self, theta: float) -> np.ndarray:
        """RY rotation gate matrix."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    
    def _rz(self, theta: float) -> np.ndarray:
        """RZ rotation gate matrix."""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate.
        
        Args:
            state: Current statevector
            control: Control qubit index
            target: Target qubit index
            
        Returns:
            Updated statevector
        """
        new_state = state.copy()
        
        # For each basis state, flip target if control is 1
        for i in range(self.dim):
            if (i >> control) & 1:  # Control qubit is 1
                # Flip the target qubit
                j = i ^ (1 << target)
                new_state[i], new_state[j] = state[j], state[i]
        
        return new_state
    
    def run_ansatz(self, params: np.ndarray, n_layers: int) -> np.ndarray:
        """Run the hardware-efficient ansatz circuit.
        
        Structure per layer:
        1. Single-qubit RY and RZ rotations on each qubit
        2. Entangling CNOT gates in a brick-layer pattern
        
        Args:
            params: Flattened parameter array (n_layers * 2 * n_qubits,)
            n_layers: Number of variational layers
            
        Returns:
            Final statevector (dim,)
        """
        # Start with |0...0⟩
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0
        
        # Apply Hadamard to all qubits (initial equal superposition)
        for q in range(self.n_qubits):
            state = self._apply_single_qubit_gate(state, self._h, q)
        
        # Variational layers
        idx = 0
        for layer in range(n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                ry_gate = self._ry(params[idx])
                rz_gate = self._rz(params[idx + 1])
                state = self._apply_single_qubit_gate(state, ry_gate, q)
                state = self._apply_single_qubit_gate(state, rz_gate, q)
                idx += 2
            
            # Entangling layer (brick pattern)
            # Even pairs: (0,1), (2,3), (4,5), ...
            for q in range(0, self.n_qubits - 1, 2):
                state = self._apply_cnot(state, q, q + 1)
            
            # Odd pairs: (1,2), (3,4), (5,6), ...
            for q in range(1, self.n_qubits - 1, 2):
                state = self._apply_cnot(state, q, q + 1)
        
        return state


def get_state(n_qubits: int, params: list, n_layers: int) -> np.ndarray:
    """Get the statevector from running the ansatz.
    
    This is the main entry point, matching cudaq.get_state() interface.
    
    Args:
        n_qubits: Number of qubits
        params: List of parameters
        n_layers: Number of variational layers
        
    Returns:
        Statevector as numpy array (2^n_qubits,)
    """
    simulator = StatevectorSimulator(n_qubits)
    params_array = np.asarray(params, dtype=np.float64)
    return simulator.run_ansatz(params_array, n_layers)
