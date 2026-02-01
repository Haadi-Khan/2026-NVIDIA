"""Quantum circuit definitions for the PCE solver."""

import cudaq


@cudaq.kernel
def ansatz(n_qubits: int, params: list[float], n_layers: int):
    """Hardware-efficient variational ansatz.

    Structure per layer:
    1. Single-qubit RY and RZ rotations on each qubit
    2. Entangling CNOT gates in a brick-layer pattern

    Initial state: Equal superposition (Hadamard on all qubits)

    Args:
        n_qubits: Number of qubits
        params: Flattened parameter array of shape (n_layers * 2 * n_qubits,)
        n_layers: Number of variational layers
    """
    qubits = cudaq.qvector(n_qubits)

    # Initial equal superposition
    for q in range(n_qubits):
        h(qubits[q])

    # Variational layers
    idx = 0
    for layer in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            ry(params[idx], qubits[q])
            rz(params[idx + 1], qubits[q])
            idx += 2

        # Entangling layer (brick pattern)
        for q in range(0, n_qubits - 1, 2):
            x.ctrl(qubits[q], qubits[q + 1])

        for q in range(1, n_qubits - 1, 2):
            x.ctrl(qubits[q], qubits[q + 1])


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
