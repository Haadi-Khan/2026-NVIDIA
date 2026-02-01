# PCE Solver Source Module

This module implements the **Pauli Correlation Encoding (PCE)** solver for the Low Autocorrelation Binary Sequences (LABS) problem using GPU-accelerated variational quantum circuits.

## Overview

The PCE solver uses a quantum-enhanced optimization approach that encodes LABS sequences as Pauli expectation values from a parameterized quantum circuit. The solver leverages NVIDIA's CUDA-Q for GPU acceleration and CuPy for GPU-native computation.

## Module Structure

### Core Components

- **[solver.py](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/solver.py)** - Main `PCESolver` class
  - GPU-native state vector computation
  - Multiple optimization methods (COBYLA, L-BFGS-B, Adam, QNG, SPSA variants)
  - Parameter-shift and SPSA gradient computation
  - Parallel restart optimization
  - Tabu search refinement

- **[circuit.py](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/circuit.py)** - Quantum circuit ansatz
  - Hardware-efficient ansatz with RY rotations and CNOT entanglement
  - CUDA-Q kernel implementation

- **[paulis.py](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/paulis.py)** - Pauli string generation
  - k-body commuting Pauli sets
  - Non-commuting Pauli sets for enhanced expressivity

- **[energy.py](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/energy.py)** - LABS energy functions
  - `labs_energy(s)` - Compute LABS energy E(s) = Σ C_k²
  - `merit_factor(s)` - Compute merit factor F = N² / (2E)

- **[search.py](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/search.py)** - Classical refinement algorithms
  - Greedy local search
  - Tabu search for escaping local minima

- **[constants.py](file:///C:/Users/arulr/Projects/IQuHacks/2026-NVIDIA/src/constants.py)** - Optimal LABS energies database
  - Known optimal energies for N ≤ 82
  - GPU availability detection

## Key Features

### 1. GPU-Native Computation
- State vectors kept on GPU (CuPy arrays)
- Avoids CPU-GPU transfer overhead
- All Pauli expectations computed on GPU

### 2. Multiple Optimization Methods

**Derivative-Free:**
- `COBYLA` - Baseline constrained optimization

**Gradient-Based (Parameter-Shift):**
- `L-BFGS-B` - Quasi-Newton with parameter-shift gradients (O(2p) cost)
- `Adam` - Adaptive momentum with parameter-shift gradients

**Gradient-Based (SPSA):**
- `SPSA` - Stochastic gradient estimation (O(2) cost)
- `SPSA-Adam` - SPSA + Adam momentum
- `SPSA-CG` - SPSA + conjugate gradient
- `SPSA-CG-Wolfe` - SPSA + CG + strong Wolfe line search

**Quantum Natural Gradient:**
- `QNG` - Diagonal metric approximation (O(3p) cost)

### 3. Pauli Set Options

**Commuting k-body Paulis** (`use_noncommuting=False`):
- Constructed from k-local terms
- Guaranteed to commute

**Non-commuting Paulis** (`use_noncommuting=True`):
- Enhanced expressivity
- Better optimization landscape

### 4. Hybrid Refinement
- Quantum circuit produces candidate solutions
- Tabu search refines solutions classically
- Combines quantum exploration with classical exploitation

## Usage Example

```python
from src import PCESolver, get_optimal_energy, merit_factor

# Initialize solver
solver = PCESolver(
    N=13,                      # Sequence length
    n_qubits=10,              # Number of qubits
    n_layers=5,               # Ansatz depth
    use_noncommuting=True     # Use non-commuting Paulis
)

# Run optimization
sequence, energy = solver.optimize(
    n_restarts=50,            # Random restarts
    maxiter=100,              # Iterations per restart
    method='COBYLA',          # Optimization method
    use_tabu=True,            # Enable tabu refinement
    tabu_iterations=20000,    # Tabu search budget
    n_parallel=4,             # Parallel workers
    verbose=True
)

# Check results
optimal = get_optimal_energy(13)
print(f"Energy: {energy} (optimal: {optimal})")
print(f"Merit factor: {merit_factor(sequence):.4f}")
print(f"Success: {energy == optimal}")
```

## Performance Characteristics

### Scaling
- **State dimension**: 2^n_qubits (exponential in qubits)
- **Parameters**: n_qubits × n_layers (linear in both)
- **GPU speedup**: ~10-100x vs CPU for n_qubits ≥ 10

### Typical Configurations
- **Small problems (N ≤ 20)**: 8 qubits, 3 layers, 10 restarts
- **Medium problems (N ≤ 40)**: 10 qubits, 5 layers, 50 restarts
- **Large problems (N ≤ 80)**: 14 qubits, 10 layers, 50 restarts

## Dependencies

- `cudaq` - NVIDIA CUDA-Q for quantum simulation
- `cupy` - GPU-accelerated NumPy
- `numpy` - Numerical computing
- `scipy` - Optimization algorithms
- `tqdm` - Progress bars

## References

The PCE approach is based on encoding combinatorial optimization problems into quantum expectation values, enabling quantum-enhanced optimization through variational quantum circuits.
