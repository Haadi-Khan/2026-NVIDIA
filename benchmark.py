import cudaq
import numpy as np
import time

cudaq.set_target('nvidia')

@cudaq.kernel
def simple_ansatz(n_qubits: int, params: list[float], n_layers: int):
    qubits = cudaq.qvector(n_qubits)
    for q in range(n_qubits):
        h(qubits[q])
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            ry(params[idx], qubits[q])
            idx += 1
        for q in range(0, n_qubits - 1, 2):
            x.ctrl(qubits[q], qubits[q + 1])

n_qubits = 10
n_layers = 5
n_params = n_layers * n_qubits
params = np.random.uniform(-np.pi, np.pi, n_params).tolist()

print(f"Qubits: {n_qubits}, Layers: {n_layers}, Params: {n_params}")
print(f"State dim: {2**n_qubits}")

for _ in range(3):
    state = cudaq.get_state(simple_ansatz, n_qubits, params, n_layers)
print("Warmup done")

n_calls = 100
t0 = time.perf_counter()
for _ in range(n_calls):
    state = cudaq.get_state(simple_ansatz, n_qubits, params, n_layers)
elapsed = time.perf_counter() - t0
print(f"get_state: {1000*elapsed/n_calls:.2f}ms per call ({n_calls} calls)")
