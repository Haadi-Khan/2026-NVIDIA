import pytest
from main import run_single
from src.constants import ENERGY_DATA


TEST_QUBITS = 10
TEST_LAYERS = 5
TEST_RESTARTS = 30
TEST_MAXITER = 300
TEST_METHOD = "COBYLA"
TEST_TABU_ITERS = 20000


# Auto generates test suite from N=2 to N=10
@pytest.mark.parametrize("N", range(2, 11))
def test_energy_matches_known_optimal(N: int):
    expected_energy = ENERGY_DATA[N]["energy"]

    _, _, actual_energy = run_single(
        N=N,
        n_qubits=TEST_QUBITS,
        n_layers=TEST_LAYERS,
        n_restarts=TEST_RESTARTS,
        maxiter=TEST_MAXITER,
        method=TEST_METHOD,
        verbose=False,
        use_tabu=True,
        tabu_iterations=TEST_TABU_ITERS,
    )

    assert actual_energy == expected_energy, (
        f"Energy mismatch for N={N}: got {actual_energy}, expected {expected_energy}"
    )
