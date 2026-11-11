import numpy as np
import pytest

from app.schemas import ExperimentCreate
from physics.integrators import simulate
from physics.scenarios import BINARY, FIGURE_EIGHT, SCENARIOS, SUN_EARTH


def as_arrays(scenario: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posiciones, velocidades y masas de un escenario, como los usa simulate()."""
    bodies = scenario["bodies"]
    positions = np.array([[b["x"], b["y"]] for b in bodies])
    velocities = np.array([[b["vx"], b["vy"]] for b in bodies])
    masses = np.array([b["mass"] for b in bodies])
    return positions, velocities, masses


def run(scenario: dict, steps: int, save_every: int = 1):
    """Simula un escenario con su dt, pero con los pasos que pida el test."""
    positions, velocities, masses = as_arrays(scenario)
    return simulate(positions, velocities, masses, scenario["dt"], steps, save_every)


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_passes_api_validation(name):
    # Un escenario debe cumplir los mismos límites que un experimento del cliente.
    ExperimentCreate.model_validate(SCENARIOS[name])


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_conserves_energy(name):
    scenario = SCENARIOS[name]

    result = run(scenario, scenario["steps"], scenario["save_every"])

    # Requisito del plan: error relativo de energía < 0,1 %.
    assert result.energy_error < 1e-3


@pytest.mark.parametrize("scenario", [BINARY, FIGURE_EIGHT])
def test_center_of_mass_does_not_move(scenario):
    _, velocities, masses = as_arrays(scenario)

    # Momento total cero: el sistema gira en su lugar y no se desplaza.
    np.testing.assert_allclose(masses @ velocities, [0.0, 0.0], atol=1e-12)


def test_sun_earth_closes_after_one_year():
    result = run(SUN_EARTH, steps=1000)

    np.testing.assert_allclose(result.frames[-1][1], [1.0, 0.0], atol=1e-3)


def test_binary_keeps_its_separation():
    # Una vuelta completa: período 1/√2 años con dt = 0,001.
    result = run(BINARY, steps=707)

    separation = np.linalg.norm(result.frames[:, 1] - result.frames[:, 0], axis=1)
    # Órbita circular: la distancia entre las estrellas no cambia.
    np.testing.assert_allclose(separation, 1.0, rtol=1e-3)
    positions, _, _ = as_arrays(BINARY)
    np.testing.assert_allclose(result.frames[-1], positions, atol=1e-2)


def test_figure_eight_returns_to_start():
    # Período: 6,3259 / 2π ≈ 1,0068 años, unos 1007 pasos con dt = 0,001.
    result = run(FIGURE_EIGHT, steps=1007)

    positions, _, _ = as_arrays(FIGURE_EIGHT)
    np.testing.assert_allclose(result.frames[-1], positions, atol=1e-2)


def test_figure_eight_bodies_never_collide():
    result = run(FIGURE_EIGHT, steps=FIGURE_EIGHT["steps"], save_every=5)

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        distances = np.linalg.norm(result.frames[:, i] - result.frames[:, j], axis=1)
        # Se acercan, pero nunca a menos de ~0,3 UA: la figura no se desarma.
        assert distances.min() > 0.3
