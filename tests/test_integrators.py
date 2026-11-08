import numpy as np
import pytest

from physics.gravity import accelerations
from physics.integrators import energy, euler_step, simulate, verlet_step

# Suavizado casi nulo para comparar con las cuentas hechas a mano.
TINY = 1e-9

# Sol en el origen y Tierra a 1 UA con velocidad de órbita circular (2π UA/año).
SUN_EARTH_POSITIONS = np.array([[0.0, 0.0], [1.0, 0.0]])
SUN_EARTH_VELOCITIES = np.array([[0.0, 0.0], [0.0, 2 * np.pi]])
SUN_EARTH_MASSES = np.array([1.0, 3e-6])


def test_energy_of_two_bodies_at_rest():
    positions = np.array([[0.0, 0.0], [2.0, 0.0]])
    velocities = np.zeros((2, 2))
    masses = np.ones(2)

    # Sin movimiento solo hay potencial: -G · 1 · 1 / 2.
    assert energy(
        positions, velocities, masses, g=1.0, softening=TINY
    ) == pytest.approx(-0.5)


def test_energy_counts_each_pair_once():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    velocities = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    masses = np.ones(3)

    # Cinética: ½·1² + ½·2² = 2,5. Potencial: -(1/1 + 1/3 + 1/2).
    expected = 2.5 - (1 + 1 / 3 + 1 / 2)
    assert energy(
        positions, velocities, masses, g=1.0, softening=TINY
    ) == pytest.approx(expected)


def test_verlet_step_returns_acceleration_at_new_positions():
    acc = accelerations(SUN_EARTH_POSITIONS, SUN_EARTH_MASSES)

    new_positions, _, new_acc = verlet_step(
        SUN_EARTH_POSITIONS, SUN_EARTH_VELOCITIES, acc, SUN_EARTH_MASSES, dt=0.01
    )

    # La aceleración devuelta debe ser la de la llegada, lista para reutilizarse.
    np.testing.assert_allclose(new_acc, accelerations(new_positions, SUN_EARTH_MASSES))


def test_verlet_conserves_energy_sun_earth():
    result = simulate(
        SUN_EARTH_POSITIONS,
        SUN_EARTH_VELOCITIES,
        SUN_EARTH_MASSES,
        dt=0.001,
        steps=1000,
        save_every=10,
    )

    # Requisito del plan: error relativo de energía < 0,1 % en Sol-Tierra.
    assert result.energy_error < 1e-3


def test_verlet_conserves_energy_on_eccentric_orbit():
    # Velocidad menor que la circular: órbita elíptica, más difícil de integrar.
    velocities = np.array([[0.0, 0.0], [0.0, 5.0]])

    result = simulate(
        SUN_EARTH_POSITIONS,
        velocities,
        SUN_EARTH_MASSES,
        dt=0.001,
        steps=10_000,
        save_every=100,
    )

    assert result.energy_error < 1e-3


def test_orbit_closes_after_one_year():
    result = simulate(
        SUN_EARTH_POSITIONS,
        SUN_EARTH_VELOCITIES,
        SUN_EARTH_MASSES,
        dt=0.001,
        steps=1000,
        save_every=10,
    )

    # Tras un año la Tierra vuelve a su punto de partida.
    np.testing.assert_allclose(result.frames[-1][1], [1.0, 0.0], atol=1e-3)


def test_euler_drifts_more_than_verlet():
    dt, steps = 0.001, 1000
    initial_energy = energy(SUN_EARTH_POSITIONS, SUN_EARTH_VELOCITIES, SUN_EARTH_MASSES)

    positions, velocities = SUN_EARTH_POSITIONS, SUN_EARTH_VELOCITIES
    for _ in range(steps):
        positions, velocities = euler_step(positions, velocities, SUN_EARTH_MASSES, dt)
    euler_error = abs(energy(positions, velocities, SUN_EARTH_MASSES) - initial_energy)
    euler_error /= abs(initial_energy)

    verlet_error = simulate(
        SUN_EARTH_POSITIONS,
        SUN_EARTH_VELOCITIES,
        SUN_EARTH_MASSES,
        dt=dt,
        steps=steps,
        save_every=steps,
    ).energy_error

    # Euler pierde más del 1 % en un año; Verlet, prácticamente nada.
    assert euler_error > 1e-2
    assert euler_error > 1000 * verlet_error


def test_simulate_frames():
    result = simulate(
        SUN_EARTH_POSITIONS,
        SUN_EARTH_VELOCITIES,
        SUN_EARTH_MASSES,
        dt=0.001,
        steps=100,
        save_every=10,
    )

    # Fotograma 0 (el inicial) + uno cada 10 pasos = 11 fotogramas de 2 cuerpos.
    assert result.frames.shape == (11, 2, 2)
    np.testing.assert_array_equal(result.frames[0], SUN_EARTH_POSITIONS)


def test_simulate_does_not_modify_inputs():
    positions = SUN_EARTH_POSITIONS.copy()
    velocities = SUN_EARTH_VELOCITIES.copy()

    simulate(
        positions, velocities, SUN_EARTH_MASSES, dt=0.001, steps=100, save_every=10
    )

    np.testing.assert_array_equal(positions, SUN_EARTH_POSITIONS)
    np.testing.assert_array_equal(velocities, SUN_EARTH_VELOCITIES)
