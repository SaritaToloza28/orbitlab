import numpy as np
import pytest

from physics.collisions import (
    MAX_PLANET_RADIUS,
    PLANET_RADIUS,
    STAR_RADIUS,
    body_radius,
    find_collisions,
    merge,
)
from physics.integrators import Merge, simulate
from physics.scenarios import SCENARIOS


def test_radius_of_sun_and_earth():
    np.testing.assert_allclose(body_radius([1.0, 3e-6]), [STAR_RADIUS, PLANET_RADIUS])


def test_radius_grows_with_volume():
    # Dos Tierras juntas: el doble de volumen, 2^(1/3) ≈ 1,26 veces el radio.
    assert body_radius([6e-6])[0] == pytest.approx(PLANET_RADIUS * 2 ** (1 / 3))


def test_radius_has_no_jumps():
    # Un planeta gigante deja de crecer y luego manda la ley de las estrellas.
    masses = np.geomspace(1e-7, 10, 2000)
    radii = body_radius(masses)
    assert np.all(np.diff(radii) >= 0)
    assert np.max(np.diff(radii) / radii[:-1]) < 0.01
    assert body_radius([1e-3])[0] == MAX_PLANET_RADIUS


def test_radius_of_absorbed_body_is_zero():
    assert body_radius([0.0])[0] == 0.0


def test_finds_contact_at_end_of_step():
    old = np.array([[0.0, 0.0], [1.0, 0.0]])
    new = np.array([[0.0, 0.0], [0.05, 0.0]])
    radii = body_radius([1.0, 1.0])

    assert find_collisions(old, new, radii, np.array([True, True])) == [(0, 1)]


def test_finds_body_that_crosses_through_in_one_step():
    # Al inicio y al final están lejos, pero en el medio del paso se cruzan.
    old = np.array([[0.0, 0.0], [-1.0, 0.01]])
    new = np.array([[0.0, 0.0], [1.0, 0.01]])
    radii = body_radius([1.0, 3e-6])

    assert find_collisions(old, new, radii, np.array([True, True])) == [(0, 1)]


def test_ignores_bodies_that_stay_apart():
    old = np.array([[0.0, 0.0], [1.0, 0.0]])
    new = np.array([[0.0, 0.0], [1.0, 0.5]])
    radii = body_radius([1.0, 1.0])

    assert find_collisions(old, new, radii, np.array([True, True])) == []


def test_ignores_absorbed_bodies():
    positions = np.zeros((2, 2))
    radii = body_radius([1.0, 1.0])

    assert find_collisions(positions, positions, radii, np.array([True, False])) == []


def test_collisions_are_sorted_by_time_of_contact():
    # El cuerpo 2 llega a tocar al 0 antes que el 1.
    old = np.array([[0.0, 0.0], [0.0, 1.0], [0.3, 0.0]])
    new = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    radii = body_radius([1.0, 1.0, 1.0])

    hits = find_collisions(old, new, radii, np.ones(3, dtype=bool))

    assert hits[0] == (0, 2)


def test_merge_conserves_mass_and_momentum():
    positions = np.array([[0.0, 0.0], [0.1, 0.0]])
    velocities = np.array([[1.0, 0.0], [-2.0, 3.0]])
    masses = np.array([3.0, 1.0])
    momentum = masses @ velocities

    survivor, absorbed = merge(positions, velocities, masses, 0, 1)

    assert (survivor, absorbed) == (0, 1)
    assert masses.tolist() == [4.0, 0.0]
    np.testing.assert_allclose(masses @ velocities, momentum)
    # En el centro de masas: (3 · 0 + 1 · 0,1) / 4.
    np.testing.assert_allclose(positions[0], [0.025, 0.0])
    # El absorbido queda pegado al superviviente.
    np.testing.assert_array_equal(positions[1], positions[0])


def test_heavier_body_survives_whatever_its_index():
    positions = np.array([[0.0, 0.0], [0.1, 0.0]])
    velocities = np.zeros((2, 2))
    masses = np.array([3e-6, 1.0])

    assert merge(positions, velocities, masses, 0, 1) == (1, 0)


def head_on(m1: float, m2: float, steps: int = 2000):
    """Dos cuerpos que se sueltan quietos a 2 UA y caen uno sobre el otro."""
    positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    velocities = np.array([[0.0, 0.5], [0.0, 0.0]])
    masses = np.array([m1, m2])
    result = simulate(positions, velocities, masses, 0.001, steps, save_every=10)
    return result, masses @ velocities / masses.sum()


def test_two_stars_merge_into_one():
    result, center_velocity = head_on(1.0, 1.0)

    assert result.merges == (Merge(frame=36, survivor=0, absorbed=1),)
    # Después del choque los dos fotogramas coinciden: es un solo cuerpo.
    np.testing.assert_array_equal(result.frames[-1][0], result.frames[-1][1])
    # El cuerpo nuevo sigue con la velocidad del centro de masas: en 2 años,
    # (0; 0,25) UA/año lo llevan a (0; 0,5).
    np.testing.assert_allclose(result.frames[-1][0], 2.0 * center_velocity, atol=1e-6)


def test_star_swallows_planet():
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    # Mucho más lento que la órbita circular (2π): cae en espiral hacia la estrella.
    velocities = np.array([[0.0, 0.0], [0.0, 1.0]])
    masses = np.array([1.0, 3e-6])

    result = simulate(positions, velocities, masses, 0.001, 1000, save_every=10)

    assert [(m.survivor, m.absorbed) for m in result.merges] == [(0, 1)]
    # La estrella casi no se movió.
    assert np.linalg.norm(result.frames[-1][0]) < 1e-4


def test_collision_does_not_count_as_energy_error():
    # La energía que se pierde en el choque es física; lo que queda es el
    # error del integrador, que los subpasos mantienen bajo el límite del plan.
    result, _ = head_on(1.0, 1.0)

    assert result.energy_error < 1e-3


def test_fast_collision_after_long_fall_keeps_energy_error_low():
    # Dos estrellas quietas a 4,4 UA y un planeta en órbita circular de una:
    # llegan al choque a unas 15 UA/año. Con la fracción fija de subpaso el
    # error era 0,18 %; el tramo final más fino lo baja del límite del plan.
    positions = np.array([[-2.940, 1.162], [0.994, -0.805], [-0.742, -0.183]])
    velocities = np.array([[0.0, 0.0], [0.0, 0.0], [-1.560, -4.355]])
    masses = np.array([1.0, 1.0, 3e-6])

    result = simulate(positions, velocities, masses, 0.001, 5000, save_every=5)

    assert [(m.survivor, m.absorbed) for m in result.merges] == [(0, 1)]
    assert result.energy_error < 5e-4


def test_body_absorbed_in_a_chain_follows_the_last_survivor():
    # Tres estrellas en línea que caen hacia el centro.
    positions = np.array([[-1.0, 0.0], [0.0, 0.0], [1.2, 0.0]])
    velocities = np.zeros((3, 2))
    masses = np.array([1.0, 2.0, 1.0])

    result = simulate(positions, velocities, masses, 0.001, 2000, save_every=10)

    assert len(result.merges) == 2
    assert all(m.survivor == 1 for m in result.merges)
    np.testing.assert_array_equal(result.frames[-1][0], result.frames[-1][1])
    np.testing.assert_array_equal(result.frames[-1][2], result.frames[-1][1])


def test_simulate_does_not_modify_masses():
    masses = np.array([1.0, 1.0])

    simulate(
        np.array([[-1.0, 0.0], [1.0, 0.0]]), np.zeros((2, 2)), masses, 0.001, 1000, 10
    )

    assert masses.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenarios_have_no_collisions(name):
    scenario = SCENARIOS[name]
    bodies = scenario["bodies"]

    result = simulate(
        np.array([[b["x"], b["y"]] for b in bodies]),
        np.array([[b["vx"], b["vy"]] for b in bodies]),
        np.array([b["mass"] for b in bodies]),
        scenario["dt"],
        scenario["steps"],
        scenario["save_every"],
    )

    assert result.merges == ()
