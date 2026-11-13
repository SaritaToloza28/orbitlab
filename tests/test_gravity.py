import numpy as np
import pytest

from physics.gravity import G, accelerations

# Suavizado casi nulo para comparar con las cuentas hechas a mano.
TINY = 1e-9


def accelerations_with_loops(positions, masses, g, softening):
    """Versión lenta con dos bucles, como referencia para la vectorizada."""
    n = len(masses)
    result = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            arrow = positions[j] - positions[i]
            dist_sq = arrow @ arrow + softening**2
            result[i] += g * masses[j] * arrow / dist_sq**1.5
    return result


def test_three_bodies_in_a_line():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    masses = np.ones(3)

    acc = accelerations(positions, masses, g=1.0, softening=TINY)

    expected = np.array([[1 + 1 / 9, 0.0], [-1 + 1 / 4, 0.0], [-1 / 9 - 1 / 4, 0.0]])
    np.testing.assert_allclose(acc, expected, atol=1e-12)


def test_triangle_3_4_5():
    positions = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    masses = np.ones(3)

    acc = accelerations(positions, masses, g=1.0, softening=TINY)

    expected = np.array(
        [
            [3 / 125 + 3 / 27, 4 / 125],
            [-3 / 125, -4 / 125 - 4 / 64],
            [-3 / 27, 4 / 64],
        ]
    )
    np.testing.assert_allclose(acc, expected, atol=1e-12)


def test_sun_earth_acceleration():
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    masses = np.array([1.0, 3e-6])

    acc = accelerations(positions, masses)

    # La Tierra cae hacia el Sol con aceleración G · 1 / 1².
    assert acc[1] == pytest.approx([-G, 0.0], rel=1e-5)
    # El Sol apenas se mueve, hacia la Tierra.
    assert acc[0] == pytest.approx([G * 3e-6, 0.0], rel=1e-5)


def test_net_force_is_zero():
    rng = np.random.default_rng(42)
    positions = rng.uniform(-5, 5, size=(10, 2))
    masses = rng.uniform(0.1, 2, size=10)

    acc = accelerations(positions, masses)

    # Acción y reacción: la suma de masa × aceleración se cancela.
    np.testing.assert_allclose(masses @ acc, [0.0, 0.0], atol=1e-10)


def test_matches_loop_version():
    rng = np.random.default_rng(7)
    positions = rng.uniform(-5, 5, size=(6, 2))
    masses = rng.uniform(0.1, 2, size=6)

    acc = accelerations(positions, masses)

    expected = accelerations_with_loops(positions, masses, G, 1e-3)
    np.testing.assert_allclose(acc, expected, rtol=1e-12)


def test_overlapping_bodies_do_not_explode():
    positions = np.array([[1.0, 1.0], [1.0, 1.0]])
    masses = np.ones(2)

    acc = accelerations(positions, masses)

    assert np.all(np.isfinite(acc))
