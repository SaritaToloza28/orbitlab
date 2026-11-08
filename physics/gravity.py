"""Aceleraciones gravitatorias de N cuerpos en 2D.

Unidades: distancia en UA, tiempo en años y masa en masas solares.
Con estas unidades G = 4π², y una órbita a 1 UA del Sol dura 1 año.
"""

import numpy as np

# Constante de gravitación en UA³ / (masa solar · año²).
G = 4 * np.pi**2

# Suavizado (en UA) que evita dividir por cero cuando dos cuerpos coinciden.
DEFAULT_SOFTENING = 1e-3


def accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    g: float = G,
    softening: float = DEFAULT_SOFTENING,
) -> np.ndarray:
    """Calcula la aceleración neta de cada cuerpo por la gravedad de los demás.

    Args:
        positions: array (N, 2) con las posiciones x, y de cada cuerpo.
        masses: array (N,) con la masa de cada cuerpo.
        g: constante de gravitación.
        softening: distancia de suavizado ε; se suma ε² a cada distancia².

    Returns:
        Array (N, 2) con la aceleración x, y de cada cuerpo.
    """
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)

    # Tabla de flechas: diff[i, j] = r_j - r_i, la flecha de i hacia j. (N, N, 2)
    diff = positions[None, :, :] - positions[:, None, :]

    # Distancia² de cada pareja, con suavizado. (N, N)
    dist_sq = np.sum(diff**2, axis=-1) + softening**2

    # 1 / distancia³. En la diagonal diff es 0, así que i no se jala a sí mismo.
    inv_dist_cubed = dist_sq**-1.5

    # Jalón de j sobre i: m_j · flecha / distancia³. (N, N, 2)
    pulls = masses[None, :, None] * diff * inv_dist_cubed[:, :, None]

    # Sumar cada fila junta todos los jalones que recibe i. (N, 2)
    return g * np.sum(pulls, axis=1)
