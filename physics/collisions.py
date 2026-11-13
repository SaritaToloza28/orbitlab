"""Choques entre cuerpos: tamaño de cada uno, detección y fusión.

Los radios están exagerados a propósito. Los reales son diminutos en UA (el
Sol mide 0,0047 UA), y con un paso de tiempo fijo dos cuerpos se acercarían
tanto antes de tocarse que la gravedad los lanzaría a velocidades absurdas.
Con radios más grandes se fusionan antes de llegar a ese punto. Fuera de un
cuerpo su gravedad es la misma que la de un punto, así que exagerar el radio
solo cambia cuándo chocan, no cómo se atraen.

La interfaz (static/index.html) repite body_radius() para dibujar cada cuerpo
de su tamaño real en la simulación: si se cambia aquí, hay que cambiarla allí.
"""

import numpy as np

# Radio (UA) de un cuerpo de 1 masa solar. Crece con la raíz cúbica de la
# masa, como una esfera de densidad fija.
STAR_RADIUS = 0.05
# Radio (UA) de un planeta de la masa de la Tierra, con la misma ley.
EARTH_MASS = 3e-6
PLANET_RADIUS = 0.012
# Como en los gigantes gaseosos reales, a partir de cierta masa un planeta ya
# no crece; desde ahí manda la ley de las estrellas, sin saltos.
MAX_PLANET_RADIUS = 0.02


def body_radius(masses: np.ndarray) -> np.ndarray:
    """Radio exagerado (UA) de cada cuerpo según su masa. Masa 0 da radio 0."""
    masses = np.asarray(masses, dtype=float)
    as_star = STAR_RADIUS * np.cbrt(masses)
    as_planet = np.minimum(
        PLANET_RADIUS * np.cbrt(masses / EARTH_MASS), MAX_PLANET_RADIUS
    )
    return np.where(masses > 0, np.maximum(as_star, as_planet), 0.0)


def find_collisions(
    old_positions: np.ndarray,
    new_positions: np.ndarray,
    radii: np.ndarray,
    alive: np.ndarray,
) -> list[tuple[int, int]]:
    """Parejas de cuerpos vivos que se tocaron durante el último paso.

    No basta con mirar el final del paso: un cuerpo rápido puede atravesar a
    otro entre una foto y la siguiente. Por eso se supone que durante el paso
    cada cuerpo va en línea recta y se busca el instante t ∈ [0, 1] en que la
    distancia de la pareja baja a r_i + r_j: |s + t·m|² = R², donde s es la
    separación al inicio y m cuánto cambió.

    Devuelve (i, j) con i < j, ordenadas por el instante del contacto.
    """
    # Todas las parejas i < j a la vez, como en gravity.py.
    i, j = np.triu_indices(len(radii), k=1)
    start = old_positions[j] - old_positions[i]
    motion = (new_positions[j] - new_positions[i]) - start
    contact_sq = (radii[i] + radii[j]) ** 2

    # Ecuación de segundo grado a·t² + b·t + c = 0.
    a = np.sum(motion**2, axis=-1)
    b = 2 * np.sum(start * motion, axis=-1)
    c = np.sum(start**2, axis=-1) - contact_sq
    disc = b**2 - 4 * a * c
    # La primera raíz es cuando empiezan a tocarse. Si a = 0 no se movieron.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    # Ya se tocaban al empezar (c ≤ 0) o se tocan dentro del paso.
    already = c <= 0
    within = (a > 0) & (disc >= 0) & (t >= 0) & (t <= 1)
    touching = (already | within) & alive[i] & alive[j]
    t = np.where(already, 0.0, t)

    order = np.argsort(t[touching], kind="stable")
    return [(int(p), int(q)) for p, q in zip(i[touching][order], j[touching][order])]


def merge(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    i: int,
    j: int,
) -> tuple[int, int]:
    """Fusiona los cuerpos i y j en el lugar (modifica los arrays).

    Choque perfectamente inelástico: la masa se suma, el momento se conserva
    y el cuerpo nuevo queda en el centro de masas de los dos. Sobrevive el
    más masivo (con masas iguales, el de menor índice) y el otro queda con
    masa 0. Devuelve (superviviente, absorbido).
    """
    survivor, absorbed = (i, j) if masses[i] >= masses[j] else (j, i)
    total = masses[i] + masses[j]
    positions[survivor] = (masses[i] * positions[i] + masses[j] * positions[j]) / total
    velocities[survivor] = (
        masses[i] * velocities[i] + masses[j] * velocities[j]
    ) / total
    masses[survivor] = total
    masses[absorbed] = 0.0
    # El absorbido acompaña al superviviente: sus fotogramas siguen existiendo
    # (el array tiene siempre N cuerpos), pero ya no pesa ni se dibuja.
    positions[absorbed] = positions[survivor]
    velocities[absorbed] = velocities[survivor]
    return survivor, absorbed
