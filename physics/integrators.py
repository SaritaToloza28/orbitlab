"""Integradores que avanzan en el tiempo un sistema de N cuerpos.

Velocity Verlet es el integrador de la simulación: es simpléctico, así que
el error en la energía oscila pero no crece. Euler queda solo como
comparación en los tests, porque acumula energía y la órbita se abre.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from physics.collisions import body_radius, find_collisions, merge
from physics.gravity import DEFAULT_SOFTENING, G, accelerations

# Función que recibe la fracción completada (de 0 a 1). La física no sabe
# qué hace con ella: la API la usa para guardar el progreso en la base de datos.
ProgressCallback = Callable[[float], None]

# Cuántas veces como máximo se informa el progreso (cada 1 %).
PROGRESS_REPORTS = 100

# El subpaso no puede pasar de esta fracción del tiempo propio de cada
# pareja (ver plan_step). Más chico es más preciso y más lento. Con 0,007 la
# Tierra a 1 UA del Sol con dt = 0,001 sigue en un solo subpaso.
SUBSTEP_FRACTION = 0.007
# En el tramo final antes de un choque (a menos de CLOSE_RANGE veces la
# distancia de contacto) es donde más se pierde precisión, así que ahí se
# usa una fracción más fina. Así dos estrellas que caen una sobre otra desde
# 4 UA chocan con menos de 0,05 % de error en la energía (antes, 0,18 %), sin
# hacer más lentas las órbitas normales: 5 veces el contacto entre dos
# estrellas es 0,5 UA, y el sistema binario las tiene a 1 UA.
CLOSE_SUBSTEP_FRACTION = 0.0025
CLOSE_RANGE = 5
# Tope de subpasos por paso, para que un caso extremo no se cuelgue.
MAX_SUBSTEPS = 1000


class Merge(NamedTuple):
    """Un choque: `absorbed` se fusionó con `survivor`."""

    # Primer fotograma que ya muestra a los dos fusionados.
    frame: int
    survivor: int
    absorbed: int


class SimulationResult(NamedTuple):
    """Resultado de una simulación completa."""

    # Posiciones guardadas cada `save_every` pasos. (n_fotogramas, N, 2)
    # Un cuerpo absorbido sigue apareciendo, pegado a quien lo absorbió.
    frames: np.ndarray
    # |E_final - E_inicial| / |E_inicial|: mide qué tan bien se conservó la energía.
    # No cuenta la energía que se pierde en los choques, que es física y no un error.
    energy_error: float
    # Choques en el orden en que ocurrieron.
    merges: tuple[Merge, ...] = ()


def energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    g: float = G,
    softening: float = DEFAULT_SOFTENING,
) -> float:
    """Energía total del sistema: cinética + potencial de todas las parejas.

    La potencial usa el mismo suavizado que `accelerations`, para que ambas
    describan la misma física y la energía se conserve de verdad.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)

    # Cinética: suma de ½ · m · v².
    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=-1))

    # Distancia suavizada de cada pareja, como en gravity.py. (N, N)
    diff = positions[None, :, :] - positions[:, None, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + softening**2)

    # Potencial: -G · m_i · m_j / distancia, contando cada pareja una sola vez
    # (solo el triángulo superior de la tabla, sin la diagonal).
    i, j = np.triu_indices(len(masses), k=1)
    potential = -g * np.sum(masses[i] * masses[j] / dist[i, j])

    return float(kinetic + potential)


def euler_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    g: float = G,
    softening: float = DEFAULT_SOFTENING,
) -> tuple[np.ndarray, np.ndarray]:
    """Avanza un paso con Euler, usando solo los valores del inicio del paso."""
    acc = accelerations(positions, masses, g, softening)
    new_positions = positions + velocities * dt
    new_velocities = velocities + acc * dt
    return new_positions, new_velocities


def verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    acc: np.ndarray,
    masses: np.ndarray,
    dt: float,
    g: float = G,
    softening: float = DEFAULT_SOFTENING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Avanza un paso con Velocity Verlet.

    Recibe la aceleración actual y devuelve la nueva, para que el paso
    siguiente la reutilice: así se llama a la gravedad una sola vez por paso.
    """
    # Medio empujón con la aceleración del inicio del paso.
    half_velocities = velocities + acc * (dt / 2)
    # Moverse un paso completo con esa velocidad intermedia.
    new_positions = positions + half_velocities * dt
    # Nueva foto de la gravedad en el punto de llegada.
    new_acc = accelerations(new_positions, masses, g, softening)
    # El otro medio empujón, con la aceleración de llegada.
    new_velocities = half_velocities + new_acc * (dt / 2)
    return new_positions, new_velocities, new_acc


class Pairs(NamedTuple):
    """Parejas de cuerpos vivos, con lo que plan_step necesita de cada una.

    Solo cambia cuando hay un choque, así que se calcula una vez y no en cada paso.
    """

    i: np.ndarray
    j: np.ndarray
    # G · (m_i + m_j)
    g_mass: np.ndarray
    # Distancia a la que se tocan: r_i + r_j.
    contact: np.ndarray


def alive_pairs(masses: np.ndarray, alive: np.ndarray, g: float = G) -> Pairs:
    """Arma las parejas i < j entre los cuerpos que siguen vivos."""
    i, j = np.triu_indices(len(masses), k=1)
    keep = alive[i] & alive[j]
    i, j = i[keep], j[keep]
    radii = body_radius(masses)
    return Pairs(i, j, g * (masses[i] + masses[j]), radii[i] + radii[j])


def plan_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    pairs: Pairs,
    dt: float,
) -> tuple[int, bool]:
    """Decide cómo dar el próximo paso: en cuántos subpasos y si hay que vigilar choques.

    Subpasos: cada pareja tiene un tiempo propio, lo que tardaría en caer una
    sobre la otra, sqrt(d³ / (G·M)), y lo que tarda en recorrer su distancia,
    d / v. El subpaso no debe pasar de SUBSTEP_FRACTION del menor, o de
    CLOSE_SUBSTEP_FRACTION si la pareja está por chocar.

    Choques: en un paso, una pareja se acerca como mucho v·dt + ½·a·dt², con
    a acotada por la gravedad en el contacto. Si ninguna puede llegar a
    tocarse, no hace falta buscar choques.

    Se llama en cada paso, así que se trabaja con distancias al cuadrado y se
    evitan operaciones de más: es lo que más pesa en el tiempo total.
    """
    if len(pairs.i) == 0:
        return 1, False
    gap = positions[pairs.j] - positions[pairs.i]
    rel_v = velocities[pairs.j] - velocities[pairs.i]
    dist_sq = np.einsum("pk,pk->p", gap, gap)
    speed_sq = np.einsum("pk,pk->p", rel_v, rel_v)
    dist = np.sqrt(dist_sq)

    reach = (
        pairs.contact
        + np.sqrt(speed_sq) * dt
        + 0.5 * pairs.g_mass / pairs.contact**2 * dt**2
    )
    may_touch = bool((dist <= reach).any())

    # Los tiempos al cuadrado: (d³ / GM) y (d² / v²). El tope evita dividir por 0.
    fall_sq = dist_sq * dist / pairs.g_mass
    cross_sq = dist_sq / np.maximum(speed_sq, 1e-300)
    pair_time = np.sqrt(np.minimum(fall_sq, cross_sq))
    # Cada pareja con su fracción: más fina si está por chocar.
    fraction = np.where(
        dist < CLOSE_RANGE * pairs.contact, CLOSE_SUBSTEP_FRACTION, SUBSTEP_FRACTION
    )
    n_sub = int(np.ceil(dt / (fraction * pair_time).min()))
    return min(max(n_sub, 1), MAX_SUBSTEPS), may_touch


def simulate(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    steps: int,
    save_every: int,
    g: float = G,
    softening: float = DEFAULT_SOFTENING,
    on_progress: ProgressCallback | None = None,
) -> SimulationResult:
    """Simula `steps` pasos con Velocity Verlet y guarda un fotograma cada `save_every`.

    El fotograma 0 es el estado inicial, así que se guardan
    `steps // save_every + 1` fotogramas.

    Si se pasa `on_progress`, se llama unas PROGRESS_REPORTS veces durante la
    simulación y una última vez con 1.0 al terminar.

    Si dos cuerpos se tocan durante un paso, se fusionan (ver collisions.py).
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    # Copia: las fusiones cambian las masas y no deben tocar el array recibido.
    masses = np.array(masses, dtype=float)
    alive = masses > 0
    # absorbed_by[k] es el cuerpo vivo al que k quedó pegado (él mismo si vive).
    absorbed_by = np.arange(len(masses))
    merges: list[Merge] = []
    # Energía que se convirtió en calor en los choques.
    dissipated = 0.0

    acc = accelerations(positions, masses, g, softening)
    initial_energy = energy(positions, velocities, masses, g, softening)

    # Informar en cada paso sería lento para quien guarda el progreso;
    # con pocos pasos (menos de 100) se informa en todos.
    report_every = max(1, steps // PROGRESS_REPORTS)

    # Copia para no guardar una referencia al array que nos pasaron.
    frames = [positions.copy()]
    radii = body_radius(masses)
    pairs = alive_pairs(masses, alive, g)
    for step in range(1, steps + 1):
        # Cerca de otro cuerpo el paso fijo es demasiado grueso y se parte en
        # subpasos. Lejos de todo queda 1 subpaso y no se buscan choques.
        n_sub, may_touch = plan_step(positions, velocities, pairs, dt)
        h = dt / n_sub
        for _ in range(n_sub):
            old_positions = positions
            positions, velocities, acc = verlet_step(
                positions, velocities, acc, masses, h, g, softening
            )
            if not may_touch:
                continue

            collisions = find_collisions(old_positions, positions, radii, alive)
            if collisions:
                before = energy(positions, velocities, masses, g, softening)
                for i, j in collisions:
                    # Un cuerpo puede tocar a dos en el mismo paso: se sigue
                    # con el que lo absorbió en la fusión anterior.
                    i, j = absorbed_by[i], absorbed_by[j]
                    if i == j:
                        continue
                    survivor, absorbed = merge(positions, velocities, masses, i, j)
                    alive[absorbed] = False
                    absorbed_by[absorbed_by == absorbed] = survivor
                    # El primer fotograma guardado desde este paso en adelante.
                    frame = -(-step // save_every)
                    merges.append(Merge(frame, int(survivor), int(absorbed)))
                after = energy(positions, velocities, masses, g, softening)
                dissipated += before - after
                acc = accelerations(positions, masses, g, softening)
                radii = body_radius(masses)
                pairs = alive_pairs(masses, alive, g)

        # Los absorbidos no tienen masa pero la gravedad sí los mueve:
        # se los vuelve a pegar a su superviviente.
        if not alive.all():
            positions[~alive] = positions[absorbed_by[~alive]]
            velocities[~alive] = velocities[absorbed_by[~alive]]

        # Submuestreo: no se guardan todos los pasos, solo uno de cada save_every.
        if step % save_every == 0:
            frames.append(positions)
        if on_progress is not None and (step % report_every == 0 or step == steps):
            on_progress(step / steps)

    # La energía perdida en choques se suma de vuelta: lo que queda es el
    # error del integrador.
    final_energy = energy(positions, velocities, masses, g, softening) + dissipated
    # Con energía inicial 0 el error relativo no existe; se usa el absoluto.
    scale = abs(initial_energy) if initial_energy != 0 else 1.0
    energy_error = abs(final_energy - initial_energy) / scale

    return SimulationResult(
        frames=np.array(frames), energy_error=energy_error, merges=tuple(merges)
    )
