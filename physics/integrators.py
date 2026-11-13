"""Integradores que avanzan en el tiempo un sistema de N cuerpos.

Velocity Verlet es el integrador de la simulación: es simpléctico, así que
el error en la energía oscila pero no crece. Euler queda solo como
comparación en los tests, porque acumula energía y la órbita se abre.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from physics.gravity import DEFAULT_SOFTENING, G, accelerations

# Función que recibe la fracción completada (de 0 a 1). La física no sabe
# qué hace con ella: la API la usa para guardar el progreso en la base de datos.
ProgressCallback = Callable[[float], None]

# Cuántas veces como máximo se informa el progreso (cada 1 %).
PROGRESS_REPORTS = 100


class SimulationResult(NamedTuple):
    """Resultado de una simulación completa."""

    # Posiciones guardadas cada `save_every` pasos. (n_fotogramas, N, 2)
    frames: np.ndarray
    # |E_final - E_inicial| / |E_inicial|: mide qué tan bien se conservó la energía.
    energy_error: float


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
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)

    acc = accelerations(positions, masses, g, softening)
    initial_energy = energy(positions, velocities, masses, g, softening)

    # Informar en cada paso sería lento para quien guarda el progreso;
    # con pocos pasos (menos de 100) se informa en todos.
    report_every = max(1, steps // PROGRESS_REPORTS)

    # Copia para no guardar una referencia al array que nos pasaron.
    frames = [positions.copy()]
    for step in range(1, steps + 1):
        positions, velocities, acc = verlet_step(
            positions, velocities, acc, masses, dt, g, softening
        )
        # Submuestreo: no se guardan todos los pasos, solo uno de cada save_every.
        if step % save_every == 0:
            frames.append(positions)
        if on_progress is not None and (step % report_every == 0 or step == steps):
            on_progress(step / steps)

    final_energy = energy(positions, velocities, masses, g, softening)
    # Con energía inicial 0 el error relativo no existe; se usa el absoluto.
    scale = abs(initial_energy) if initial_energy != 0 else 1.0
    energy_error = abs(final_energy - initial_energy) / scale

    return SimulationResult(frames=np.array(frames), energy_error=energy_error)
