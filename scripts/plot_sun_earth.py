"""Dibuja la órbita Sol-Tierra con Velocity Verlet y con Euler, lado a lado.

Con el mismo dt, Verlet repite el mismo círculo cada año y Euler se abre
en espiral porque acumula energía.

Uso, desde la carpeta del proyecto:
    python -m scripts.plot_sun_earth
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from physics.integrators import energy, euler_step, simulate

DT = 0.001
YEARS = 3
STEPS = int(YEARS / DT)
SAVE_EVERY = 10

# Sol en el origen y Tierra a 1 UA con velocidad de órbita circular (2π UA/año).
POSITIONS = np.array([[0.0, 0.0], [1.0, 0.0]])
VELOCITIES = np.array([[0.0, 0.0], [0.0, 2 * np.pi]])
MASSES = np.array([1.0, 3e-6])

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "orbita_sol_tierra.png"


def simulate_euler() -> tuple[np.ndarray, float]:
    """Igual que `simulate`, pero con Euler. Solo existe para comparar."""
    positions, velocities = POSITIONS, VELOCITIES
    frames = [positions]
    for step in range(1, STEPS + 1):
        positions, velocities = euler_step(positions, velocities, MASSES, DT)
        if step % SAVE_EVERY == 0:
            frames.append(positions)

    initial_energy = energy(POSITIONS, VELOCITIES, MASSES)
    final_energy = energy(positions, velocities, MASSES)
    energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
    return np.array(frames), energy_error


def plot_orbit(
    ax: plt.Axes, frames: np.ndarray, title: str, energy_error: float
) -> None:
    """Dibuja la trayectoria de la Tierra (cuerpo 1) alrededor del Sol (cuerpo 0)."""
    # frames[:, 1, 0] son todas las x de la Tierra; frames[:, 1, 1], todas sus y.
    ax.plot(frames[:, 1, 0], frames[:, 1, 1], lw=1, color="tab:blue", label="Tierra")
    ax.plot(*frames[0, 1], "o", color="tab:blue", ms=5, label="Inicio")
    ax.plot(*frames[0, 0], "o", color="orange", ms=14, label="Sol")

    ax.set_title(f"{title}\nerror de energía: {energy_error:.2e}")
    ax.set_xlabel("x (UA)")
    ax.set_ylabel("y (UA)")
    # Misma escala en x e y: si no, un círculo se ve como un óvalo.
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    verlet = simulate(POSITIONS, VELOCITIES, MASSES, DT, STEPS, SAVE_EVERY)
    euler_frames, euler_error = simulate_euler()

    # Ejes compartidos para que las dos órbitas se comparen a la misma escala.
    fig, (ax_verlet, ax_euler) = plt.subplots(
        1, 2, figsize=(11, 5.5), sharex=True, sharey=True, layout="constrained"
    )
    plot_orbit(ax_verlet, verlet.frames, "Velocity Verlet", verlet.energy_error)
    plot_orbit(ax_euler, euler_frames, "Euler", euler_error)
    fig.suptitle(f"Órbita Sol-Tierra: {YEARS} años con dt = {DT} años")

    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=120)
    print(f"Imagen guardada en {OUTPUT}")
    plt.show()


if __name__ == "__main__":
    main()
