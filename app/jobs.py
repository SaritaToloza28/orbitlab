"""Simulaciones en segundo plano con un pool de procesos.

Procesos y no hilos: la simulación es cálculo puro, y el GIL de Python
impide que dos hilos calculen a la vez. Cada proceso tiene su propio GIL.

Los procesos no comparten memoria, así que al worker solo se le envía el id
del experimento y la dirección de la base de datos; con eso abre su propia
conexión, simula y guarda el resultado.

Este módulo no importa app.main: en Windows cada worker vuelve a importar
el módulo de su función, y así no carga FastAPI ni las rutas.
"""

from concurrent.futures import Future, ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine, select

from app.db import DATABASE_URL, Experiment, ExperimentFrames
from app.schemas import ExperimentStatus
from physics.integrators import simulate

# Simulaciones simultáneas como máximo; las demás esperan en la cola.
POOL_SIZE = 2

INTERRUPTED_MESSAGE = "interrumpido por un reinicio del servidor"

_executor: ProcessPoolExecutor | None = None


@lru_cache
def engine_for(database_url: str) -> Engine:
    """Un engine por dirección y por proceso, reutilizado entre experimentos."""
    return create_engine(database_url, connect_args={"check_same_thread": False})


def run_experiment(experiment_id: int, database_url: str = DATABASE_URL) -> None:
    """Simula un experimento guardado y deja el resultado en la base de datos.

    Corre dentro de un worker. Pasa por running y termina en completed, o en
    failed con el motivo si algo sale mal (RF7).
    """
    with Session(engine_for(database_url)) as session:
        experiment = session.get(Experiment, experiment_id)
        # Pudo haberse borrado mientras esperaba en la cola.
        if experiment is None:
            return

        experiment.status = ExperimentStatus.RUNNING
        experiment.progress = 0.0
        session.add(experiment)
        session.commit()

        def save_progress(fraction: float) -> None:
            # Commit para que la API, en otro proceso, vea el avance.
            experiment.progress = fraction
            session.add(experiment)
            session.commit()

        # Los cuerpos están guardados como dicts; simulate() necesita arrays.
        positions = np.array([[b["x"], b["y"]] for b in experiment.bodies])
        velocities = np.array([[b["vx"], b["vy"]] for b in experiment.bodies])
        masses = np.array([b["mass"] for b in experiment.bodies])

        try:
            result = simulate(
                positions,
                velocities,
                masses,
                experiment.dt,
                experiment.steps,
                experiment.save_every,
                on_progress=save_progress,
            )
            # NumPy no lanza error con NaN: lo comprobamos nosotros.
            if not (
                np.all(np.isfinite(result.frames)) and np.isfinite(result.energy_error)
            ):
                raise ValueError(
                    "la simulación produjo valores no finitos (NaN o infinito)"
                )
        # Cualquier error se guarda como fallo en vez de perderse en el worker
        # (RF7), por eso se captura Exception a propósito.
        except Exception as error:  # noqa: BLE001
            session.rollback()
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = str(error)
        else:
            session.add(
                ExperimentFrames(
                    experiment_id=experiment.id, frames=result.frames.tolist()
                )
            )
            experiment.status = ExperimentStatus.COMPLETED
            experiment.progress = 1.0
            experiment.energy_error = result.energy_error

        session.add(experiment)
        session.commit()


def recover_interrupted(database_url: str = DATABASE_URL) -> int:
    """Marca como fallidos los experimentos que quedaron a medias.

    Si el servidor se apaga con simulaciones en curso o en cola, quedarían
    running o pending para siempre. Se llama al arrancar; devuelve cuántos marcó.
    """
    unfinished = [ExperimentStatus.PENDING, ExperimentStatus.RUNNING]
    with Session(engine_for(database_url)) as session:
        experiments = session.exec(
            select(Experiment).where(Experiment.status.in_(unfinished))
        ).all()
        for experiment in experiments:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = INTERRUPTED_MESSAGE
            session.add(experiment)
        session.commit()
        return len(experiments)


def start_pool() -> None:
    """Crea el pool de procesos. Se llama una vez al arrancar la API."""
    global _executor
    _executor = ProcessPoolExecutor(max_workers=POOL_SIZE)


def stop_pool() -> None:
    """Apaga el pool sin esperar: los que estaban en cola se cancelan.

    Esos quedan pending en la base y recover_interrupted() los marca al
    volver a arrancar.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


def submit_experiment(
    experiment_id: int, database_url: str = DATABASE_URL
) -> Future[None]:
    """Encola un experimento y vuelve de inmediato, sin esperar el cálculo."""
    if _executor is None:
        raise RuntimeError("el pool de procesos no está iniciado")
    return _executor.submit(run_experiment, experiment_id, database_url)
