"""Rutas de la API de Orbitlab.

Conecta las piezas: valida con schemas.py, simula con physics/ y guarda
con db.py. En la fase 2 el POST simula antes de responder (síncrono);
en la fase 3 el cálculo pasa a segundo plano.

Para levantarla, desde la carpeta del proyecto:
    uvicorn app.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from sqlmodel import Session, select

from app.db import Experiment, ExperimentFrames, create_db_and_tables, get_session
from app.schemas import (
    ExperimentCreate,
    ExperimentDetail,
    ExperimentStatus,
    ExperimentSummary,
)
from physics.integrators import simulate


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Corre una vez al arrancar el servidor: crea las tablas si no existen."""
    create_db_and_tables()
    yield


app = FastAPI(
    title="Orbitlab",
    description="Simulación gravitatoria de N cuerpos en 2D.",
    lifespan=lifespan,
)

# Atajo: cada endpoint que lo pida recibe su propia sesión de base de datos.
SessionDep = Annotated[Session, Depends(get_session)]


def run_simulation(session: Session, experiment: Experiment) -> None:
    """Simula el experimento y guarda el resultado, o el motivo si falla (RF7)."""
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
        )
        # NumPy no lanza error con NaN: lo comprobamos nosotros.
        if not (
            np.all(np.isfinite(result.frames)) and np.isfinite(result.energy_error)
        ):
            raise ValueError(
                "la simulación produjo valores no finitos (NaN o infinito)"
            )
    # Cualquier error se guarda como fallo en vez de tumbar la petición (RF7),
    # por eso se captura Exception a propósito.
    except Exception as error:  # noqa: BLE001
        experiment.status = ExperimentStatus.FAILED
        experiment.error_message = str(error)
    else:
        session.add(
            ExperimentFrames(experiment_id=experiment.id, frames=result.frames.tolist())
        )
        experiment.status = ExperimentStatus.COMPLETED
        experiment.progress = 1.0
        experiment.energy_error = result.energy_error

    session.add(experiment)
    session.commit()


@app.post(
    "/experiments",
    response_model=ExperimentDetail,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(data: ExperimentCreate, session: SessionDep) -> Experiment:
    """Crea un experimento y lo simula (RF1). Datos inválidos dan 422 (RF6)."""
    experiment = Experiment(**data.model_dump(), status=ExperimentStatus.RUNNING)
    session.add(experiment)
    session.commit()
    # refresh trae el id que asignó la base de datos.
    session.refresh(experiment)

    run_simulation(session, experiment)
    session.refresh(experiment)
    return experiment


@app.get("/experiments", response_model=list[ExperimentSummary])
def list_experiments(session: SessionDep) -> list[Experiment]:
    """Lista resumida, del más nuevo al más viejo (RF5). No carga fotogramas."""
    return session.exec(select(Experiment).order_by(Experiment.id.desc())).all()


@app.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: int, session: SessionDep) -> Experiment:
    """Parámetros, estado, progreso y error de energía de un experimento (RF3, RF8)."""
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no existe el experimento {experiment_id}",
        )
    return experiment
