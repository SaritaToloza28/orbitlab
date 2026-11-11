"""Rutas de la API de Orbitlab.

Conecta las piezas: valida con schemas.py, guarda con db.py y encola la
simulación en el pool de procesos de jobs.py. La petición HTTP nunca espera
al cálculo: el POST responde 202 y el cliente consulta el estado (polling).

Para levantarla, desde la carpeta del proyecto:
    uvicorn app.main:app --reload
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from sqlmodel import Session, select

from app.db import Experiment, create_db_and_tables, get_session
from app.jobs import recover_interrupted, start_pool, stop_pool, submit_experiment
from app.schemas import (
    ExperimentAccepted,
    ExperimentCreate,
    ExperimentDetail,
    ExperimentSummary,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Al arrancar prepara la base y el pool; al apagar, detiene el pool."""
    create_db_and_tables()
    # Lo que quedó a medias en un apagado anterior ya no se va a terminar.
    recover_interrupted()
    start_pool()
    yield
    stop_pool()


app = FastAPI(
    title="Orbitlab",
    description="Simulación gravitatoria de N cuerpos en 2D.",
    lifespan=lifespan,
)

# Función que encola un experimento por su id y vuelve sin esperar.
Submitter = Callable[[int], object]


def get_submitter() -> Submitter:
    """Dependencia para encolar: los tests la cambian por una que corre al instante."""
    return submit_experiment


# Atajos: cada endpoint que los pida recibe su sesión y la función para encolar.
SessionDep = Annotated[Session, Depends(get_session)]
SubmitterDep = Annotated[Submitter, Depends(get_submitter)]


@app.post(
    "/experiments",
    response_model=ExperimentAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_experiment(
    data: ExperimentCreate, session: SessionDep, submit: SubmitterDep
) -> Experiment:
    """Guarda el experimento como pending y lo encola (RF1, RF2).

    Responde 202 de inmediato, sin esperar el cálculo. Datos inválidos dan 422 (RF6).
    """
    experiment = Experiment(**data.model_dump())
    session.add(experiment)
    session.commit()
    # refresh trae el id que asignó la base de datos.
    session.refresh(experiment)

    submit(experiment.id)
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
