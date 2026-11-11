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

from app.db import (
    Experiment,
    ExperimentFrames,
    create_db_and_tables,
    delete_experiment,
    get_session,
)
from app.jobs import recover_interrupted, start_pool, stop_pool, submit_experiment
from app.schemas import (
    ExperimentAccepted,
    ExperimentCreate,
    ExperimentDetail,
    ExperimentStatus,
    ExperimentSummary,
    FramesOut,
)
from physics.scenarios import SCENARIOS


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


def save_and_submit(
    data: ExperimentCreate, session: Session, submit: Submitter
) -> Experiment:
    """Guarda el experimento como pending y lo encola, sin esperar el cálculo."""
    experiment = Experiment(**data.model_dump())
    session.add(experiment)
    session.commit()
    # refresh trae el id que asignó la base de datos.
    session.refresh(experiment)

    submit(experiment.id)
    return experiment


@app.post(
    "/experiments",
    response_model=ExperimentAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_experiment(
    data: ExperimentCreate, session: SessionDep, submit: SubmitterDep
) -> Experiment:
    """Crea un experimento y lo encola (RF1, RF2).

    Responde 202 de inmediato, sin esperar el cálculo. Datos inválidos dan 422 (RF6).
    """
    return save_and_submit(data, session, submit)


@app.post(
    "/experiments/from-scenario/{name}",
    response_model=ExperimentAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_from_scenario(
    name: str, session: SessionDep, submit: SubmitterDep
) -> Experiment:
    """Crea un experimento a partir de un escenario precargado (RF9)."""
    scenario = SCENARIOS.get(name)
    if scenario is None:
        available = ", ".join(SCENARIOS)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no existe el escenario '{name}'; los disponibles son: {available}",
        )
    # Pasa por las mismas validaciones que un experimento enviado por el cliente.
    data = ExperimentCreate.model_validate(scenario)
    return save_and_submit(data, session, submit)


@app.get("/experiments", response_model=list[ExperimentSummary])
def list_experiments(session: SessionDep) -> list[Experiment]:
    """Lista resumida, del más nuevo al más viejo (RF5). No carga fotogramas."""
    return session.exec(select(Experiment).order_by(Experiment.id.desc())).all()


def get_experiment_or_404(session: Session, experiment_id: int) -> Experiment:
    """Busca un experimento; si no existe, responde 404."""
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no existe el experimento {experiment_id}",
        )
    return experiment


@app.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: int, session: SessionDep) -> Experiment:
    """Parámetros, estado, progreso y error de energía de un experimento (RF3, RF8)."""
    return get_experiment_or_404(session, experiment_id)


@app.get("/experiments/{experiment_id}/frames", response_model=FramesOut)
def get_frames(experiment_id: int, session: SessionDep) -> FramesOut:
    """Fotogramas de un experimento terminado (RF4).

    404 si no existe. 409 si existe pero no tiene fotogramas: todavía no
    terminó, o falló. El experimento está, pero su estado no lo permite.
    """
    experiment = get_experiment_or_404(session, experiment_id)

    if experiment.status is ExperimentStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"el experimento {experiment_id} falló: {experiment.error_message}",
        )
    if experiment.status is not ExperimentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"el experimento {experiment_id} todavía no terminó "
                f"(estado: {experiment.status}, progreso: {experiment.progress:.0%})"
            ),
        )

    # Solo aquí se leen los fotogramas: la tabla aparte evita cargarlos antes.
    saved = session.get(ExperimentFrames, experiment_id)
    return FramesOut(
        experiment_id=experiment_id,
        frame_interval=experiment.dt * experiment.save_every,
        frames=saved.frames,
    )


@app.delete("/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete(experiment_id: int, session: SessionDep) -> None:
    """Borra un experimento y sus fotogramas (RF10).

    No se puede borrar mientras corre: el worker, en otro proceso, intentaría
    guardar en una fila que ya no existe. Uno pending sí se puede: cuando el
    worker lo tome, no lo encontrará y no hará nada.
    """
    experiment = get_experiment_or_404(session, experiment_id)
    if experiment.status is ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"el experimento {experiment_id} se está simulando; "
                "espera a que termine para borrarlo"
            ),
        )
    delete_experiment(session, experiment)
