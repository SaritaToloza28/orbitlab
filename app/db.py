"""Tablas de la base de datos (SQLite) y manejo de la sesión.

La base de datos es el lugar común entre la API y los workers: corren en
procesos distintos, que no comparten memoria.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import JSON, Column
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from app.schemas import ExperimentStatus

# El archivo vive en la raíz del proyecto, sin importar desde dónde se ejecute.
DATABASE_PATH = Path(__file__).resolve().parent.parent / "orbitlab.db"

# check_same_thread=False: FastAPI puede atender una petición en otro hilo
# distinto al que abrió la conexión, y SQLite lo prohíbe por defecto.
engine = create_engine(
    f"sqlite:///{DATABASE_PATH}", connect_args={"check_same_thread": False}
)


class Experiment(SQLModel, table=True):
    """Un experimento: parámetros, estado y resultados livianos."""

    # None hasta guardarlo; la base de datos asigna 1, 2, 3...
    id: int | None = Field(default=None, primary_key=True)
    name: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    dt: float
    steps: int
    save_every: int
    # Lista de cuerpos guardada entera como JSON en una sola columna.
    # Ojo: modificar la lista en el lugar no se detecta; hay que reasignarla.
    bodies: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    progress: float = 0.0
    energy_error: float | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExperimentFrames(SQLModel, table=True):
    """Fotogramas de un experimento, aparte porque son pesados.

    Así listar experimentos no carga los fotogramas de todos.
    """

    # Clave foránea y primaria a la vez: un experimento tiene a lo sumo
    # una fila de fotogramas (relación 1 a 1).
    experiment_id: int = Field(foreign_key="experiment.id", primary_key=True)
    # (n_fotogramas, N, 2) convertido a listas anidadas.
    frames: list = Field(sa_column=Column(JSON, nullable=False))


def create_db_and_tables(db_engine: Engine = engine) -> None:
    """Crea las tablas si no existen. No modifica tablas ya creadas."""
    SQLModel.metadata.create_all(db_engine)


def get_session() -> Iterator[Session]:
    """Da una sesión por petición y la cierra al terminar, aunque haya error.

    Los endpoints la piden con Depends(get_session); los tests la reemplazan
    por una que usa una base de datos temporal.
    """
    with Session(engine) as session:
        yield session


def delete_experiment(session: Session, experiment: Experiment) -> None:
    """Borra un experimento junto con sus fotogramas, si los tiene."""
    frames = session.get(ExperimentFrames, experiment.id)
    if frames is not None:
        session.delete(frames)
    session.delete(experiment)
    session.commit()
