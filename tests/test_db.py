import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from app.db import (
    Experiment,
    ExperimentFrames,
    create_db_and_tables,
    delete_experiment,
    get_session,
)
from app.schemas import ExperimentCreate, ExperimentDetail, ExperimentStatus
from physics.integrators import simulate


def plan_example() -> ExperimentCreate:
    """El JSON de ejemplo del plan, ya validado."""
    return ExperimentCreate.model_validate(
        {
            "name": "Sol y Tierra",
            "dt": 0.001,
            "steps": 100,
            "save_every": 10,
            "bodies": [
                {"x": 0, "y": 0, "vx": 0, "vy": 0, "mass": 1.0},
                {"x": 1, "y": 0, "vx": 0, "vy": 6.283, "mass": 3e-6},
            ],
        }
    )


@pytest.fixture
def engine():
    """Base de datos en memoria: nace en cada test y desaparece al terminar.

    StaticPool reutiliza una sola conexión; sin eso, cada sesión vería una
    base en memoria distinta y vacía.
    """
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_db_and_tables(test_engine)
    return test_engine


def save_example(engine) -> int:
    """Guarda el ejemplo del plan y devuelve su id."""
    with Session(engine) as session:
        experiment = Experiment(**plan_example().model_dump())
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        return experiment.id


def test_save_assigns_id(engine):
    first = save_example(engine)
    second = save_example(engine)

    assert first == 1
    assert second == 2


def test_new_experiment_defaults(engine):
    experiment_id = save_example(engine)

    # Sesión nueva: lee lo que de verdad quedó guardado, no un objeto en memoria.
    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)

    assert experiment.status is ExperimentStatus.PENDING
    assert experiment.progress == 0.0
    assert experiment.energy_error is None
    assert experiment.error_message is None
    assert experiment.created_at is not None


def test_bodies_survive_json_round_trip(engine):
    experiment_id = save_example(engine)

    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)

    assert experiment.bodies == plan_example().model_dump()["bodies"]


def test_frames_survive_json_round_trip(engine):
    experiment_id = save_example(engine)
    data = plan_example()
    positions = [[b.x, b.y] for b in data.bodies]
    velocities = [[b.vx, b.vy] for b in data.bodies]
    masses = [b.mass for b in data.bodies]
    result = simulate(
        positions, velocities, masses, data.dt, data.steps, data.save_every
    )

    with Session(engine) as session:
        # tolist() convierte el array de NumPy en listas, que sí caben en JSON.
        session.add(
            ExperimentFrames(experiment_id=experiment_id, frames=result.frames.tolist())
        )
        session.commit()

    with Session(engine) as session:
        saved = session.get(ExperimentFrames, experiment_id)

    assert saved.frames == result.frames.tolist()
    assert len(saved.frames) == 11


def test_status_update_is_saved(engine):
    experiment_id = save_example(engine)

    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)
        experiment.status = ExperimentStatus.COMPLETED
        experiment.progress = 1.0
        experiment.energy_error = 1e-12
        session.add(experiment)
        session.commit()

    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)

    assert experiment.status is ExperimentStatus.COMPLETED
    assert experiment.energy_error == 1e-12


def test_get_missing_returns_none(engine):
    # Este None es el que la API convertirá en un 404.
    with Session(engine) as session:
        assert session.get(Experiment, 999) is None


def test_detail_from_database_row(engine):
    experiment_id = save_example(engine)

    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)
        detail = ExperimentDetail.model_validate(experiment)

    assert detail.id == experiment_id
    assert detail.status is ExperimentStatus.PENDING
    assert detail.bodies[1].vy == 6.283


def test_delete_removes_experiment_and_frames(engine):
    experiment_id = save_example(engine)
    with Session(engine) as session:
        session.add(ExperimentFrames(experiment_id=experiment_id, frames=[[[0, 0]]]))
        session.commit()

    with Session(engine) as session:
        delete_experiment(session, session.get(Experiment, experiment_id))

    with Session(engine) as session:
        assert session.get(Experiment, experiment_id) is None
        assert session.get(ExperimentFrames, experiment_id) is None
        assert session.exec(select(ExperimentFrames)).all() == []


def test_delete_experiment_without_frames(engine):
    experiment_id = save_example(engine)

    with Session(engine) as session:
        delete_experiment(session, session.get(Experiment, experiment_id))

    with Session(engine) as session:
        assert session.get(Experiment, experiment_id) is None


def test_get_session_yields_a_session():
    # Solo abre la sesión; no consulta, así que no crea el archivo orbitlab.db.
    sessions = get_session()
    session = next(sessions)

    assert isinstance(session, Session)
    sessions.close()
