import numpy as np
import pytest
from sqlmodel import Session, select

from app import jobs
from app.db import Experiment, ExperimentFrames, create_db_and_tables
from app.jobs import (
    INTERRUPTED_MESSAGE,
    engine_for,
    recover_interrupted,
    run_experiment,
    start_pool,
    stop_pool,
    submit_experiment,
)
from app.schemas import ExperimentStatus
from physics.integrators import SimulationResult

SUN_EARTH_BODIES = [
    {"x": 0, "y": 0, "vx": 0, "vy": 0, "mass": 1.0},
    {"x": 1, "y": 0, "vx": 0, "vy": 6.283, "mass": 3e-6},
]


@pytest.fixture
def database_url(tmp_path):
    """Base de datos en un archivo temporal.

    No sirve una en memoria: el worker real corre en otro proceso y necesita
    abrir el mismo archivo por su cuenta.
    """
    url = f"sqlite:///{tmp_path / 'test.db'}"
    create_db_and_tables(engine_for(url))
    return url


def save_experiment(
    database_url: str, status: ExperimentStatus = ExperimentStatus.PENDING
) -> int:
    """Guarda un Sol-Tierra corto y devuelve su id."""
    with Session(engine_for(database_url)) as session:
        experiment = Experiment(
            name="Sol y Tierra",
            dt=0.001,
            steps=1000,
            save_every=10,
            bodies=SUN_EARTH_BODIES,
            status=status,
        )
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        return experiment.id


def load(database_url: str, experiment_id: int) -> Experiment:
    """Lee el experimento con una sesión nueva: lo que de verdad quedó guardado."""
    with Session(engine_for(database_url)) as session:
        return session.get(Experiment, experiment_id)


def test_run_experiment_completes(database_url):
    experiment_id = save_experiment(database_url)

    run_experiment(experiment_id, database_url)

    experiment = load(database_url, experiment_id)
    assert experiment.status is ExperimentStatus.COMPLETED
    assert experiment.progress == 1.0
    assert experiment.energy_error < 1e-3
    with Session(engine_for(database_url)) as session:
        frames = session.get(ExperimentFrames, experiment_id)
    # 1000 pasos guardando cada 10 + el inicial.
    assert len(frames.frames) == 101


def test_progress_is_visible_while_running(database_url, monkeypatch):
    experiment_id = save_experiment(database_url)
    seen = {}

    def fake_simulate(*args, on_progress, **kwargs):
        on_progress(0.5)
        # Mirando desde otra sesión, como lo haría la API.
        seen["experiment"] = load(database_url, experiment_id)
        return SimulationResult(frames=np.zeros((2, 2, 2)), energy_error=0.0)

    monkeypatch.setattr("app.jobs.simulate", fake_simulate)

    run_experiment(experiment_id, database_url)

    assert seen["experiment"].status is ExperimentStatus.RUNNING
    assert seen["experiment"].progress == 0.5


def test_failed_simulation_is_saved_with_reason(database_url, monkeypatch):
    experiment_id = save_experiment(database_url)

    def broken_simulate(*args, **kwargs):
        raise RuntimeError("algo salió mal")

    monkeypatch.setattr("app.jobs.simulate", broken_simulate)

    run_experiment(experiment_id, database_url)

    experiment = load(database_url, experiment_id)
    assert experiment.status is ExperimentStatus.FAILED
    assert experiment.error_message == "algo salió mal"
    assert experiment.energy_error is None


def test_nan_is_a_failure(database_url, monkeypatch):
    experiment_id = save_experiment(database_url)

    def nan_simulate(*args, **kwargs):
        return SimulationResult(frames=np.full((2, 2, 2), np.nan), energy_error=0.0)

    monkeypatch.setattr("app.jobs.simulate", nan_simulate)

    run_experiment(experiment_id, database_url)

    experiment = load(database_url, experiment_id)
    assert experiment.status is ExperimentStatus.FAILED
    assert "NaN" in experiment.error_message
    with Session(engine_for(database_url)) as session:
        assert session.exec(select(ExperimentFrames)).all() == []


def test_missing_experiment_does_nothing(database_url):
    # Borrado mientras esperaba en la cola: el worker no debe fallar.
    run_experiment(999, database_url)


def test_recover_interrupted(database_url):
    pending = save_experiment(database_url, ExperimentStatus.PENDING)
    running = save_experiment(database_url, ExperimentStatus.RUNNING)
    completed = save_experiment(database_url, ExperimentStatus.COMPLETED)

    assert recover_interrupted(database_url) == 2

    for experiment_id in (pending, running):
        experiment = load(database_url, experiment_id)
        assert experiment.status is ExperimentStatus.FAILED
        assert experiment.error_message == INTERRUPTED_MESSAGE
    # Los que ya habían terminado no se tocan.
    assert load(database_url, completed).status is ExperimentStatus.COMPLETED


def test_submit_without_pool_fails():
    with pytest.raises(RuntimeError, match="no está iniciado"):
        submit_experiment(1)


def test_stop_pool_twice_is_safe():
    stop_pool()
    stop_pool()


def test_real_pool_runs_experiment_in_another_process(database_url):
    """Flujo real: el worker es otro proceso y se comunica solo por la base."""
    experiment_id = save_experiment(database_url)
    start_pool()
    try:
        future = submit_experiment(experiment_id, database_url)
        # Solo el test espera; la API nunca lo hará.
        future.result(timeout=60)
    finally:
        stop_pool()

    assert jobs._executor is None
    assert load(database_url, experiment_id).status is ExperimentStatus.COMPLETED
