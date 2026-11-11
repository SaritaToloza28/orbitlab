import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.db import Experiment, ExperimentFrames, create_db_and_tables, get_session
from app.jobs import (
    engine_for,
    run_experiment,
    start_pool,
    stop_pool,
    submit_experiment,
)
from app.main import app, get_submitter
from app.schemas import ExperimentStatus
from physics.integrators import SimulationResult


def plan_example(name: str = "Sol y Tierra") -> dict:
    """El JSON de ejemplo del plan."""
    return {
        "name": name,
        "dt": 0.001,
        "steps": 20000,
        "save_every": 20,
        "bodies": [
            {"x": 0, "y": 0, "vx": 0, "vy": 0, "mass": 1.0},
            {"x": 1, "y": 0, "vx": 0, "vy": 6.283, "mass": 3e-6},
        ],
    }


def light_example(name: str) -> dict:
    """El mismo sistema con pocos pasos, para tests que no miran la física."""
    return {**plan_example(name), "steps": 500, "save_every": 10}


@pytest.fixture
def database_url(tmp_path):
    """Base de datos en un archivo temporal.

    No sirve una en memoria: el worker abre la base por su cuenta, y en el
    test con el pool real lo hace desde otro proceso.
    """
    url = f"sqlite:///{tmp_path / 'test.db'}"
    create_db_and_tables(engine_for(url))
    return url


@pytest.fixture
def engine(database_url):
    return engine_for(database_url)


def make_client(engine, submitter) -> TestClient:
    """Cliente HTTP que usa la base temporal y la forma de encolar indicada.

    No se usa `with TestClient(app)`: así no corre el lifespan, que tocaría
    el archivo real y arrancaría el pool.
    """

    def get_test_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_submitter] = lambda: submitter
    return TestClient(app)


@pytest.fixture
def client(engine, database_url):
    """Executor "falso": simula al instante, en este mismo proceso.

    Así los tests ven el resultado final sin esperar a otro proceso.
    """
    yield make_client(
        engine, lambda experiment_id: run_experiment(experiment_id, database_url)
    )
    app.dependency_overrides.clear()


def test_startup_and_shutdown(monkeypatch):
    calls = []
    # Reemplaza cada paso del arranque para no tocar el archivo real ni crear procesos.
    monkeypatch.setattr("app.main.create_db_and_tables", lambda: calls.append("tablas"))
    monkeypatch.setattr(
        "app.main.recover_interrupted", lambda: calls.append("recuperar")
    )
    monkeypatch.setattr("app.main.start_pool", lambda: calls.append("iniciar pool"))
    monkeypatch.setattr("app.main.stop_pool", lambda: calls.append("detener pool"))

    # Con `with`, TestClient arranca y apaga la app como el servidor real.
    with TestClient(app):
        assert calls == ["tablas", "recuperar", "iniciar pool"]

    assert calls[-1] == "detener pool"


def test_post_returns_202_without_waiting(engine):
    queued = []
    # Un executor que solo anota: el cálculo nunca ocurre durante la petición.
    client = make_client(engine, queued.append)

    response = client.post("/experiments", json=plan_example())

    assert response.status_code == 202
    assert response.json() == {"id": 1, "status": "pending"}
    assert queued == [1]
    detail = client.get("/experiments/1").json()
    assert detail["status"] == "pending"
    assert detail["progress"] == 0.0
    app.dependency_overrides.clear()


def test_create_experiment_completes(client, engine):
    response = client.post("/experiments", json=plan_example())

    assert response.status_code == 202
    detail = client.get(f"/experiments/{response.json()['id']}").json()
    assert detail["status"] == "completed"
    assert detail["progress"] == 1.0
    # Requisito del plan: error de energía < 0,1 % en Sol-Tierra.
    assert detail["energy_error"] < 1e-3
    assert detail["error_message"] is None

    # Los fotogramas quedaron guardados: 20000 / 20 + el inicial.
    with Session(engine) as session:
        frames = session.get(ExperimentFrames, detail["id"])
    assert len(frames.frames) == 1001


def test_invalid_experiment_returns_422_and_saves_nothing(client):
    data = plan_example()
    data["bodies"][1]["mass"] = -1

    response = client.post("/experiments", json=data)

    assert response.status_code == 422
    # El error dice exactamente dónde está el problema.
    assert response.json()["detail"][0]["loc"] == ["body", "bodies", 1, "mass"]
    assert client.get("/experiments").json() == []


def test_list_experiments_newest_first(client):
    client.post("/experiments", json=light_example("primero"))
    client.post("/experiments", json=light_example("segundo"))

    response = client.get("/experiments")

    assert response.status_code == 200
    names = [experiment["name"] for experiment in response.json()]
    assert names == ["segundo", "primero"]
    # La lista es liviana: sin cuerpos ni parámetros.
    assert set(response.json()[0]) == {"id", "name", "status", "created_at"}


def test_get_experiment(client):
    created = client.post("/experiments", json=light_example("Sol y Tierra")).json()

    response = client.get(f"/experiments/{created['id']}")

    assert response.status_code == 200
    assert response.json()["name"] == "Sol y Tierra"
    assert response.json()["steps"] == 500
    assert len(response.json()["bodies"]) == 2


def test_get_missing_experiment_returns_404(client):
    response = client.get("/experiments/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "no existe el experimento 999"


def test_failed_simulation_is_saved_with_reason(client, engine, monkeypatch):
    def broken_simulate(*args, **kwargs):
        raise RuntimeError("algo salió mal")

    # Reemplaza simulate solo durante este test; el worker vive en jobs.py.
    monkeypatch.setattr("app.jobs.simulate", broken_simulate)

    response = client.post("/experiments", json=plan_example())

    # La API responde igual y el experimento queda marcado como fallido.
    assert response.status_code == 202
    detail = client.get(f"/experiments/{response.json()['id']}").json()
    assert detail["status"] == "failed"
    assert detail["error_message"] == "algo salió mal"
    assert detail["energy_error"] is None
    with Session(engine) as session:
        assert session.exec(select(ExperimentFrames)).all() == []


def test_nan_in_simulation_is_a_failure(client, monkeypatch):
    def nan_simulate(*args, **kwargs):
        return SimulationResult(frames=np.full((2, 2, 2), np.nan), energy_error=0.0)

    monkeypatch.setattr("app.jobs.simulate", nan_simulate)

    response = client.post("/experiments", json=plan_example())

    detail = client.get(f"/experiments/{response.json()['id']}").json()
    assert detail["status"] == "failed"
    assert "NaN" in detail["error_message"]


def test_full_flow_with_real_pool(engine, database_url):
    """De punta a punta: POST, polling del estado y resultado, con un worker real."""
    start_pool()
    client = make_client(
        engine, lambda experiment_id: submit_experiment(experiment_id, database_url)
    )
    try:
        response = client.post("/experiments", json=plan_example())
        assert response.status_code == 202
        experiment_id = response.json()["id"]

        # Polling, como hará la página HTML: preguntar hasta que termine.
        deadline = time.monotonic() + 60
        while True:
            detail = client.get(f"/experiments/{experiment_id}").json()
            if (
                detail["status"] in ("completed", "failed")
                or time.monotonic() > deadline
            ):
                break
            time.sleep(0.1)
    finally:
        stop_pool()
        app.dependency_overrides.clear()

    assert detail["status"] == "completed"
    assert detail["energy_error"] < 1e-3


def test_real_submitter_is_the_process_pool():
    # Sin reemplazos, la API encola en el pool de procesos de jobs.py.
    assert get_submitter() is submit_experiment


def test_frames_of_completed_experiment(client):
    data = light_example("Sol y Tierra")
    experiment_id = client.post("/experiments", json=data).json()["id"]

    response = client.get(f"/experiments/{experiment_id}/frames")

    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"] == experiment_id
    # 0,001 años por paso, guardando cada 10 pasos.
    assert body["frame_interval"] == pytest.approx(0.01)
    # 500 / 10 + el inicial = 51 fotogramas, de 2 cuerpos con x, y.
    assert len(body["frames"]) == 51
    assert all(len(frame) == 2 for frame in body["frames"])
    # El primer fotograma es la posición inicial.
    assert body["frames"][0] == [[0, 0], [1, 0]]


def test_frames_of_pending_experiment_returns_409(engine):
    # Un executor que solo anota: el experimento se queda pending.
    client = make_client(engine, lambda experiment_id: None)
    experiment_id = client.post("/experiments", json=plan_example()).json()["id"]

    response = client.get(f"/experiments/{experiment_id}/frames")

    assert response.status_code == 409
    assert "todavía no terminó" in response.json()["detail"]
    assert "pending" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_frames_of_running_experiment_returns_409_with_progress(engine):
    client = make_client(engine, lambda experiment_id: None)
    experiment_id = client.post("/experiments", json=plan_example()).json()["id"]
    # Simula a un worker a mitad de camino.
    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)
        experiment.status = ExperimentStatus.RUNNING
        experiment.progress = 0.35
        session.add(experiment)
        session.commit()

    response = client.get(f"/experiments/{experiment_id}/frames")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        f"el experimento {experiment_id} todavía no terminó "
        "(estado: running, progreso: 35%)"
    )
    app.dependency_overrides.clear()


def test_frames_of_failed_experiment_returns_409_with_reason(client, monkeypatch):
    def broken_simulate(*args, **kwargs):
        raise RuntimeError("algo salió mal")

    monkeypatch.setattr("app.jobs.simulate", broken_simulate)
    experiment_id = client.post("/experiments", json=plan_example()).json()["id"]

    response = client.get(f"/experiments/{experiment_id}/frames")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        f"el experimento {experiment_id} falló: algo salió mal"
    )


def test_frames_of_missing_experiment_returns_404(client):
    response = client.get("/experiments/999/frames")

    assert response.status_code == 404


def set_running(engine, experiment_id: int) -> None:
    """Deja el experimento como si un worker lo estuviera simulando."""
    with Session(engine) as session:
        experiment = session.get(Experiment, experiment_id)
        experiment.status = ExperimentStatus.RUNNING
        session.add(experiment)
        session.commit()


def test_delete_completed_experiment(client, engine):
    experiment_id = client.post("/experiments", json=light_example("x")).json()["id"]

    response = client.delete(f"/experiments/{experiment_id}")

    # 204: hecho, sin nada que devolver.
    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/experiments/{experiment_id}").status_code == 404
    assert client.get(f"/experiments/{experiment_id}/frames").status_code == 404
    assert client.get("/experiments").json() == []
    # Los fotogramas también se borraron, no quedaron huérfanos.
    with Session(engine) as session:
        assert session.exec(select(ExperimentFrames)).all() == []


def test_delete_failed_experiment(client, monkeypatch):
    def broken_simulate(*args, **kwargs):
        raise RuntimeError("algo salió mal")

    monkeypatch.setattr("app.jobs.simulate", broken_simulate)
    experiment_id = client.post("/experiments", json=plan_example()).json()["id"]

    assert client.delete(f"/experiments/{experiment_id}").status_code == 204


def test_delete_pending_experiment(engine, database_url):
    queued = []
    client = make_client(engine, queued.append)
    experiment_id = client.post("/experiments", json=plan_example()).json()["id"]

    assert client.delete(f"/experiments/{experiment_id}").status_code == 204

    # Cuando el worker por fin lo toma de la cola, no lo encuentra y no falla.
    run_experiment(queued[0], database_url)
    assert client.get(f"/experiments/{experiment_id}").status_code == 404
    app.dependency_overrides.clear()


def test_delete_running_experiment_returns_409(engine):
    client = make_client(engine, lambda experiment_id: None)
    experiment_id = client.post("/experiments", json=plan_example()).json()["id"]
    set_running(engine, experiment_id)

    response = client.delete(f"/experiments/{experiment_id}")

    assert response.status_code == 409
    assert "se está simulando" in response.json()["detail"]
    # Sigue ahí.
    assert client.get(f"/experiments/{experiment_id}").status_code == 200
    app.dependency_overrides.clear()


def test_delete_missing_experiment_returns_404(client):
    assert client.delete("/experiments/999").status_code == 404


@pytest.mark.parametrize(
    "name, expected_name",
    [
        ("sol-tierra", "Sol y Tierra"),
        ("binario", "Sistema binario"),
        ("figura-8", "Figura 8"),
    ],
)
def test_create_from_scenario(client, name, expected_name):
    response = client.post(f"/experiments/from-scenario/{name}")

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    detail = client.get(f"/experiments/{response.json()['id']}").json()
    assert detail["name"] == expected_name
    assert detail["status"] == "completed"
    assert detail["energy_error"] < 1e-3
    frames = client.get(f"/experiments/{detail['id']}/frames").json()["frames"]
    assert len(frames) == 1001


def test_unknown_scenario_returns_404_with_options(client):
    response = client.post("/experiments/from-scenario/saturno")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "no existe el escenario 'saturno'; "
        "los disponibles son: sol-tierra, binario, figura-8"
    )
    assert client.get("/experiments").json() == []
