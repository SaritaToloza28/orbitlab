import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from app.db import ExperimentFrames, create_db_and_tables, get_session
from app.main import app
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
def engine():
    """Base de datos en memoria para cada test, igual que en test_db.py."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_db_and_tables(test_engine)
    return test_engine


@pytest.fixture
def client(engine):
    """Cliente HTTP de prueba que usa la base en memoria en vez de orbitlab.db.

    No se usa `with TestClient(app)`: así no corre el lifespan, que crearía
    las tablas en el archivo real.
    """

    def get_test_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_startup_creates_tables(monkeypatch):
    calls = []
    # Reemplaza la creación de tablas para no tocar el archivo real.
    monkeypatch.setattr("app.main.create_db_and_tables", lambda: calls.append(True))

    # Con `with`, TestClient arranca la app como el servidor real: corre el lifespan.
    with TestClient(app):
        pass

    assert calls == [True]


def test_create_experiment(client, engine):
    response = client.post("/experiments", json=plan_example())

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 1
    assert body["status"] == "completed"
    assert body["progress"] == 1.0
    # Requisito del plan: error de energía < 0,1 % en Sol-Tierra.
    assert body["energy_error"] < 1e-3
    assert body["error_message"] is None

    # Los fotogramas quedaron guardados: 20000 / 20 + el inicial.
    with Session(engine) as session:
        frames = session.get(ExperimentFrames, body["id"])
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
    assert response.json() == created


def test_get_missing_experiment_returns_404(client):
    response = client.get("/experiments/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "no existe el experimento 999"


def test_failed_simulation_is_saved_with_reason(client, engine, monkeypatch):
    def broken_simulate(*args, **kwargs):
        raise RuntimeError("algo salió mal")

    # Reemplaza simulate solo durante este test.
    monkeypatch.setattr("app.main.simulate", broken_simulate)

    response = client.post("/experiments", json=plan_example())

    # La API no se cae: responde y deja el experimento marcado como fallido.
    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert response.json()["error_message"] == "algo salió mal"
    assert response.json()["energy_error"] is None
    with Session(engine) as session:
        assert session.exec(select(ExperimentFrames)).all() == []


def test_nan_in_simulation_is_a_failure(client, monkeypatch):
    def nan_simulate(*args, **kwargs):
        return SimulationResult(frames=np.full((2, 2, 2), np.nan), energy_error=0.0)

    monkeypatch.setattr("app.main.simulate", nan_simulate)

    response = client.post("/experiments", json=plan_example())

    assert response.json()["status"] == "failed"
    assert "NaN" in response.json()["error_message"]
