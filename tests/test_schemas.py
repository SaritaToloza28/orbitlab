from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas import (
    MAX_BODIES,
    MAX_FRAMES,
    MAX_STEPS,
    ExperimentCreate,
    ExperimentDetail,
    ExperimentStatus,
)


def valid_data() -> dict:
    """El JSON de ejemplo del plan: Sol y Tierra."""
    return {
        "name": "Sol y Tierra",
        "dt": 0.001,
        "steps": 20000,
        "save_every": 20,
        "bodies": [
            {"x": 0, "y": 0, "vx": 0, "vy": 0, "mass": 1.0},
            {"x": 1, "y": 0, "vx": 0, "vy": 6.283, "mass": 3e-6},
        ],
    }


def body_at(x: float) -> dict:
    """Un cuerpo válido en (x, 0), para armar listas de cuerpos distintos."""
    return {"x": x, "y": 0, "vx": 0, "vy": 0, "mass": 1.0}


def test_plan_example_is_valid():
    experiment = ExperimentCreate.model_validate(valid_data())

    assert experiment.name == "Sol y Tierra"
    assert len(experiment.bodies) == 2
    assert experiment.bodies[1].vy == 6.283


def test_converts_numeric_strings():
    data = valid_data()
    data["dt"] = "0.001"

    assert ExperimentCreate.model_validate(data).dt == 0.001


# Cada caso toma un JSON válido y le rompe una sola cosa.
@pytest.mark.parametrize(
    "field, value",
    [
        ("name", ""),
        ("name", "   "),
        ("name", "x" * 101),
        ("dt", 0),
        ("dt", -0.001),
        ("steps", 0),
        ("steps", MAX_STEPS + 1),
        ("steps", "muchos"),
        ("save_every", 0),
        ("bodies", [body_at(0)]),
        ("bodies", [body_at(x) for x in range(MAX_BODIES + 1)]),
    ],
)
def test_rejects_invalid_field(field, value):
    data = valid_data()
    data[field] = value

    with pytest.raises(ValidationError):
        ExperimentCreate.model_validate(data)


@pytest.mark.parametrize("mass", [0, -1])
def test_rejects_non_positive_mass(mass):
    data = valid_data()
    data["bodies"][1]["mass"] = mass

    with pytest.raises(ValidationError) as error:
        ExperimentCreate.model_validate(data)

    # El error dice exactamente dónde está: cuerpo 1, campo mass.
    assert error.value.errors()[0]["loc"] == ("bodies", 1, "mass")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nan_and_infinity(value):
    data = valid_data()
    data["bodies"][1]["x"] = value

    with pytest.raises(ValidationError):
        ExperimentCreate.model_validate(data)


def test_rejects_missing_body_field():
    data = valid_data()
    del data["bodies"][0]["vx"]

    with pytest.raises(ValidationError):
        ExperimentCreate.model_validate(data)


def test_rejects_duplicate_bodies():
    data = valid_data()
    data["bodies"].append({"x": 1, "y": 0, "vx": 3, "vy": 0, "mass": 2.0})

    with pytest.raises(ValidationError, match="los cuerpos 1 y 2"):
        ExperimentCreate.model_validate(data)


def test_rejects_bodies_that_start_touching():
    data = valid_data()
    # La estrella mide 0,05 UA y el planeta 0,012: a 0,06 UA ya se tocan.
    data["bodies"][1]["x"] = 0.06

    with pytest.raises(ValidationError, match="los cuerpos 0 y 1 se solapan"):
        ExperimentCreate.model_validate(data)


def test_accepts_bodies_just_apart():
    data = valid_data()
    data["bodies"][1]["x"] = 0.063

    ExperimentCreate.model_validate(data)


def test_rejects_too_many_frames():
    data = valid_data()
    # 2000 pasos guardando todos: 2000 + el inicial = 2001 fotogramas.
    data["steps"] = MAX_FRAMES
    data["save_every"] = 1

    with pytest.raises(ValidationError, match="2001 fotogramas"):
        ExperimentCreate.model_validate(data)


def test_rejects_save_every_greater_than_steps():
    data = valid_data()
    data["steps"] = 10
    data["save_every"] = 11

    with pytest.raises(ValidationError, match="save_every"):
        ExperimentCreate.model_validate(data)


# Los casos justo en el límite deben pasar: ahí se esconden los errores de < contra <=.
@pytest.mark.parametrize(
    "changes",
    [
        {"bodies": [body_at(0), body_at(1)]},
        {"bodies": [body_at(x) for x in range(MAX_BODIES)]},
        {"steps": MAX_FRAMES - 1, "save_every": 1},
        {"steps": MAX_STEPS, "save_every": 101},
        {"steps": 10, "save_every": 10},
    ],
)
def test_accepts_limits(changes):
    data = valid_data()
    data.update(changes)

    ExperimentCreate.model_validate(data)


def test_detail_from_database_object():
    # Simula una fila de la base de datos: un objeto con atributos, no un dict.
    row = SimpleNamespace(
        id=1,
        name="Sol y Tierra",
        status="completed",
        created_at=datetime(2026, 11, 7, 12, 0),
        dt=0.001,
        steps=20000,
        save_every=20,
        bodies=valid_data()["bodies"],
        progress=1.0,
        energy_error=1e-12,
        error_message=None,
    )

    detail = ExperimentDetail.model_validate(row)

    assert detail.status is ExperimentStatus.COMPLETED
    assert detail.bodies[0].mass == 1.0


def test_detail_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ExperimentDetail.model_validate(
            {
                "id": 1,
                "name": "x",
                "status": "complted",
                "created_at": "2026-11-07T12:00:00",
                "dt": 0.001,
                "steps": 10,
                "save_every": 1,
                "bodies": [],
                "progress": 0,
            }
        )
