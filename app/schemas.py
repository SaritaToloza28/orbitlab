"""Modelos Pydantic de entrada y salida de la API.

Describen cómo debe ser el JSON que llega y el que sale. Si un JSON no
cumple, FastAPI responde 422 solo y el endpoint nunca se ejecuta.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Límites del plan: evitan que una petición abusiva tumbe el servidor.
MIN_BODIES = 2
MAX_BODIES = 10
MAX_STEPS = 200_000
MAX_FRAMES = 2_000


class ExperimentStatus(StrEnum):
    """Estados posibles de un experimento. Una lista cerrada evita typos."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BodyIn(BaseModel):
    """Un cuerpo: posición (UA), velocidad (UA/año) y masa (masas solares)."""

    # Rechaza NaN e infinito: una posición NaN contaminaría toda la simulación.
    model_config = ConfigDict(allow_inf_nan=False)

    x: float
    y: float
    vx: float
    vy: float
    mass: float = Field(gt=0)


class ExperimentCreate(BaseModel):
    """Lo que el cliente envía en POST /experiments."""

    # Quita espacios al inicio y al final, así "   " cuenta como nombre vacío.
    model_config = ConfigDict(allow_inf_nan=False, str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    dt: float = Field(gt=0)
    steps: int = Field(ge=1, le=MAX_STEPS)
    save_every: int = Field(ge=1)
    bodies: list[BodyIn] = Field(min_length=MIN_BODIES, max_length=MAX_BODIES)

    @model_validator(mode="after")
    def check_frames(self) -> "ExperimentCreate":
        """Limita los fotogramas, que dependen de steps y save_every a la vez."""
        if self.save_every > self.steps:
            raise ValueError("save_every no puede ser mayor que steps")

        # Misma cuenta que simulate(): el fotograma inicial + uno cada save_every.
        n_frames = self.steps // self.save_every + 1
        if n_frames > MAX_FRAMES:
            raise ValueError(
                f"saldrían {n_frames} fotogramas y el máximo es {MAX_FRAMES}; "
                "aumenta save_every"
            )
        return self

    @model_validator(mode="after")
    def check_duplicate_bodies(self) -> "ExperimentCreate":
        """Rechaza dos cuerpos en la misma posición: casi siempre es un error."""
        seen: dict[tuple[float, float], int] = {}
        for index, body in enumerate(self.bodies):
            position = (body.x, body.y)
            if position in seen:
                raise ValueError(
                    f"los cuerpos {seen[position]} y {index} están en la misma "
                    f"posición ({body.x}, {body.y})"
                )
            seen[position] = index
        return self


class ExperimentAccepted(BaseModel):
    """Respuesta 202 del POST: el pedido se recibió, el cálculo va en camino."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: ExperimentStatus


class ExperimentSummary(BaseModel):
    """Versión liviana para GET /experiments: sin cuerpos ni fotogramas."""

    # Permite construirlo desde un objeto de la base de datos, no solo un dict.
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    status: ExperimentStatus
    created_at: datetime


class ExperimentDetail(ExperimentSummary):
    """Todo sobre un experimento, para GET /experiments/{id}."""

    dt: float
    steps: int
    save_every: int
    bodies: list[BodyIn]
    # Fracción de pasos calculados, de 0 a 1.
    progress: float = Field(ge=0, le=1)
    # Se completa al terminar la simulación (RF8).
    energy_error: float | None = None
    # Motivo del fallo, solo si el estado es failed (RF7).
    error_message: str | None = None
