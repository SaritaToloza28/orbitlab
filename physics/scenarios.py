"""Escenarios precargados: sistemas conocidos que sirven de demo y de prueba.

Si el integrador fuera incorrecto, estas órbitas se desarmarían; por eso
también se usan en los tests.

Cada escenario tiene la misma forma que el JSON de POST /experiments, así la
API lo valida con las mismas reglas. Este módulo no importa nada de la API.

Unidades: distancia en UA, tiempo en años y masa en masas solares (G = 4π²).
"""

import math

TWO_PI = 2 * math.pi

# Sol en el origen y Tierra a 1 UA con velocidad circular: v = sqrt(G·M/r) = 2π.
SUN_EARTH = {
    "name": "Sol y Tierra",
    "dt": 0.001,
    "steps": 20_000,  # 20 años
    "save_every": 20,
    "bodies": [
        {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "mass": 1.0},
        {"x": 1.0, "y": 0.0, "vx": 0.0, "vy": TWO_PI, "mass": 3e-6},
    ],
}

# Dos estrellas iguales separadas 1 UA que giran alrededor de su centro común.
# ω = sqrt(G·(m1 + m2) / d³) = 2π·√2 y cada una está a 0,5 UA del centro,
# así que v = ω · 0,5 = π·√2. Período: 1/√2 ≈ 0,71 años.
_BINARY_SPEED = math.pi * math.sqrt(2)
BINARY = {
    "name": "Sistema binario",
    "dt": 0.001,
    "steps": 5_000,  # 5 años, unas 7 vueltas
    "save_every": 5,
    "bodies": [
        {"x": -0.5, "y": 0.0, "vx": 0.0, "vy": -_BINARY_SPEED, "mass": 1.0},
        {"x": 0.5, "y": 0.0, "vx": 0.0, "vy": _BINARY_SPEED, "mass": 1.0},
    ],
}

# Figura 8 de tres cuerpos iguales (Chenciner y Montgomery, 2000).
# Los valores se publicaron con G = 1; con G = 4π² las mismas órbitas se
# recorren 2π veces más rápido, así que las velocidades se multiplican por 2π.
# Período: 6,3259 / 2π ≈ 1,007 años.
_X1, _Y1 = 0.97000436, -0.24308753
_VX3, _VY3 = -0.93240737 * TWO_PI, -0.86473146 * TWO_PI
FIGURE_EIGHT = {
    "name": "Figura 8",
    "dt": 0.001,
    "steps": 5_000,  # 5 años, unas 5 vueltas
    "save_every": 5,
    "bodies": [
        {"x": _X1, "y": _Y1, "vx": -_VX3 / 2, "vy": -_VY3 / 2, "mass": 1.0},
        {"x": -_X1, "y": -_Y1, "vx": -_VX3 / 2, "vy": -_VY3 / 2, "mass": 1.0},
        {"x": 0.0, "y": 0.0, "vx": _VX3, "vy": _VY3, "mass": 1.0},
    ],
}

# Nombre en la URL -> escenario.
SCENARIOS = {
    "sol-tierra": SUN_EARTH,
    "binario": BINARY,
    "figura-8": FIGURE_EIGHT,
}
