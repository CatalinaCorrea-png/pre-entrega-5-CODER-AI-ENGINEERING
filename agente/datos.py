"""Base de datos simulada: dos tablas separadas, a propósito.

Al usuario solo le pedimos el nombre del cliente, nunca su id. Eso obliga al
agente a razonar en dos pasos —"primero necesito el ID, después los pedidos"—
y es lo que produce el razonamiento multi-paso que pide la consigna: la
herramienta se invoca dos veces para una sola pregunta.
"""

CLIENTES_DB: dict[str, int] = {
    "ana garcía": 102,
    "juan pérez": 205,
    "maría lópez": 310,
}

PEDIDOS_DB: dict[int, dict[str, int]] = {
    102: {"pedidos": 3, "total": 14500},
    205: {"pedidos": 1, "total": 3200},
    310: {"pedidos": 5, "total": 27800},
}
