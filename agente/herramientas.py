"""Herramientas del agente.

El docstring de cada función es lo único que el LLM lee para decidir cuándo
usarla: por eso dicen explícitamente qué reciben, qué devuelven y en qué caso
devuelven un ERROR.
"""

from langchain_core.tools import tool

from agente.datos import CLIENTES_DB, PEDIDOS_DB


@tool
def buscar_cliente_por_nombre(nombre: str) -> str:
    """Busca el ID interno (cliente_id) de un cliente a partir de su nombre completo.
    Usar esta herramienta SIEMPRE que el usuario mencione un cliente por su nombre
    y necesites averiguar su cliente_id antes de poder consultar sus pedidos.
    Devuelve un mensaje de ERROR si el nombre no coincide exactamente con ningún
    cliente registrado."""
    clave = nombre.strip().lower()
    if clave not in CLIENTES_DB:
        return (
            f"ERROR: no se encontró ningún cliente con el nombre '{nombre}'. "
            "Verificá que el nombre esté completo y bien escrito."
        )
    return f"Cliente encontrado: '{nombre}' → cliente_id={CLIENTES_DB[clave]}"


@tool
def buscar_pedidos(cliente_id: int) -> str:
    """Busca la cantidad de pedidos y el monto total gastado por un cliente,
    dado su cliente_id NUMÉRICO (no su nombre — si solo tenés el nombre, primero
    usá buscar_cliente_por_nombre para obtener el ID).
    Devuelve un mensaje de ERROR si el cliente_id no existe en la base de datos."""
    if cliente_id not in PEDIDOS_DB:
        return f"ERROR: no existe ningún cliente con id={cliente_id}."
    datos = PEDIDOS_DB[cliente_id]
    return f"pedidos={datos['pedidos']}, total=${datos['total']}"


herramientas = [buscar_cliente_por_nombre, buscar_pedidos]
