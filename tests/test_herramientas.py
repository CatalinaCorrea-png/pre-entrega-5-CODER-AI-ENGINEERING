"""Las herramientas como funciones (1) y el contrato que ve el LLM (2)."""

import pytest

from agente.datos import CLIENTES_DB, PEDIDOS_DB
from agente.herramientas import buscar_cliente_por_nombre, buscar_pedidos, herramientas


# --- 1. Comportamiento -------------------------------------------------------

def test_busca_un_cliente_que_existe():
    assert "cliente_id=102" in buscar_cliente_por_nombre.invoke({"nombre": "Ana García"})


@pytest.mark.parametrize(
    "nombre", ["Ana García", "ana garcía", "ANA GARCÍA", "  Ana García  "]
)
def test_el_nombre_se_normaliza_antes_de_buscar(nombre):
    """El LLM copia el nombre como lo escribió el usuario, con la capitalización y los
    espacios que se le hayan colado. El `.strip().lower()` es lo que evita que la búsqueda
    falle por eso."""
    assert "cliente_id=102" in buscar_cliente_por_nombre.invoke({"nombre": nombre})


def test_un_cliente_inexistente_devuelve_ERROR():
    """El prefijo `ERROR:` no es decorativo: es lo que el LLM lee para darse cuenta de que
    la búsqueda falló y pedir una aclaración en vez de inventar un id."""
    respuesta = buscar_cliente_por_nombre.invoke({"nombre": "Roberto Sánchez"})
    assert respuesta.startswith("ERROR:")
    assert "Roberto Sánchez" in respuesta  # el mensaje nombra lo que no encontró


def test_busca_los_pedidos_de_un_id_que_existe():
    respuesta = buscar_pedidos.invoke({"cliente_id": 102})
    assert "pedidos=3" in respuesta
    assert "total=$14500" in respuesta


def test_un_id_inexistente_devuelve_ERROR():
    assert buscar_pedidos.invoke({"cliente_id": 999}).startswith("ERROR:")


def test_las_dos_tablas_son_coherentes():
    """Todo id de CLIENTES_DB tiene fila en PEDIDOS_DB. Si no, el agente encadenaría bien
    las dos herramientas y aun así terminaría en un ERROR, que es el peor caso para
    depurar: la lógica está bien y los datos no."""
    assert set(CLIENTES_DB.values()) <= set(PEDIDOS_DB)


# --- 2. El contrato que ve el LLM --------------------------------------------

def test_los_nombres_son_los_que_el_agente_va_a_invocar():
    assert {t.name for t in herramientas} == {"buscar_cliente_por_nombre", "buscar_pedidos"}


@pytest.mark.parametrize("herramienta", herramientas, ids=lambda h: h.name)
def test_la_descripcion_no_es_vaga(herramienta):
    """El docstring es lo ÚNICO que el LLM lee para decidir qué herramienta usar, y una
    descripción vaga no rompe nada: el agente simplemente empieza a elegir mal, en silencio.
    Estas dos rondan los 300 caracteres; un "busca datos" no llega a 15."""
    assert len(herramienta.description) > 100
    assert "ERROR" in herramienta.description  # avisa que puede fallar y cómo se ve


def test_el_esquema_tipa_el_id_como_entero():
    """`cliente_id` tiene que viajar como integer. Si el esquema dijera string, el LLM
    mandaría "102" y el lookup en PEDIDOS_DB —cuyas claves son int— fallaría siempre."""
    assert buscar_pedidos.args["cliente_id"]["type"] == "integer"
    assert buscar_cliente_por_nombre.args["nombre"]["type"] == "string"
