"""Estructura del grafo (3) y la decisión que dirige el ciclo (4)."""

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from agente.grafo import grafo

from tests.conftest import tool_call


# --- 3. Estructura -----------------------------------------------------------

def test_el_grafo_compila():
    """`compile()` no es un no-op: valida que haya un punto de entrada, que ningún nodo
    quede aislado y que todos los caminos lleguen a END."""
    assert grafo.compile() is not None


def test_los_nodos_son_los_esperados():
    nodos = set(grafo.compile().get_graph().nodes)
    assert {"agente", "herramientas"} <= nodos


def test_existe_la_arista_que_cierra_el_ciclo():
    """`herramientas -> agente` es *el* ciclo del título de la consigna. Sin esta arista el
    grafo sería lineal: el agente llamaría una herramienta y terminaría sin poder leer el
    resultado ni encadenar una segunda."""
    aristas = {(a.source, a.target) for a in grafo.compile().get_graph().edges}
    assert ("herramientas", "agente") in aristas


def test_la_bifurcacion_del_agente_es_condicional():
    """Del nodo `agente` salen dos caminos —seguir el ciclo o terminar— y los dos son
    condicionales: los elige `tools_condition`, no un if/else nuestro."""
    salidas = [a for a in grafo.compile().get_graph().edges if a.source == "agente"]
    assert {a.target for a in salidas} == {"herramientas", "__end__"}
    assert all(a.conditional for a in salidas)


# --- 4. El ruteo, sin ningún modelo de por medio -----------------------------

def test_con_tool_calls_el_ciclo_sigue():
    estado = {"messages": [tool_call("buscar_pedidos", {"cliente_id": 102})]}
    assert tools_condition(estado) == "tools"


def test_sin_tool_calls_el_grafo_termina():
    """La condición de corte: el grafo termina solo cuando el LLM devuelve un AIMessage sin
    tool_calls, es decir cuando él decide que ya no necesita herramientas."""
    estado = {"messages": [AIMessage(content="Ana García tuvo 3 pedidos.")]}
    assert tools_condition(estado) == END
