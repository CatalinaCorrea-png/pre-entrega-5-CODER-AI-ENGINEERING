"""El ciclo ReAct completo con un LLM guionado (5) y su freno de seguridad (7)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agente.grafo import grafo

from tests.conftest import tool_call


def _herramientas_ejecutadas(mensajes) -> list[str]:
    return [m.name for m in mensajes if isinstance(m, ToolMessage)]


# --- 5. Razonamiento multi-paso ----------------------------------------------

@pytest.mark.asyncio
async def test_encadena_las_dos_herramientas_en_orden(llm_falso):
    """El escenario del Turno 1, determinista: el guion pide primero el id y después los
    pedidos. Lo que se prueba no es que el LLM lo decida bien —eso solo lo prueba una
    corrida real— sino que el grafo **sostiene** la cadena: que la observación de la primera
    herramienta vuelve al agente y habilita la segunda vuelta."""
    llm_falso(
        tool_call("buscar_cliente_por_nombre", {"nombre": "Ana García"}, "c1"),
        tool_call("buscar_pedidos", {"cliente_id": 102}, "c2"),
        AIMessage(content="Ana García tuvo 3 pedidos por un total de $14500."),
    )

    resultado = await grafo.compile().ainvoke(
        {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo Ana García?")]},
        config={"recursion_limit": 10},
    )

    assert _herramientas_ejecutadas(resultado["messages"]) == [
        "buscar_cliente_por_nombre",
        "buscar_pedidos",
    ]
    assert "3 pedidos" in resultado["messages"][-1].content


@pytest.mark.asyncio
async def test_las_observaciones_se_acumulan_en_el_estado(llm_falso):
    """El reducer `add_messages` de MessagesState concatena en vez de sobrescribir. Sin él,
    cada vuelta del ciclo borraría la observación anterior y el agente nunca podría usar el
    id que acaba de averiguar."""
    llm_falso(
        tool_call("buscar_cliente_por_nombre", {"nombre": "Ana García"}, "c1"),
        tool_call("buscar_pedidos", {"cliente_id": 102}, "c2"),
        AIMessage(content="Listo."),
    )

    resultado = await grafo.compile().ainvoke(
        {"messages": [HumanMessage(content="¿Y Ana García?")]},
        config={"recursion_limit": 10},
    )

    # humano + (ai + tool) + (ai + tool) + ai final
    assert len(resultado["messages"]) == 6
    tipos = [type(m).__name__ for m in resultado["messages"]]
    assert tipos == ["HumanMessage", "AIMessage", "ToolMessage",
                     "AIMessage", "ToolMessage", "AIMessage"]


@pytest.mark.asyncio
async def test_sin_tool_calls_responde_en_una_sola_vuelta(llm_falso):
    """El otro lado de la autonomía: el agente también tiene que poder NO usar herramientas.
    Un saludo no dispara ninguna."""
    llm_falso(AIMessage(content="¡Hola! ¿En qué te ayudo?"))

    resultado = await grafo.compile().ainvoke(
        {"messages": [HumanMessage(content="hola")]},
        config={"recursion_limit": 10},
    )

    assert _herramientas_ejecutadas(resultado["messages"]) == []
    assert len(resultado["messages"]) == 2


# --- Ciclo de retorno: el error de la herramienta vuelve al agente -----------

@pytest.mark.asyncio
async def test_el_error_de_la_herramienta_vuelve_al_agente(llm_falso):
    """`handle_tool_errors=True` hace que un fallo de la herramienta llegue al agente como
    un ToolMessage normal en vez de romper el programa. Acá el nombre no existe: el ERROR
    tiene que quedar en el estado y el agente tiene que llegar a razonar sobre él."""
    llm_falso(
        tool_call("buscar_cliente_por_nombre", {"nombre": "Roberto Sánchez"}, "c1"),
        AIMessage(content="No encontré ese cliente, ¿podés confirmarme el nombre?"),
    )

    resultado = await grafo.compile().ainvoke(
        {"messages": [HumanMessage(content="¿Pedidos de Roberto Sánchez?")]},
        config={"recursion_limit": 10},
    )

    observacion = next(m for m in resultado["messages"] if isinstance(m, ToolMessage))
    assert observacion.content.startswith("ERROR:")
    assert "confirmarme" in resultado["messages"][-1].content


@pytest.mark.asyncio
async def test_los_argumentos_invalidos_no_rompen_el_grafo(llm_falso):
    """Un LLM puede alucinar los argumentos. Acá manda un string donde el esquema pide un
    int: Pydantic lo rechaza, pero `handle_tool_errors=True` convierte esa excepción en un
    ToolMessage y el agente sigue vivo para reintentar."""
    llm_falso(
        tool_call("buscar_pedidos", {"cliente_id": "ciento dos"}, "c1"),
        AIMessage(content="Necesito el id numérico del cliente."),
    )

    resultado = await grafo.compile().ainvoke(
        {"messages": [HumanMessage(content="pedidos del cliente ciento dos")]},
        config={"recursion_limit": 10},
    )

    assert any(isinstance(m, ToolMessage) for m in resultado["messages"])
    assert isinstance(resultado["messages"][-1], AIMessage)


# --- 7. El freno del ciclo ---------------------------------------------------

@pytest.mark.asyncio
async def test_el_recursion_limit_corta_un_agente_colgado(llm_falso):
    """Un agente que pide herramientas y nunca se detiene. Sin `recursion_limit` esto sería
    un bucle infinito quemando tokens; con él, LangGraph corta con GraphRecursionError.

    El guion trae más pedidos que vueltas permitidas (cada vuelta consume dos pasos del
    límite: `agente` + `herramientas`) y cada uno con su propio id, porque dos AIMessage con
    el mismo id se pisan entre sí en vez de acumularse y el ciclo terminaría solo."""
    llm_falso(*(tool_call("buscar_pedidos", {"cliente_id": 102}, f"c{i}") for i in range(12)))

    with pytest.raises(GraphRecursionError):
        await grafo.compile().ainvoke(
            {"messages": [HumanMessage(content="dale para siempre")]},
            config={"recursion_limit": 10},
        )
