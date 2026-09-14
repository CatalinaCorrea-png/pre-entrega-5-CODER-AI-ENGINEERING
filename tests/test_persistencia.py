"""Memoria entre turnos y entre procesos con AsyncSqliteSaver (6)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agente.grafo import grafo

from tests.conftest import tool_call

CONFIG_A = {"configurable": {"thread_id": "hilo-a"}, "recursion_limit": 10}
CONFIG_B = {"configurable": {"thread_id": "hilo-b"}, "recursion_limit": 10}


@pytest.fixture
def db(tmp_path):
    """Un SQLite descartable por test. Nunca el `checkpoints.sqlite` del repo: los tests no
    tienen por qué pisar la demo que quedó grabada."""
    return str(tmp_path / "checkpoints-test.sqlite")


@pytest.mark.asyncio
async def test_el_mismo_thread_id_acumula_el_historial(llm_falso, db):
    """Lo que hace posible el Turno 2: preguntar "¿Y Juan Pérez?" sin repetir el contexto.
    El segundo invoke tiene que ver los mensajes del primero."""
    llm_falso(AIMessage(content="primera"), AIMessage(content="segunda"))

    async with AsyncSqliteSaver.from_conn_string(db) as checkpointer:
        app = grafo.compile(checkpointer=checkpointer)

        await app.ainvoke({"messages": [HumanMessage(content="turno 1")]}, config=CONFIG_A)
        segundo = await app.ainvoke(
            {"messages": [HumanMessage(content="turno 2")]}, config=CONFIG_A
        )

    contenidos = [m.content for m in segundo["messages"]]
    assert contenidos == ["turno 1", "primera", "turno 2", "segunda"]


@pytest.mark.asyncio
async def test_un_thread_id_distinto_arranca_limpio(llm_falso, db):
    """La contracara: los hilos están aislados. Es lo que permite que la prueba del ciclo de
    retorno corra en `conversacion-error-1` sin arrastrar los turnos de la demo."""
    llm_falso(AIMessage(content="primera"), AIMessage(content="segunda"))

    async with AsyncSqliteSaver.from_conn_string(db) as checkpointer:
        app = grafo.compile(checkpointer=checkpointer)

        await app.ainvoke({"messages": [HumanMessage(content="turno 1")]}, config=CONFIG_A)
        otro = await app.ainvoke(
            {"messages": [HumanMessage(content="otra cosa")]}, config=CONFIG_B
        )

    assert [m.content for m in otro["messages"]] == ["otra cosa", "segunda"]


@pytest.mark.asyncio
async def test_el_historial_sobrevive_al_cierre_del_checkpointer(llm_falso, db):
    """La diferencia entre SqliteSaver y MemorySaver, y la razón de que el agente sea
    "resiliente": el estado vive en un archivo, no en el proceso. Acá se cierra el
    checkpointer por completo y se abre uno nuevo sobre el mismo archivo — que es lo que
    pasa cuando volvés a correr `main.py` mañana."""
    llm_falso(AIMessage(content="respuesta"))

    async with AsyncSqliteSaver.from_conn_string(db) as checkpointer:
        await grafo.compile(checkpointer=checkpointer).ainvoke(
            {"messages": [HumanMessage(content="acordate de esto")]}, config=CONFIG_A
        )

    async with AsyncSqliteSaver.from_conn_string(db) as checkpointer:
        snapshot = await grafo.compile(checkpointer=checkpointer).aget_state(CONFIG_A)

    assert [m.content for m in snapshot.values["messages"]] == [
        "acordate de esto",
        "respuesta",
    ]


@pytest.mark.asyncio
async def test_las_observaciones_de_las_herramientas_tambien_se_persisten(llm_falso, db):
    """No alcanza con guardar las preguntas y respuestas: la traza ReAct completa —los
    tool_calls y sus observaciones— tiene que quedar en el checkpoint, porque es de ahí que
    sale el `traza_ejecucion.json` del entregable."""
    llm_falso(
        tool_call("buscar_cliente_por_nombre", {"nombre": "Ana García"}, "c1"),
        AIMessage(content="Su id es 102."),
    )

    async with AsyncSqliteSaver.from_conn_string(db) as checkpointer:
        await grafo.compile(checkpointer=checkpointer).ainvoke(
            {"messages": [HumanMessage(content="¿id de Ana García?")]}, config=CONFIG_A
        )
        snapshot = await grafo.compile(checkpointer=checkpointer).aget_state(CONFIG_A)

    tipos = [type(m).__name__ for m in snapshot.values["messages"]]
    assert tipos == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
