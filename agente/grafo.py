"""El grafo ReAct: nodo modelo + nodo herramientas + arista condicional.

El estado es `MessagesState`, que ya trae el reducer
`add_messages`: los mensajes se concatenan en vez de sobrescribirse, así el
ciclo no pierde las observaciones de las herramientas.
"""

from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from agente.herramientas import herramientas
from agente.llm import crear_llm

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")  # antes de crear_llm(): de ahí salen LLM_PROVIDER y las claves

# Archivo SQLite donde el checkpointer guarda el historial de cada thread_id.
DB_CHECKPOINTS = RAIZ / "checkpoints.sqlite"

# --- 1. LLM vinculado a las herramientas ---
# El proveedor y el modelo salen de LLM_PROVIDER y MODELO_<PROVEEDOR> (ver .env.example).
llm, PROVEEDOR, MODELO = crear_llm()
llm_con_herramientas = llm.bind_tools(herramientas)


# --- 2. Nodo de razonamiento ---
async def call_model(state: MessagesState) -> dict:
    """Le pasa el historial completo al LLM y devuelve su respuesta (que puede
    incluir tool_calls o ser la respuesta final)."""
    respuesta = await llm_con_herramientas.ainvoke(state["messages"])
    return {"messages": [respuesta]}


# --- 3. Orquestación ---
grafo = StateGraph(MessagesState)  # hereda el reducer add_messages, como pide la consigna
grafo.add_node("agente", call_model)
# handle_tool_errors=True: si la herramienta falla, el error vuelve al agente como un
# mensaje normal en vez de romper el programa. Eso habilita el "Ciclo de Retorno".
grafo.add_node("herramientas", ToolNode(herramientas, handle_tool_errors=True))

grafo.add_edge(START, "agente")

# tools_condition inspecciona el último mensaje del estado y devuelve:
#   "tools" -> el AIMessage trae tool_calls (hay que ejecutar herramientas)
#   END     -> el AIMessage NO trae tool_calls (respuesta final al usuario)
grafo.add_conditional_edges(
    "agente",
    tools_condition,
    {"tools": "herramientas", END: END},
)
grafo.add_edge("herramientas", "agente")  # el ciclo: observación -> razonamiento
