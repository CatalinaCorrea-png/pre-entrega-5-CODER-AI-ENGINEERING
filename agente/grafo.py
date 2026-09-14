"""El grafo ReAct: nodo modelo + nodo herramientas + arista condicional.

El estado es `MessagesState`, que ya trae el reducer
`add_messages`: los mensajes se concatenan en vez de sobrescribirse, así el
ciclo no pierde las observaciones de las herramientas.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from agente.herramientas import herramientas

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "Falta la variable GOOGLE_API_KEY.\n"
        "  1. copy .env.example .env      (cp en Linux / macOS)\n"
        "  2. completá GOOGLE_API_KEY (gratis en https://aistudio.google.com/apikey)\n"
        "El .env está en .gitignore: nunca se sube al repositorio."
    )

# Archivo SQLite donde el checkpointer guarda el historial de cada thread_id.
DB_CHECKPOINTS = RAIZ / "checkpoints.sqlite"

# --- 1. LLM vinculado a las herramientas ---
# El default vale también sin .env; el porqué de flash-lite está en .env.example.
MODELO = os.getenv("MODELO_LLM", "gemini-flash-lite-latest")

llm = ChatGoogleGenerativeAI(model=MODELO, temperature=0)
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
