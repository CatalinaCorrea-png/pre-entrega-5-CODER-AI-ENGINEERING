"""Prueba de ejecución del agente: razonamiento multi-paso, memoria y ciclo de retorno.

Corre los tres escenarios que pide la consigna y deja la traza ReAct en
`traza_ejecucion.json`.

    python main.py                # usa el checkpoints.sqlite existente
    python main.py --reiniciar    # borra el checkpoint y arranca la demo de cero
"""

import asyncio
import json
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agente.grafo import DB_CHECKPOINTS, RAIZ, grafo
from agente.traza import extraer_texto, serializar_traza

# recursion_limit: techo de pasos del ciclo. Sin él, un agente que se equivoca
# puede quedar llamando herramientas indefinidamente.
CONFIG = {"configurable": {"thread_id": "conversacion-demo-1"}, "recursion_limit": 10}
CONFIG_ERROR = {"configurable": {"thread_id": "conversacion-error-1"}, "recursion_limit": 10}

TRAZA_PATH = RAIZ / "traza_ejecucion.json"


def titulo(texto: str) -> None:
    print("\n" + "=" * 80)
    print(texto)
    print("=" * 80)


async def main() -> None:
    if "--reiniciar" in sys.argv:
        DB_CHECKPOINTS.unlink(missing_ok=True)
        print(f"Checkpoint borrado: {DB_CHECKPOINTS.name}")

    async with AsyncSqliteSaver.from_conn_string(str(DB_CHECKPOINTS)) as checkpointer:
        app = grafo.compile(checkpointer=checkpointer)

        # --- Turno 1: razonamiento multi-paso (la herramienta se invoca 2 veces) ---
        titulo("Turno 1: '¿Cuántos pedidos tuvo Ana García y cuál fue el total?'")
        resultado1 = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo Ana García y cuál fue el total?")]},
            config=CONFIG,
        )
        print(extraer_texto(resultado1["messages"][-1]))

        # --- Turno 2: mismo thread_id, la pregunta no se entiende sin el turno 1 ---
        titulo("Turno 2 (mismo thread_id): '¿Y Juan Pérez?'")
        resultado2 = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Y Juan Pérez?")]},
            config=CONFIG,
        )
        print(extraer_texto(resultado2["messages"][-1]))

        # --- Ciclo de retorno: la herramienta devuelve ERROR y el agente pide aclaración ---
        titulo("Ciclo de retorno: '¿Cuántos pedidos tuvo el cliente Roberto Sánchez?'")
        resultado_error = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo el cliente Roberto Sánchez?")]},
            config=CONFIG_ERROR,
        )
        print(extraer_texto(resultado_error["messages"][-1]))

        # --- Estado persistido en el checkpointer ---
        titulo("Estado persistido en el checkpointer")
        snapshot = await app.aget_state(CONFIG)
        print(f"Mensajes acumulados en '{CONFIG['configurable']['thread_id']}': "
              f"{len(snapshot.values['messages'])}")
        for m in snapshot.values["messages"]:
            print(f"  [{m.type}] {(extraer_texto(m) or '<tool_call>')[:70]}")

    # --- Traza ReAct como .json (entregable pedido) ---
    traza_completa = {
        "turno_1_multi_paso": serializar_traza(resultado1["messages"]),
        "turno_2_memoria": serializar_traza(resultado2["messages"]),
        "prueba_error_y_aclaracion": serializar_traza(resultado_error["messages"]),
    }
    TRAZA_PATH.write_text(
        json.dumps(traza_completa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTraza guardada en {TRAZA_PATH.name}")


if __name__ == "__main__":
    # Windows imprime en cp1252 y los acentos rompen la salida; UTF-8 explícito.
    for flujo in (sys.stdout, sys.stderr):
        flujo.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
