# Pre-entrega 5 — Agente de razonamiento cíclico con memoria persistente

Agente **ReAct** construido con LangGraph: el propio LLM decide, turno a turno, si necesita
llamar a una herramienta o si ya puede responder. El ciclo
`agente → ¿tool? → herramientas → agente → …` no está cableado con `if/else`; lo dirige una
arista condicional (`tools_condition`) que mira si el último mensaje del modelo trae
`tool_calls`. El estado se guarda en SQLite, así que la conversación sobrevive al final del
proceso.

| Componente | Elección |
|---|---|
| Grafo | `StateGraph(MessagesState)` — hereda el reducer `add_messages` |
| Nodos | `agente` (LLM async) + `herramientas` (`ToolNode`) |
| Ruteo | `tools_condition` → `{"tools": "herramientas", END: END}` |
| Ciclo | `add_edge("herramientas", "agente")` |
| Herramientas | `buscar_cliente_por_nombre`, `buscar_pedidos` (`@tool`) |
| LLM | `gemini-flash-lite-latest` (`temperature=0`) con `.bind_tools()` |
| Persistencia | `AsyncSqliteSaver` sobre `checkpoints.sqlite` + `thread_id` |
| Tope de ciclo | `recursion_limit=10` |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

copy .env.example .env          # Windows  (cp en Linux / macOS)
```

Requiere **Python 3.12**.

Completar en `.env`:

- `GOOGLE_API_KEY` — gratis en [Google AI Studio](https://aistudio.google.com/apikey)
- `MODELO_LLM` — opcional, ya viene con un valor por defecto en `.env.example`

El `.env` está en `.gitignore` y nunca se sube al repositorio.

## Uso

```bash
python main.py                # usa el checkpoints.sqlite existente
python main.py --reiniciar    # borra el checkpoint y arranca la demo de cero
```

Corre los tres escenarios de la consigna y deja la traza ReAct en `traza_ejecucion.json`.

## Cómo funciona

### El grafo (`agente/grafo.py`)

```
START ──▶ agente ──tools_condition──▶ herramientas
             ▲                             │
             └─────────────────────────────┘
             │
             └──▶ END   (cuando el modelo ya no pide herramientas)
```

`MessagesState` trae el reducer `add_messages`: cada nodo devuelve `{"messages": [...]}` y
LangGraph **concatena** en vez de sobrescribir. Sin ese reducer, cada vuelta del ciclo
borraría las observaciones anteriores y el agente no podría encadenar dos herramientas.

El grafo termina **solo** cuando el LLM decide no llamar más herramientas: `tools_condition`
devuelve `END` al ver un `AIMessage` sin `tool_calls`. No hay ninguna ruta manual.

### Las herramientas (`agente/herramientas.py`)

La base de datos simulada tiene **dos tablas separadas a propósito** (`agente/datos.py`):
al usuario solo le pedimos el nombre del cliente, nunca su id. Eso obliga al agente a razonar
en dos pasos —"primero necesito el ID, después los pedidos"— sin que nadie le haya dicho el
orden.

El docstring de cada `@tool` es lo único que el LLM lee para decidir cuándo usarla, así que
dicen explícitamente qué reciben, qué devuelven y **en qué caso devuelven un `ERROR:`**.

### La persistencia (`main.py`)

`AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")` guarda una foto del estado después
de cada nodo. Con el mismo `thread_id`, LangGraph recupera el historial completo antes de
responder: por eso el Turno 2 entiende `"¿Y Juan Pérez?"`, una pregunta que no significa nada
por sí sola. El archivo SQLite hace que eso también funcione entre ejecuciones distintas del
script, no solo dentro de un proceso.

## Traza de ejecución

Salida real de `python main.py --reiniciar` (completa en
[`traza_ejecucion.json`](traza_ejecucion.json)).

### 1. Razonamiento multi-paso — la herramienta se invoca dos veces

```
Usuario: "¿Cuántos pedidos tuvo Ana García y cuál fue el total?"
  → tool_call: buscar_cliente_por_nombre(nombre="Ana García")
  → observación: Cliente encontrado: 'Ana García' → cliente_id=102
  → tool_call: buscar_pedidos(cliente_id=102)
  → observación: pedidos=3, total=$14500
  → el modelo ya tiene los datos: no pide más herramientas → END
Respuesta: "Ana García tuvo 3 pedidos y el monto total gastado fue de $14.500."
```

Dos vueltas del ciclo para una sola pregunta, y el orden lo eligió el agente.

### 2. Memoria entre turnos — mismo `thread_id`

```
Usuario: "¿Y Juan Pérez?"
  → buscar_cliente_por_nombre(nombre="Juan Pérez") → cliente_id=205
  → buscar_pedidos(cliente_id=205) → pedidos=1, total=$3200
Respuesta: "Juan Pérez tuvo 1 pedido y el monto total gastado fue de $3.200."
```

La pregunta no dice *qué* quiere saber de Juan Pérez. El agente lo sabe porque el
checkpointer le devolvió el turno anterior.

### 3. Ciclo de retorno — error de la herramienta → pedido de aclaración

```
Usuario: "¿Cuántos pedidos tuvo el cliente Roberto Sánchez?"
  → buscar_cliente_por_nombre(nombre="Roberto Sánchez")
  → observación: ERROR: no se encontró ningún cliente con el nombre 'Roberto Sánchez'...
Respuesta: "No se encontró ningún cliente registrado con el nombre 'Roberto Sánchez'.
            Por favor, verificá que el nombre esté bien escrito o completo."
```

Con `handle_tool_errors=True`, el error vuelve al modelo como un mensaje normal en vez de
romper el programa. El agente lo lee y, en vez de inventar un número, pide la aclaración.

### Estado persistido

```
Mensajes acumulados en 'conversacion-demo-1': 12
  [human] ¿Cuántos pedidos tuvo Ana García y cuál fue el total?
  [ai]    <tool_call>
  [tool]  Cliente encontrado: 'Ana García' → cliente_id=102
  [ai]    <tool_call>
  [tool]  pedidos=3, total=$14500
  [ai]    Ana García tuvo **3 pedidos** ...
  [human] ¿Y Juan Pérez?
  [ai]    <tool_call>
  [tool]  Cliente encontrado: 'Juan Pérez' → cliente_id=205
  [ai]    <tool_call>
  [tool]  pedidos=1, total=$3200
  [ai]    Juan Pérez tuvo **1 pedido** ...
```

## Estructura

```
pre-entrega-5/
├── agente/
│   ├── datos.py          # "base de datos" simulada (dos tablas separadas)
│   ├── herramientas.py   # @tool con docstrings descriptivos
│   ├── grafo.py          # StateGraph(MessagesState) + LLM + ToolNode + ciclo
│   └── traza.py          # serialización de la traza ReAct a JSON
├── main.py               # los tres escenarios de prueba + guardado de la traza
├── traza_ejecucion.json  # traza real de la última corrida
├── requirements.txt
├── .env.example
└── .gitignore
```

## Notas

- **`AsyncSqliteSaver` y no `SqliteSaver`**: la consigna pide persistencia con SQLite *y*
  gestión asíncrona. `AsyncSqliteSaver` es la variante async del mismo checkpointer, del
  paquete `langgraph-checkpoint-sqlite`; `SqliteSaver` es síncrono y bloquearía el
  event loop en cada nodo.
- **`recursion_limit=10`**: techo de pasos del ciclo. Sin él, un agente que se equivoca puede
  quedar llamando herramientas indefinidamente y gastando tokens.
- **Modelo**: `gemini-flash-lite-latest` porque la capa gratuita le da 15 pedidos por minuto
  (contra 5 de `gemini-flash-latest`) y una corrida completa hace 8 llamadas al LLM. Se puede
  cambiar con la variable de entorno `MODELO_LLM`.
