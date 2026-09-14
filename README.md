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
| LLM | intercambiable: OpenAI · Anthropic · Gemini · Ollama (`temperature=0`, `.bind_tools()`) |
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

- `LLM_PROVIDER` — `openai`, `anthropic`, `gemini` (por defecto) u `ollama`
- la API key **solo** del proveedor elegido (`ollama` no lleva ninguna)

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
             └──▶ END   (cuando el agente ya no pide herramientas)
```

`MessagesState` trae el reducer `add_messages`: cada nodo devuelve `{"messages": [...]}` y
LangGraph **concatena** en vez de sobrescribir. Sin ese reducer, cada vuelta del ciclo
borraría las observaciones anteriores y el agente no podría encadenar dos herramientas.

El grafo termina **solo** cuando el LLM decide no llamar más herramientas: `tools_condition`
devuelve `END` al ver un `AIMessage` sin `tool_calls`. No hay ninguna ruta manual.

### El proveedor del LLM (`agente/llm.py`)

El grafo no cambia según el proveedor: `bind_tools()` es parte de la interfaz `BaseChatModel`
de LangChain, así que los cuatro se enchufan igual en el nodo `agente`. Lo único que varía es
qué clase se instancia, con qué modelo y con qué API key.

| `LLM_PROVIDER` | Clase | Modelo por defecto | API key | Paquete |
|---|---|---|---|---|
| `gemini` *(default)* | `ChatGoogleGenerativeAI` | `gemini-flash-lite-latest` | `GOOGLE_API_KEY` | `langchain-google-genai` |
| `openai` | `ChatOpenAI` | `gpt-4o-mini` | `OPENAI_API_KEY` | `langchain-openai` |
| `anthropic` | `ChatAnthropic` | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` | `langchain-anthropic` |
| `ollama` | `ChatOllama` | `llama3.1` | — (local) | `langchain-ollama` |

Cada proveedor tiene **su propia** variable de modelo (`MODELO_GEMINI`, `MODELO_OPENAI`,
`MODELO_ANTHROPIC`, `MODELO_OLLAMA`) en vez de una sola compartida: así cambiar
`LLM_PROVIDER` no obliga a cambiar también el modelo, y no puede quedar un `gpt-4o-mini`
apuntando a Anthropic.

Los imports son perezosos —dentro de cada rama— para que no haga falta instalar los cuatro
paquetes. `requirements.txt` trae solo el de Gemini; los otros tres están comentados ahí. Si
elegís un proveedor sin su paquete, el error te dice exactamente qué instalar.

Cambiar de proveedor es editar una línea del `.env`:

```bash
LLM_PROVIDER=anthropic     # + pip install langchain-anthropic
```

Con Ollama no hace falta API key, pero el modelo tiene que **soportar tool calling**
(`llama3.1`, `qwen3`, `mistral-nemo`…): uno que no lo soporte nunca emite `tool_calls` y el
ciclo ReAct termina en la primera vuelta sin llamar a ninguna herramienta.

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
│   ├── llm.py            # selección de proveedor por LLM_PROVIDER
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
- **Modelo por defecto de Gemini**: `gemini-flash-lite-latest` porque la capa gratuita le da
  15 pedidos por minuto (contra 5 de `gemini-flash-latest`) y una corrida completa hace 8
  llamadas al LLM.
