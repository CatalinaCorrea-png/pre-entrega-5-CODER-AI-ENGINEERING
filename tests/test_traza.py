"""Serialización de la traza ReAct al JSON del entregable (8)."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agente.traza import extraer_texto, serializar_traza

from tests.conftest import tool_call


# --- extraer_texto -----------------------------------------------------------

def test_content_de_texto_pasa_derecho():
    assert extraer_texto(AIMessage(content="hola")) == "hola"


def test_content_en_bloques_se_aplana():
    """El caso que motivó la función: Gemini a veces devuelve el content como una lista de
    bloques en vez de un string, y sin aplanarla la traza guardaría un repr de Python."""
    mensaje = AIMessage(content=[{"type": "text", "text": "hola "}, {"type": "text", "text": "mundo"}])
    assert extraer_texto(mensaje) == "hola mundo"


def test_los_bloques_que_no_son_texto_se_ignoran():
    mensaje = AIMessage(content=[{"type": "text", "text": "ok"}, {"type": "image", "url": "x"}])
    assert extraer_texto(mensaje) == "ok"


def test_un_content_vacio_no_rompe():
    """Un AIMessage que solo pide una herramienta tiene el content vacío. Es el caso más
    frecuente de la traza, no un borde raro."""
    assert extraer_texto(tool_call("buscar_pedidos", {"cliente_id": 102})) == ""


# --- serializar_traza --------------------------------------------------------

def _traza_de_ejemplo() -> list[dict]:
    return serializar_traza([
        HumanMessage(content="¿Cuántos pedidos tuvo Ana García?"),
        tool_call("buscar_cliente_por_nombre", {"nombre": "Ana García"}, "c1"),
        ToolMessage(content="cliente_id=102", name="buscar_cliente_por_nombre", tool_call_id="c1"),
        AIMessage(content="Tuvo 3 pedidos."),
    ])


def test_conserva_el_orden_y_el_tipo_de_cada_mensaje():
    """El orden ES la traza: es lo que muestra que el agente razonó en varios pasos."""
    assert [e["tipo"] for e in _traza_de_ejemplo()] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]


def test_registra_la_herramienta_pedida_y_sus_argumentos():
    entrada = _traza_de_ejemplo()[1]
    assert entrada["tool_calls"] == [
        {"nombre": "buscar_cliente_por_nombre", "argumentos": {"nombre": "Ana García"}}
    ]


def test_registra_que_herramienta_produjo_cada_observacion():
    assert _traza_de_ejemplo()[2]["herramienta"] == "buscar_cliente_por_nombre"


def test_los_mensajes_sin_tool_calls_no_llevan_esa_clave():
    """Ruido de menos en el JSON: un AIMessage final no arrastra un `tool_calls: []`."""
    assert "tool_calls" not in _traza_de_ejemplo()[3]
    assert "herramienta" not in _traza_de_ejemplo()[0]


def test_la_traza_es_serializable_a_json_con_acentos():
    """`json.dump(..., ensure_ascii=False)` es lo que hace main.py. Si algún valor no fuera
    serializable, el entregable se rompería recién al final de la corrida."""
    volcado = json.dumps(_traza_de_ejemplo(), ensure_ascii=False)
    assert "Ana García" in volcado  # sin escapar a \u00ed
    assert json.loads(volcado) == _traza_de_ejemplo()
