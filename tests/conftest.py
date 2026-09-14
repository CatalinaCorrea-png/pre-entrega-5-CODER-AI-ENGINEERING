"""Configuración común de la suite.

Ningún test de este directorio llama a la API: el LLM se reemplaza por un guion fijo de
respuestas, así que la suite corre gratis, sin red y con resultados deterministas.
"""

import os

# Antes de importar `agente.grafo`: ese módulo exige GOOGLE_API_KEY al importarse e
# instancia el cliente de Gemini. Con una clave de mentira alcanza —construir el cliente
# no dispara ninguna llamada— y así la suite corre en una máquina sin credenciales.
# `setdefault` no pisa una clave real que ya esté en el entorno.
os.environ.setdefault("GOOGLE_API_KEY", "clave-de-prueba-sin-uso")

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import agente.grafo


def tool_call(nombre: str, args: dict, id_: str = "call-1") -> AIMessage:
    """Construye el AIMessage que el LLM real devolvería para pedir una herramienta.

    El `id` del propio AIMessage se fija a partir de `id_` a propósito. `add_messages`
    deduplica por ese id: dos respuestas del guion que compartan uno se **reemplazan** en
    vez de acumularse, y el ciclo termina donde no debería.
    """
    return AIMessage(
        id=f"ai-{id_}",
        content="",
        tool_calls=[{"name": nombre, "args": args, "id": id_}],
    )


class _LLMProhibido:
    """Ocupa el lugar del cliente de Gemini durante los tests y falla si alguien lo usa."""

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError(
            "Un test intentó usar el LLM real. Pedí la fixture `llm_falso` e instalá un "
            "guion de respuestas antes de invocar el grafo."
        )

    def invoke(self, *args, **kwargs):
        return self.ainvoke()


@pytest.fixture(autouse=True)
def sin_llm_real(monkeypatch):
    """Red de seguridad: ningún test puede llamar a la API, ni por olvido.

    Se aplica a **todos** los tests. Los que piden `llm_falso` vuelven a pisar el global con
    su guion (esta fixture corre antes, por ser autouse), así que la bomba solo queda armada
    para el que invoque el grafo sin haber instalado un fake: en vez de gastar tokens en
    silencio, falla con un mensaje que dice qué le falta.
    """
    monkeypatch.setattr(agente.grafo, "llm_con_herramientas", _LLMProhibido())


@pytest.fixture
def llm_falso(monkeypatch):
    """Reemplaza el LLM del nodo `agente` por un guion fijo de respuestas.

    `call_model` resuelve `llm_con_herramientas` como global de su módulo en cada llamada,
    así que pisar esa variable desvía el grafo entero sin tocar el código de producción ni
    reconstruir el grafo en el test (lo que dejaría de probar el grafo real).

    Ojo con un detalle del fake: cuando se le acaban las respuestas **vuelve a la primera**.
    `test_el_recursion_limit_corta_un_agente_colgado` se apoya en eso para simular un agente
    que nunca deja de pedir herramientas.
    """

    def instalar(*respuestas: AIMessage) -> FakeMessagesListChatModel:
        fake = FakeMessagesListChatModel(responses=list(respuestas))
        monkeypatch.setattr(agente.grafo, "llm_con_herramientas", fake)
        return fake

    return instalar
