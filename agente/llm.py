"""Selección del proveedor de LLM por variable de entorno (`LLM_PROVIDER`).

El grafo no cambia según el proveedor: `bind_tools()` es parte de la interfaz
`BaseChatModel` de LangChain, así que los cuatro se enchufan igual en el nodo del modelo.
Lo único que varía es qué clase se instancia, con qué modelo y con qué API key.

Cada proveedor tiene **su propia** variable de modelo (`MODELO_OPENAI`, `MODELO_ANTHROPIC`,
…) en vez de una sola compartida: así cambiar `LLM_PROVIDER` no obliga a cambiar también el
modelo, y no puede quedar un `gpt-4o-mini` apuntando a Anthropic.

Los imports son perezosos —dentro de cada rama— para que no haga falta instalar los cuatro
paquetes: alcanza con el del proveedor que uses.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel


class ProveedorNoSoportado(ValueError):
    """`LLM_PROVIDER` tiene un valor que no corresponde a ninguno de los cuatro."""


# proveedor -> (variable con el modelo, modelo por defecto, variable con la API key, paquete pip)
# La API key en None significa que el proveedor no necesita ninguna (Ollama es local).
PROVEEDORES: dict[str, tuple[str, str, str | None, str]] = {
    "openai": ("MODELO_OPENAI", "gpt-4o-mini", "OPENAI_API_KEY", "langchain-openai"),
    "anthropic": ("MODELO_ANTHROPIC", "claude-haiku-4-5", "ANTHROPIC_API_KEY", "langchain-anthropic"),
    "gemini": ("MODELO_GEMINI", "gemini-flash-lite-latest", "GOOGLE_API_KEY", "langchain-google-genai"),
    "ollama": ("MODELO_OLLAMA", "llama3.1", None, "langchain-ollama"),
}


def _proveedor_elegido() -> str:
    """Lee `LLM_PROVIDER` y falla con un mensaje accionable si no es uno de los cuatro."""
    proveedor = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if proveedor not in PROVEEDORES:
        raise ProveedorNoSoportado(
            f"LLM_PROVIDER='{proveedor}' no es válido. "
            f"Opciones: {', '.join(sorted(PROVEEDORES))}."
        )
    return proveedor


def _api_key(variable: str | None, proveedor: str) -> None:
    """Verifica la API key del proveedor antes de instanciar nada."""
    if variable is None:  # Ollama corre local: no lleva clave
        return
    if not os.getenv(variable):
        raise RuntimeError(
            f"Falta la variable {variable}, que necesita LLM_PROVIDER={proveedor}.\n"
            f"  1. copy .env.example .env      (cp en Linux / macOS)\n"
            f"  2. completá {variable} en el archivo .env\n"
            f"El .env está en .gitignore: nunca se sube al repositorio."
        )


def crear_llm() -> tuple[BaseChatModel, str, str]:
    """Instancia el chat model del proveedor elegido.

    Devuelve `(llm, proveedor, modelo)`: los dos últimos son solo para poder imprimir
    contra qué se está corriendo, algo que se agradece cuando la traza sale distinta.
    """
    proveedor = _proveedor_elegido()
    variable_modelo, modelo_por_defecto, variable_clave, paquete = PROVEEDORES[proveedor]

    modelo = os.getenv(variable_modelo, modelo_por_defecto).strip() or modelo_por_defecto
    _api_key(variable_clave, proveedor)

    try:
        if proveedor == "openai":
            from langchain_openai import ChatOpenAI

            llm: BaseChatModel = ChatOpenAI(model=modelo, temperature=0)

        elif proveedor == "anthropic":
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model=modelo, temperature=0, max_tokens=4096)

        elif proveedor == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model=modelo, temperature=0)

        else:  # ollama
            from langchain_ollama import ChatOllama

            # base_url configurable por si el servidor de Ollama no corre en localhost.
            llm = ChatOllama(
                model=modelo,
                temperature=0,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
    except ImportError as error:
        raise RuntimeError(
            f"Falta el paquete de LangChain para LLM_PROVIDER={proveedor}:\n"
            f"  pip install {paquete}\n"
            f"(requirements.txt trae solo el del proveedor por defecto; los otros tres "
            f"están comentados ahí.)"
        ) from error

    return llm, proveedor, modelo
