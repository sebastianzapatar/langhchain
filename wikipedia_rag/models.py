"""
═══════════════════════════════════════════════════════════════
  ModelFactory — Fábrica de modelos (OpenAI / Ollama)
═══════════════════════════════════════════════════════════════
  Encapsula la creación de modelos de chat y embeddings según
  el proveedor configurado. Esto permite cambiar entre OpenAI
  y Ollama sin modificar el resto del código.

  Patrón de diseño: Factory Method
  ────────────────────────────────────
  En lugar de que cada clase (Agent, VectorStore) decida qué
  modelo instanciar, delegan esa decisión al ModelFactory.

  Esto cumple el principio Open/Closed: si mañana se agrega
  un tercer proveedor (Anthropic, Groq, etc.), solo se modifica
  este archivo.

  Proveedores:
  ┌──────────┬─────────────────────────┬──────────────────────┐
  │ Provider │ Chat Model              │ Embedding Model      │
  ├──────────┼─────────────────────────┼──────────────────────┤
  │ openai   │ ChatOpenAI              │ OpenAIEmbeddings     │
  │          │ (gpt-4o-mini, gpt-4o)   │ (text-embedding-3-*) │
  ├──────────┼─────────────────────────┼──────────────────────┤
  │ ollama   │ ChatOllama              │ OllamaEmbeddings     │
  │          │ (llama3.2, mistral)     │ (nomic-embed-text)   │
  └──────────┴─────────────────────────┴──────────────────────┘
═══════════════════════════════════════════════════════════════
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from wikipedia_rag.config import Config


class ModelFactory:
    """
    Fábrica para crear modelos de chat y embeddings.

    Uso:
        config = Config(provider="openai")
        llm = ModelFactory.crear_llm(config)
        embeddings = ModelFactory.crear_embeddings(config)

        config_local = Config(provider="ollama")
        llm_local = ModelFactory.crear_llm(config_local)
        embeddings_local = ModelFactory.crear_embeddings(config_local)
    """

    @staticmethod
    def crear_llm(config: Config) -> BaseChatModel:
        """
        Crea el modelo de chat según el proveedor configurado.

        OpenAI (cloud):
            - Requiere OPENAI_API_KEY
            - Modelos: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
            - Mejor calidad, tiene costo por token

        Ollama (local):
            - Requiere servidor Ollama corriendo (ollama serve)
            - Modelos: llama3.2:1b, llama3.1:8b, mistral:7b
            - Gratis, privado, sin internet
            - Descargar modelo: ollama pull llama3.1:8b

        Args:
            config: Configuración con provider, chat_model y temperature.

        Returns:
            Instancia de BaseChatModel (ChatOpenAI o ChatOllama).
        """
        if config.provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=config.chat_model,
                temperature=config.chat_temperature,
            )
            print(f"  🌐 LLM: OpenAI → {config.chat_model}")
            return llm

        elif config.provider == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=config.chat_model,
                temperature=config.chat_temperature,
                base_url=config.ollama_base_url,
            )
            print(f"  🏠 LLM: Ollama → {config.chat_model} (local)")
            print(f"     URL: {config.ollama_base_url}")
            return llm

        else:
            raise ValueError(f"Proveedor no soportado: {config.provider}")

    @staticmethod
    def crear_embeddings(config: Config) -> Embeddings:
        """
        Crea el modelo de embeddings según el proveedor configurado.

        OpenAI (cloud):
            - text-embedding-3-small: 1536 dimensiones
            - text-embedding-3-large: 3072 dimensiones
            - Mejor calidad para búsqueda semántica

        Ollama (local):
            - nomic-embed-text: 768 dimensiones (recomendado)
            - mxbai-embed-large: 1024 dimensiones
            - Descargar: ollama pull nomic-embed-text

        ⚠️ IMPORTANTE: Los embeddings de OpenAI y Ollama NO son
        compatibles entre sí. Si ingresaste con OpenAI, debes
        buscar con OpenAI. Por eso se usa una colección diferente
        para cada proveedor.

        Args:
            config: Configuración con provider y embedding_model.

        Returns:
            Instancia de Embeddings (OpenAIEmbeddings o OllamaEmbeddings).
        """
        if config.provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=config.embedding_model)
            print(f"  🌐 Embeddings: OpenAI → {config.embedding_model}")
            return embeddings

        elif config.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(
                model=config.embedding_model,
                base_url=config.ollama_base_url,
            )
            print(f"  🏠 Embeddings: Ollama → {config.embedding_model} (local)")
            return embeddings

        else:
            raise ValueError(f"Proveedor no soportado: {config.provider}")

    @staticmethod
    def imprimir_info(config: Config) -> None:
        """Imprime información del proveedor configurado."""
        if config.provider == "openai":
            print(f"  ☁️  Proveedor: OpenAI (cloud)")
            print(f"     Chat: {config.chat_model}")
            print(f"     Embeddings: {config.embedding_model}")
            print(f"     ⚠️  Requiere OPENAI_API_KEY y tiene costo por uso")
        else:
            print(f"  🏠 Proveedor: Ollama (local)")
            print(f"     Chat: {config.chat_model}")
            print(f"     Embeddings: {config.embedding_model}")
            print(f"     Servidor: {config.ollama_base_url}")
            print(f"     ✅ Gratis, privado, sin internet")
