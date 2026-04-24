"""
═══════════════════════════════════════════════════════════════
  Configuración centralizada del proyecto
═══════════════════════════════════════════════════════════════
  Todas las constantes, URLs y parámetros del pipeline se
  definen aquí como una única fuente de verdad.

  Proveedores soportados:
    - "openai":  GPT-4o-mini, GPT-4o, text-embedding-3-small (API cloud)
    - "ollama":  Llama 3.2, Mistral, nomic-embed-text (local, gratis)
═══════════════════════════════════════════════════════════════
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """
    Configuración del pipeline Wikipedia → pgvector → RAG.

    Uso con OpenAI (por defecto):
        config = Config()

    Uso con Ollama (local, gratis):
        config = Config(
            provider="ollama",
            chat_model="llama3.1:8b",
            embedding_model="nomic-embed-text",
        )

    Uso desde CLI:
        python -m wikipedia_rag --provider ollama
    """

    # ══════════════════════════════════════════════════════════
    # PROVEEDOR DE MODELOS
    # ══════════════════════════════════════════════════════════
    # "openai" → Modelos en la nube (requiere API key, mejor calidad)
    # "ollama" → Modelos locales (gratis, privado, sin internet)
    #
    # Cada proveedor usa sus propias clases de LangChain:
    #   openai: ChatOpenAI, OpenAIEmbeddings
    #   ollama: ChatOllama, OllamaEmbeddings
    provider: str = "openai"

    # ══════════════════════════════════════════════════════════
    # Ollama — Configuración local
    # ══════════════════════════════════════════════════════════
    # Ollama corre un servidor local en http://localhost:11434
    # Los modelos se descargan con: ollama pull <modelo>
    #
    # Modelos de chat recomendados:
    #   - llama3.2:3b    → 2.0 GB, modelo medio robusto, mejor calidad
    #   - llama3.1:8b    → 4.7 GB, muy buena calidad
    #   - mistral:7b     → 4.1 GB, bueno en español
    #   - gemma2:9b      → 5.4 GB, excelente calidad
    #
    # Modelos de embeddings recomendados:
    #   - nomic-embed-text  → 274 MB, 768 dims, buena calidad
    #   - mxbai-embed-large → 670 MB, 1024 dims, mejor calidad
    #   - all-minilm        → 45 MB, 384 dims, ultraligero
    ollama_base_url: str = "http://localhost:11434"

    # ══════════════════════════════════════════════════════════
    # API Keys (solo para OpenAI)
    # ══════════════════════════════════════════════════════════
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        repr=False  # No mostrar la key en print
    )

    # ══════════════════════════════════════════════════════════
    # PostgreSQL + pgvector
    # ══════════════════════════════════════════════════════════
    # Formato: postgresql+psycopg://usuario:contraseña@host:puerto/base_datos
    # Driver: psycopg3 (moderno, async-ready)
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/vectordb"
        )
    )
    collection_name: str = "wikipedia_colombia"

    # ══════════════════════════════════════════════════════════
    # Modelo de Embeddings
    # ══════════════════════════════════════════════════════════
    # OpenAI:
    #   - text-embedding-3-small: 1536 dims, rápido y económico
    #   - text-embedding-3-large: 3072 dims, mayor precisión
    # Ollama:
    #   - nomic-embed-text: 768 dims, buena calidad (recomendado)
    #   - mxbai-embed-large: 1024 dims, mejor calidad
    embedding_model: str = "text-embedding-3-small"

    # ══════════════════════════════════════════════════════════
    # Modelo de Chat (LLM)
    # ══════════════════════════════════════════════════════════
    # OpenAI:
    #   - gpt-4o-mini: rápido, económico, buena calidad
    #   - gpt-4o: mejor calidad, más lento y costoso
    # Ollama:
    #   - llama3.1:8b: modelo muy robusto, excelente calidad
    #   - mistral:7b: muy bueno en español
    chat_model: str = "gpt-4o-mini"
    chat_temperature: float = 0  # 0 = determinístico, sin creatividad

    # ══════════════════════════════════════════════════════════
    # Wikipedia — Fuentes de datos
    # ══════════════════════════════════════════════════════════
    # IMPORTANTE: Cada query genera una búsqueda en Wikipedia.
    # La URL base es: https://es.wikipedia.org/wiki/{titulo_articulo}
    #
    # Ejemplo:
    #   query "Colombia" → https://es.wikipedia.org/wiki/Colombia
    #   query "Bogotá"   → https://es.wikipedia.org/wiki/Bogotá
    #
    # WikipediaLoader busca por relevancia y puede retornar
    # artículos distintos al query exacto.
    wikipedia_lang: str = "es"  # Idioma de Wikipedia
    wikipedia_max_docs_per_query: int = 2  # Máx artículos por búsqueda
    wikipedia_max_chars_per_doc: int = 15000  # Máx caracteres por artículo
    wikipedia_base_url: str = "https://es.wikipedia.org/wiki/"

    wikipedia_queries: list = field(default_factory=lambda: [
        # ─── Tema general ────────────────────────────────────
        "Colombia",                  # https://es.wikipedia.org/wiki/Colombia
        # ─── Historia ────────────────────────────────────────
        "Historia de Colombia",      # https://es.wikipedia.org/wiki/Historia_de_Colombia
        # ─── Geografía ───────────────────────────────────────
        "Geografía de Colombia",     # https://es.wikipedia.org/wiki/Geografía_de_Colombia
        # ─── Economía ────────────────────────────────────────
        "Economía de Colombia",      # https://es.wikipedia.org/wiki/Economía_de_Colombia
        # ─── Cultura ─────────────────────────────────────────
        "Cultura de Colombia",       # https://es.wikipedia.org/wiki/Cultura_de_Colombia
        # ─── Biodiversidad ───────────────────────────────────
        "Biodiversidad de Colombia", # https://es.wikipedia.org/wiki/Biodiversidad_de_Colombia
        # ─── Ciudades principales ────────────────────────────
        "Bogotá",                    # https://es.wikipedia.org/wiki/Bogotá
        "Medellín",                  # https://es.wikipedia.org/wiki/Medellín
        "Cartagena de Indias",       # https://es.wikipedia.org/wiki/Cartagena_de_Indias
        # ─── Productos emblemáticos ──────────────────────────
        "Café de Colombia",          # https://es.wikipedia.org/wiki/Café_de_Colombia
    ])

    # ══════════════════════════════════════════════════════════
    # Text Splitting (Chunking)
    # ══════════════════════════════════════════════════════════
    chunk_size: int = 1000       # Tamaño máximo de cada chunk (caracteres)
    chunk_overlap: int = 200     # Solapamiento entre chunks consecutivos
    chunk_batch_size: int = 50   # Tamaño de lote para inserción en pgvector

    # ══════════════════════════════════════════════════════════
    # Retriever (búsqueda vectorial)
    # ══════════════════════════════════════════════════════════
    retriever_k: int = 5         # Número de documentos a recuperar

    def validate(self) -> None:
        """Valida que la configuración sea correcta antes de ejecutar."""
        if self.provider not in ("openai", "ollama"):
            raise ValueError(
                f"❌ Proveedor '{self.provider}' no soportado.\n"
                "   Usa 'openai' o 'ollama'."
            )

        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY no encontrada.\n"
                "   Configúrala en el archivo .env o usa --provider ollama."
            )

        if not self.database_url:
            raise ValueError(
                "❌ DATABASE_URL no encontrada.\n"
                "   Configúrala en el archivo .env o como variable de entorno."
            )

    def get_wikipedia_url(self, titulo: str) -> str:
        """
        Construye la URL de Wikipedia para un artículo dado.

        Args:
            titulo: Título del artículo de Wikipedia.

        Returns:
            URL completa del artículo.

        Ejemplo:
            >>> config.get_wikipedia_url("Colombia")
            'https://es.wikipedia.org/wiki/Colombia'
        """
        return f"{self.wikipedia_base_url}{titulo.replace(' ', '_')}"

    def aplicar_defaults_ollama(self) -> None:
        """
        Aplica valores por defecto optimizados para Ollama.

        Si el usuario selecciona provider="ollama" pero no cambia
        los modelos, esta función asigna modelos locales por defecto.
        """
        if self.provider == "ollama":
            # Si los modelos siguen siendo los de OpenAI, cambiarlos
            if self.chat_model == "gpt-4o-mini":
                self.chat_model = "llama3.1:8b"
            if self.embedding_model == "text-embedding-3-small":
                self.embedding_model = "nomic-embed-text"
            # Ollama necesita una colección separada porque los
            # embeddings tienen dimensiones diferentes
            if self.collection_name == "wikipedia_colombia":
                self.collection_name = "wikipedia_colombia_ollama"
