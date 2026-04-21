"""
═══════════════════════════════════════════════════════════════
  VectorStoreManager — Gestión de pgvector en PostgreSQL
═══════════════════════════════════════════════════════════════
  Responsabilidades:
    1. Conectar a PostgreSQL con pgvector
    2. Almacenar chunks como vectores (embeddings)
    3. Buscar documentos por similitud semántica
    4. Proveer un Retriever para usar en cadenas RAG

  Soporta embeddings de:
    - OpenAI (text-embedding-3-small/large)
    - Ollama (nomic-embed-text, mxbai-embed-large)

  Base de datos:
    PostgreSQL + extensión pgvector
    Driver: psycopg3 (postgresql+psycopg://)
    Tabla de embeddings: langchain_pg_embedding
    Tabla de colecciones: langchain_pg_collection
═══════════════════════════════════════════════════════════════
"""

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from wikipedia_rag.config import Config
from wikipedia_rag.models import ModelFactory


class VectorStoreManager:
    """
    Gestiona la conexión y operaciones con pgvector en PostgreSQL.

    Encapsula:
      - Creación del modelo de embeddings
      - Conexión al vector store
      - Inserción de documentos (ingesta)
      - Búsqueda por similitud
      - Creación de retrievers para cadenas RAG

    Uso:
        config = Config()
        vs_manager = VectorStoreManager(config)

        # Ingestar documentos
        vs_manager.ingestar_documentos(chunks)

        # Buscar por similitud
        resultados = vs_manager.buscar("¿Qué es Colombia?", k=3)

        # Obtener retriever para RAG
        retriever = vs_manager.crear_retriever()
    """

    def __init__(self, config: Config):
        """
        Inicializa la conexión al vector store.

        Args:
            config: Configuración con database_url, collection_name y embedding_model.

        Internamente:
          1. Crea el modelo de embeddings (via ModelFactory)
          2. Conecta a PostgreSQL vía PGVector (crea tablas si no existen)
        """
        self._config = config

        # ── Crear modelo de embeddings (OpenAI o Ollama) ─────
        # ModelFactory decide qué clase usar según config.provider
        self._embeddings = ModelFactory.crear_embeddings(config)

        # ── Conectar al vector store en PostgreSQL ────────────
        # PGVector crea automáticamente:
        #   - Tabla langchain_pg_collection (colecciones)
        #   - Tabla langchain_pg_embedding (vectores + metadata)
        #   - Extensión pgvector si no existe
        self._vectorstore = PGVector(
            embeddings=self._embeddings,
            collection_name=config.collection_name,
            connection=config.database_url,
            use_jsonb=True,  # Metadata en JSONB (permite filtrado SQL)
        )

        print(f"  ✅ VectorStore conectado")
        print(f"     Colección: {config.collection_name}")
        print(f"     Embeddings: {config.embedding_model}")
        db_display = config.database_url.split("@")[1] if "@" in config.database_url else config.database_url
        print(f"     PostgreSQL: {db_display}")

    def ingestar_documentos(self, chunks: list[Document]) -> int:
        """
        Inserta chunks en PostgreSQL como vectores.

        Proceso interno:
          1. Para cada chunk, genera un embedding (vector de 1536 dims)
          2. Almacena el vector + texto + metadata en langchain_pg_embedding
          3. Inserta en lotes para evitar timeouts

        Args:
            chunks: Lista de Documents a vectorizar e insertar.

        Returns:
            Número total de chunks insertados.
        """
        print(f"\n🗄️  Insertando {len(chunks)} chunks en pgvector...")
        print(f"   Lotes de {self._config.chunk_batch_size}")
        print("-" * 55)

        total = 0
        batch_size = self._config.chunk_batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = self._vectorstore.add_documents(batch)
            total += len(ids)
            lote_num = i // batch_size + 1
            print(f"  ✅ Lote {lote_num}: {len(ids)} chunks ({total}/{len(chunks)})")

        print(f"\n  🎉 Total insertados: {total}")
        return total

    def tiene_documentos(self) -> bool:
        """
        Verifica si la colección ya tiene documentos.

        Returns:
            True si hay al menos un documento en la colección.
        """
        resultados = self._vectorstore.similarity_search("test", k=1)
        return len(resultados) > 0

    def buscar(self, consulta: str, k: int = 5) -> list[Document]:
        """
        Busca los K documentos más similares a la consulta.

        Proceso:
          1. Convierte la consulta en un vector (embedding)
          2. Busca los K vectores más cercanos por similitud coseno
          3. Retorna los Documents originales

        Args:
            consulta: Texto de búsqueda.
            k: Número de resultados a retornar.

        Returns:
            Lista de Documents ordenados por similitud (más similar primero).
        """
        return self._vectorstore.similarity_search(consulta, k=k)

    def buscar_con_scores(self, consulta: str, k: int = 5) -> list[tuple[Document, float]]:
        """
        Busca documentos similares e incluye el score de similitud.

        Args:
            consulta: Texto de búsqueda.
            k: Número de resultados.

        Returns:
            Lista de tuplas (Document, score). Score más bajo = más similar.
        """
        return self._vectorstore.similarity_search_with_score(consulta, k=k)

    def crear_retriever(self) -> VectorStoreRetriever:
        """
        Crea un Retriever para usar en cadenas RAG.

        Un Retriever es la interfaz estándar de LangChain para
        buscar documentos. Se integra directamente con cadenas
        LCEL y grafos de LangGraph.

        Returns:
            VectorStoreRetriever configurado con similitud coseno.
        """
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._config.retriever_k}
        )

    @property
    def vectorstore(self) -> PGVector:
        """Acceso directo al PGVector subyacente (para uso avanzado)."""
        return self._vectorstore
