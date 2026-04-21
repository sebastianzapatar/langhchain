"""
═══════════════════════════════════════════════════════════════
  WikipediaScraper — Descarga y procesamiento de artículos
═══════════════════════════════════════════════════════════════
  Responsabilidades:
    1. Buscar artículos en Wikipedia (es.wikipedia.org)
    2. Descargar el contenido de cada artículo
    3. Dividir los artículos en chunks para vectorización

  Fuente de datos:
    Wikipedia en español → https://es.wikipedia.org
    API utilizada: WikipediaLoader de LangChain Community
    Paquete: pip install wikipedia langchain-community
═══════════════════════════════════════════════════════════════
"""

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from wikipedia_rag.config import Config


class WikipediaScraper:
    """
    Descarga artículos de Wikipedia y los divide en chunks
    listos para vectorización.

    Uso:
        config = Config()
        scraper = WikipediaScraper(config)
        documentos = scraper.descargar_articulos()
        chunks = scraper.dividir_en_chunks(documentos)

    Flujo:
        Wikipedia API → Documents → Text Splitter → Chunks
    """

    def __init__(self, config: Config):
        """
        Args:
            config: Configuración del proyecto con queries y parámetros.
        """
        self._config = config
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len,
        )

    def descargar_articulos(self) -> list[Document]:
        """
        Descarga artículos de Wikipedia para cada query configurado.

        Cada query busca en: https://es.wikipedia.org
        Por ejemplo:
          - "Colombia" → https://es.wikipedia.org/wiki/Colombia
          - "Bogotá"   → https://es.wikipedia.org/wiki/Bogotá

        Returns:
            Lista de Documents con el contenido de cada artículo.
            Cada Document incluye metadata con:
              - title: título del artículo
              - source: URL del artículo en Wikipedia
              - summary: resumen del artículo
        """
        print("\n📥 Descargando artículos de Wikipedia...")
        print(f"   Base URL: {self._config.wikipedia_base_url}")
        print(f"   Idioma: {self._config.wikipedia_lang}")
        print("-" * 55)

        documentos: list[Document] = []
        titulos_vistos: set[str] = set()

        for query in self._config.wikipedia_queries:
            print(f"\n  🔍 Buscando: \"{query}\"")
            print(f"     URL esperada: {self._config.get_wikipedia_url(query)}")

            articulos = self._buscar_query(query)

            for doc in articulos:
                titulo = doc.metadata.get("title", "Sin título")
                if titulo in titulos_vistos:
                    print(f"     ⏭️  \"{titulo}\" (duplicado, saltando)")
                    continue

                titulos_vistos.add(titulo)

                # ── Enriquecer metadata con la URL de Wikipedia ──
                url_real = self._config.get_wikipedia_url(titulo)
                doc.metadata["source_url"] = url_real
                doc.metadata["wikipedia_lang"] = self._config.wikipedia_lang
                doc.metadata["query_original"] = query

                documentos.append(doc)
                chars = len(doc.page_content)
                print(f"     ✅ \"{titulo}\" ({chars:,} chars)")
                print(f"        📎 {url_real}")

        self._imprimir_resumen(documentos)
        return documentos

    def dividir_en_chunks(self, documentos: list[Document]) -> list[Document]:
        """
        Divide documentos largos en fragmentos más pequeños (chunks).

        ¿Por qué dividir?
          - Los modelos de embedding tienen límite de tokens
          - Chunks más pequeños = búsqueda más precisa
          - Cada chunk mantiene la metadata del documento original

        Args:
            documentos: Lista de Documents descargados de Wikipedia.

        Returns:
            Lista de Documents divididos en chunks.
        """
        print(f"\n✂️  Dividiendo {len(documentos)} documentos en chunks...")
        print(f"   chunk_size={self._config.chunk_size}, overlap={self._config.chunk_overlap}")
        print("-" * 55)

        chunks = self._text_splitter.split_documents(documentos)

        total_chars = sum(len(c.page_content) for c in chunks)
        promedio = total_chars // len(chunks) if chunks else 0

        print(f"  📦 Chunks generados: {len(chunks)}")
        print(f"  📏 Tamaño promedio: {promedio} caracteres")

        if chunks:
            ejemplo = chunks[0]
            print(f"\n  📋 Ejemplo (primer chunk):")
            print(f"     Título: {ejemplo.metadata.get('title', 'N/A')}")
            print(f"     URL:    {ejemplo.metadata.get('source_url', 'N/A')}")
            print(f"     Texto:  {ejemplo.page_content[:120]}...")

        return chunks

    # ── Métodos privados ─────────────────────────────────────

    def _buscar_query(self, query: str) -> list[Document]:
        """Busca artículos en Wikipedia para un query dado."""
        try:
            loader = WikipediaLoader(
                query=query,
                lang=self._config.wikipedia_lang,
                load_max_docs=self._config.wikipedia_max_docs_per_query,
                doc_content_chars_max=self._config.wikipedia_max_chars_per_doc,
            )
            return loader.load()
        except Exception as e:
            print(f"     ⚠️  Error: {e}")
            return []

    def _imprimir_resumen(self, documentos: list[Document]) -> None:
        """Imprime un resumen de los artículos descargados."""
        total_chars = sum(len(d.page_content) for d in documentos)
        print(f"\n  {'═' * 45}")
        print(f"  📊 Resumen de descarga:")
        print(f"     Artículos únicos: {len(documentos)}")
        print(f"     Total caracteres: {total_chars:,}")
        print(f"  {'═' * 45}")

        # Listar todos los artículos con sus URLs
        print(f"\n  📚 Artículos descargados:")
        for i, doc in enumerate(documentos, 1):
            titulo = doc.metadata.get("title", "N/A")
            url = doc.metadata.get("source_url", "N/A")
            chars = len(doc.page_content)
            print(f"     {i:2d}. {titulo} ({chars:,} chars)")
            print(f"         🔗 {url}")
