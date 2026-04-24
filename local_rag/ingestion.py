import os
import uuid
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector

from local_rag.config import config

def ingestar_documentos():
    """
    =================================================================
    PROCESO DE INGESTA (ETL: Extract, Transform, Load)
    =================================================================
    Este script se encarga de:
    1. Extraer (Extract): Leer los PDFs y TXTs de la carpeta local.
    2. Transformar (Transform): Partir el texto en pequeños "chunks" 
       para no saturar la memoria del modelo.
    3. Cargar (Load): Convertir esos chunks a vectores usando Ollama
       y guardarlos en la base de datos PostgreSQL (pgvector).
    """
    print(f"📂 Buscando documentos en: {config.DOCS_DIR}")
    
    # ── 1. Preparación de la Carpeta ────────────────────────────
    if not os.path.exists(config.DOCS_DIR):
        os.makedirs(config.DOCS_DIR)
        print("Carpeta creada. Añade algunos documentos y vuelve a intentarlo.")
        return 0

    # ── 2. Extracción de Documentos (Loaders) ───────────────────
    # Mapeamos extensiones de archivo a su Loader específico de LangChain.
    # PyPDFLoader extrae texto de PDFs y TextLoader de archivos de texto plano.
    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader
    }
    
    documentos = []
    
    # Recorremos cada archivo en la carpeta "documentos/"
    for filename in os.listdir(config.DOCS_DIR):
        file_path = os.path.join(config.DOCS_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in loaders:
            print(f"📄 Leyendo {filename}...")
            loader_class = loaders[ext]
            loader = loader_class(file_path)
            # .load() extrae el texto y devuelve una lista de objetos Document
            # (Ej: 1 Document por cada página del PDF)
            documentos.extend(loader.load())
        else:
            if os.path.isfile(file_path):
                print(f"⚠️ Formato no soportado para: {filename}")

    if not documentos:
        print("❌ No se encontraron documentos válidos para ingestar.")
        return 0
        
    print(f"✅ Se leyeron {len(documentos)} páginas/documentos en total.")
    
    # ── 3. Transformación: Text Splitting ───────────────────────
    # Los LLMs tienen un límite de cuánto texto pueden leer a la vez (Context Window).
    # Por eso, dividimos un documento grande en trozos (chunks) más pequeños.
    print("✂️ Dividiendo documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Cada trozo tendrá máximo 1000 caracteres
        chunk_overlap=200,  # Se solapan 200 caracteres para no cortar ideas por la mitad
        add_start_index=True # Guarda en qué posición del documento original estaba el trozo
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Generados {len(chunks)} chunks.")

    # ── 4. Carga: Embeddings y Base de Datos Vectorial ──────────
    # Los "Embeddings" son representaciones matemáticas (vectores numéricos) del texto.
    # Usamos nomic-embed-text corriendo localmente en Ollama para generar estos vectores.
    print(f"🧠 Conectando a Ollama ({config.EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_URL
    )

    # Conectamos a PostgreSQL usando la extensión pgvector
    print(f"💾 Guardando embeddings en la base de datos (Colección: {config.COLLECTION_NAME})...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_URI,
        use_jsonb=True, # Usa formato JSON avanzado en PostgreSQL
    )
    
    # Generamos un identificador único (UUID) para cada trozo de texto
    # y los guardamos en la base de datos.
    ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    
    print("🚀 ¡Ingesta completada con éxito!")
    return len(chunks)

if __name__ == "__main__":
    ingestar_documentos()
