import os
import uuid
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from openai_rag.config import config

def ingestar_documentos():
    """
    =================================================================
    PROCESO DE INGESTA (ETL) con OPENAI
    =================================================================
    """
    print(f"📂 Buscando documentos en: {config.DOCS_DIR}")
    
    # ── 1. Preparación de la Carpeta ────────────────────────────
    if not os.path.exists(config.DOCS_DIR):
        os.makedirs(config.DOCS_DIR)
        print("Carpeta creada. Añade algunos documentos y vuelve a intentarlo.")
        return 0

    # ── 2. Extracción de Documentos (Loaders) ───────────────────
    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader
    }
    
    documentos = []
    
    for filename in os.listdir(config.DOCS_DIR):
        file_path = os.path.join(config.DOCS_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in loaders:
            print(f"📄 Leyendo {filename}...")
            loader_class = loaders[ext]
            loader = loader_class(file_path)
            documentos.extend(loader.load())
        else:
            if os.path.isfile(file_path):
                print(f"⚠️ Formato no soportado para: {filename}")

    if not documentos:
        print("❌ No se encontraron documentos válidos para ingestar.")
        return 0
        
    print(f"✅ Se leyeron {len(documentos)} páginas/documentos en total.")
    
    # ── 3. Transformación: Text Splitting ───────────────────────
    print("✂️ Dividiendo documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Generados {len(chunks)} chunks.")

    # ── 4. Carga: Embeddings y Base de Datos Vectorial ──────────
    print(f"🧠 Conectando a OpenAI ({config.EMBEDDING_MODEL})...")
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )

    print(f"💾 Guardando embeddings en la base de datos (Colección: {config.COLLECTION_NAME})...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_URI,
        use_jsonb=True,
    )
    
    ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    
    print("🚀 ¡Ingesta completada con éxito en la colección OpenAI!")
    return len(chunks)

if __name__ == "__main__":
    ingestar_documentos()
