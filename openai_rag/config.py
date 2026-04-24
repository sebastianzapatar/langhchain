import os
from dotenv import load_dotenv

# Cargar variables desde .env
load_dotenv()

class Config:
    """
    =================================================================
    CONFIGURACIÓN CENTRAL DEL SISTEMA (OpenAI RAG)
    =================================================================
    """
    
    # ── 1. Base de datos Vectorial (PostgreSQL + pgvector) ──────
    PG_USER = os.getenv("POSTGRES_USER", "postgres")
    PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
    PG_PORT = os.getenv("POSTGRES_PORT", "5432")
    PG_DB = os.getenv("POSTGRES_DB", "vectordb")
    
    PG_URI = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    
    # Nombre de la tabla distinto al local para no mezclar vectores de distinta dimensión
    COLLECTION_NAME = "documentos_openai"
    
    # ── 2. Modelos Cloud (OpenAI) ───────────────────────────────
    # Requiere que OPENAI_API_KEY esté configurada en el archivo .env
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")
        
    CHAT_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # ── 3. Rutas de Archivos ────────────────────────────────────
    # Usamos la misma carpeta de documentos para ambos sistemas
    DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentos")

config = Config()
