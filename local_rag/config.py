import os

class Config:
    """
    =================================================================
    CONFIGURACIÓN CENTRAL DEL SISTEMA (Local RAG)
    =================================================================
    Aquí definimos todas las variables que necesita el sistema para
    funcionar. Al tenerlas en un solo archivo, es más fácil cambiar
    de base de datos o de modelo de IA en el futuro sin buscar
    en todo el código.
    """
    
    # ── 1. Base de datos Vectorial (PostgreSQL + pgvector) ──────
    # Leemos las variables de entorno (por si usamos Docker o un .env).
    # Si no existen, usamos valores por defecto (localhost, postgres, etc.)
    PG_USER = os.getenv("POSTGRES_USER", "postgres")
    PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
    PG_PORT = os.getenv("POSTGRES_PORT", "5432")
    PG_DB = os.getenv("POSTGRES_DB", "vectordb")
    
    # Armamos la cadena de conexión estándar para SQLAlchemy/psycopg
    PG_URI = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    
    # Nombre de la tabla/espacio donde guardaremos estos vectores
    COLLECTION_NAME = "documentos_locales"
    
    # ── 2. Modelos Locales (Ollama) ─────────────────────────────
    # Estos son los modelos que descargamos previamente con `ollama pull`
    CHAT_MODEL = "llama3.1:8b"           # El cerebro principal que "habla"
    EMBEDDING_MODEL = "nomic-embed-text" # El modelo que convierte texto en números (vectores)
    OLLAMA_URL = "http://localhost:11434" # Dirección del servidor de Ollama
    
    # ── 3. Rutas de Archivos ────────────────────────────────────
    # Calculamos de forma dinámica la ruta absoluta a la carpeta "documentos/"
    # Esto asegura que funcione sin importar desde dónde ejecutemos el script.
    DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentos")

# Instanciamos la clase para poder importarla fácilmente desde otros archivos
# Ejemplo: from local_rag.config import config
config = Config()
