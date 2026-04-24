"""
Script para ingerir datos de Wikipedia y guardarlos en una base de datos vectorial en memoria.
Usamos los embeddings locales de HuggingFace.
"""
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from huggingface_rag.models import ModelFactory

def crear_vectorstore_desde_wikipedia(tema: str, lang: str = "es") -> InMemoryVectorStore:
    """
    Descarga un artículo de Wikipedia, lo divide en partes y 
    crea un VectorStore en memoria usando embeddings locales.
    """
    print(f"\n📚 1. Buscando '{tema}' en Wikipedia ({lang})...")
    
    # Descargar artículo de Wikipedia
    loader = WikipediaLoader(query=tema, lang=lang, load_max_docs=1)
    docs = loader.load()
    
    if not docs:
        raise ValueError(f"No se encontró información para '{tema}' en Wikipedia.")
    
    print(f"   Encontrado artículo: {docs[0].metadata.get('title', tema)}")
    
    # Dividir el documento en chunks más pequeños
    print("✂️  2. Dividiendo el texto en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"   Generados {len(splits)} chunks.")
    
    # Inicializar embeddings locales
    print("🧠 3. Inicializando modelo de Embeddings locales...")
    embeddings = ModelFactory.crear_embeddings()
    
    # Crear vector store en memoria
    print("💾 4. Guardando en base de datos vectorial (en memoria)...")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    
    print("✅ VectorStore listo para ser consultado.")
    return vectorstore

if __name__ == "__main__":
    # Prueba rápida
    vs = crear_vectorstore_desde_wikipedia("inteligencia artificial")
    print(f"Total de vectores guardados: {len(vs.similarity_search('hola'))} (ejemplo de búsqueda)")
