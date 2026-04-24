"""
Punto de entrada principal para ejecutar el RAG local con HuggingFace.
"""
from huggingface_rag.ingestion import crear_vectorstore_desde_wikipedia
from huggingface_rag.rag_chain import crear_cadena_rag
from huggingface_rag.config import config

def main():
    print("="*60)
    print("🤖 INICIANDO RAG CON HUGGINGFACE LOCAL 🤖")
    print(f"Modelo LLM: {config.LLM_MODEL}")
    print(f"Modelo Embeddings: {config.EMBEDDING_MODEL}")
    print("="*60)
    
    try:
        # 1. Ingestar datos (descargar, chunkear y vectorizar)
        tema = "Computación cuántica" # Puedes cambiar el tema aquí
        vectorstore = crear_vectorstore_desde_wikipedia(tema, lang="es")
        
        # 2. Crear la cadena RAG
        print("\n⚙️  Configurando cadena RAG...")
        rag_chain = crear_cadena_rag(vectorstore)
        print("✅ Cadena RAG lista.\n")
        
        # 3. Hacer una pregunta
        pregunta = "¿Qué es la computación cuántica según el texto?"
        print("="*60)
        print(f"👤 Pregunta: {pregunta}")
        print("="*60)
        
        print("\n🤖 Asistente procesando (esto puede tardar si se ejecuta en CPU)...")
        # El input a la cadena debe coincidir con el {input} definido en el prompt
        respuesta = rag_chain.invoke({"input": pregunta})
        
        print("\n📝 Respuesta Generada:")
        print("-" * 40)
        print(respuesta["answer"])
        print("-" * 40)
        
        print("\n📑 Fragmentos de contexto utilizados:")
        for i, doc in enumerate(respuesta["context"]):
            print(f"  [{i+1}] {doc.page_content[:150]}...")

    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")

if __name__ == "__main__":
    main()
