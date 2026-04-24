"""
Ejemplo básico de cómo usar un modelo local de HuggingFace con LangChain.
"""
from langchain_core.prompts import PromptTemplate
from huggingface_rag.models import ModelFactory

def run_example():
    print("Iniciando ejemplo básico de HuggingFace Local...")
    
    # 1. Cargar el LLM local
    llm = ModelFactory.crear_llm()
    
    # 2. Crear un prompt sencillo
    template = """
    Eres un asistente virtual útil, inteligente y amigable. 
    Por favor responde a la siguiente pregunta en español.
    
    Pregunta: {pregunta}
    
    Respuesta:
    """
    prompt = PromptTemplate.from_template(template)
    
    # 3. Crear la cadena
    chain = prompt | llm
    
    # 4. Ejecutar
    pregunta = "¿Cuáles son las tres leyes de la robótica de Isaac Asimov?"
    print(f"\nUsuario: {pregunta}\n")
    print("Asistente (pensando... esto puede tomar unos segundos usando CPU/GPU local):")
    
    respuesta = chain.invoke({"pregunta": pregunta})
    print(respuesta)

if __name__ == "__main__":
    run_example()
