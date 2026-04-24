"""
Script para crear la cadena de RAG (Retrieval-Augmented Generation) 
usando el modelo local de HuggingFace mediante LCEL (LangChain Expression Language).
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from huggingface_rag.models import ModelFactory

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def crear_cadena_rag(vectorstore):
    """
    Crea la cadena RAG uniendo el VectorStore (como retriever) 
    con el LLM local (para generar la respuesta) usando LCEL.
    """
    # 1. Crear el Retriever a partir del VectorStore
    # k=3 significa que traerá los 3 fragmentos de texto más relevantes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. Inicializar el LLM Local
    llm = ModelFactory.crear_llm()
    
    # 3. Definir el Prompt para el RAG
    template = """Eres un asistente experto para tareas de respuesta a preguntas.
Usa los siguientes fragmentos de contexto recuperados para responder la pregunta.
Si no sabes la respuesta basándote en el contexto, simplemente di que no lo sabes.
Usa un máximo de tres oraciones y mantén la respuesta concisa.

Contexto:
{context}

Pregunta: {input}

Respuesta:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Crear la cadena con LCEL (LangChain Expression Language)
    # Esto es más robusto y compatible con todas las versiones de LangChain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    retrieve_docs = (lambda x: x["input"]) | retriever

    # Cadena final que devuelve tanto la respuesta como los documentos fuente
    rag_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    
    return rag_chain
