from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from local_rag.config import config

# ══════════════════════════════════════════════════════════════
# RETRIEVER (Herramienta)
# ══════════════════════════════════════════════════════════════

def obtener_retriever():
    """
    Se conecta a la base de datos PostgreSQL (pgvector) y devuelve
    un 'Retriever' (recuperador). El retriever es un objeto de LangChain
    que sabe cómo buscar el texto más relevante (búsqueda semántica) 
    cuando se le da una consulta.
    """
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_URL
    )
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_URI,
        use_jsonb=True,
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def buscar_en_documentos(consulta: str) -> str:
    """
    Esta función está decorada con @tool, lo que significa que el Agente
    de IA puede "decidir" llamarla por su cuenta cuando necesite información.
    
    Busca información en los documentos locales del usuario.
    Usa esta herramienta SIEMPRE que te pregunten sobre documentos, archivos, 
    o conocimiento que no sepas.
    """
    try:
        retriever = obtener_retriever()
        resultados = retriever.invoke(consulta)
        
        if not resultados:
            return "No se encontró información relevante en los documentos locales."
            
        contexto = []
        for doc in resultados:
            origen = doc.metadata.get("source", "Documento desconocido")
            contexto.append(f"--- Información de {origen} ---\n{doc.page_content}")
            
        return "\n\n".join(contexto)
    except Exception as e:
        return f"Error al buscar en documentos: {str(e)}"

# ══════════════════════════════════════════════════════════════
# AGENTE CON MEMORIA (LangGraph)
# ══════════════════════════════════════════════════════════════

class EstadoAgente(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def crear_agente_local():
    """
    =================================================================
    CREACIÓN DEL AGENTE (Cerebro + Memoria + Herramientas)
    =================================================================
    Crea el agente RAG usando llama3.1:8b.
    Utiliza un "Grafo de Estados" (StateGraph) que define cómo fluye
    la conversación.
    """
    modelo = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0.3)
    herramientas = [buscar_en_documentos]
    modelo_con_tools = modelo.bind_tools(herramientas)
    
    # MemorySaver guarda automáticamente todos los mensajes de una charla
    # usando un 'thread_id'. Esto le da memoria a corto plazo al agente.
    checkpointer = MemorySaver()

    def nodo_asistente(estado: EstadoAgente):
        """
        El 'Nodo' principal del grafo. Aquí es donde el modelo Llama 3.1
        piensa y decide su respuesta o si necesita usar una herramienta.
        """
        mensajes_historial = estado["messages"]
        
        system_prompt = SystemMessage(content="""Eres un asistente de IA avanzado y amigable.
Tu objetivo principal es ayudar al usuario basándote en los documentos locales.

REGLAS IMPORTANTES:
1. Tienes acceso a una herramienta de búsqueda de documentos locales.
2. Si el usuario te pregunta por documentos, reportes, manuales o información técnica, SIEMPRE usa la herramienta 'buscar_en_documentos'.
3. Recuerda el contexto de la conversación (memoria). Si el usuario hace referencia a algo dicho antes, usa el historial.
4. Responde SIEMPRE en español claro, profesional y estructurado.
5. Si la herramienta de búsqueda no devuelve información útil, indícale al usuario que no está en sus documentos.
""")
        
        # Invocamos el modelo
        respuesta = modelo_con_tools.invoke([system_prompt] + mensajes_historial)
        return {"messages": [respuesta]}

    def enrutador_herramientas(estado: EstadoAgente):
        """
        Esta función decide hacia dónde va el grafo después de que el modelo piensa.
        Si el modelo pidió usar la herramienta de buscar, va a "tools".
        Si el modelo simplemente respondió al usuario, termina (END).
        """
        ultimo_mensaje = estado["messages"][-1]
        if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
            return "tools"
        return END

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("asistente", nodo_asistente)
    grafo.add_node("tools", ToolNode(herramientas))
    
    grafo.add_edge(START, "asistente")
    grafo.add_conditional_edges("asistente", enrutador_herramientas)
    grafo.add_edge("tools", "asistente")

    agente_compilado = grafo.compile(checkpointer=checkpointer)
    
    return agente_compilado, checkpointer

# Se crea la instancia una sola vez cuando el módulo se importa
agente_local, memoria_checkpointer = crear_agente_local()
