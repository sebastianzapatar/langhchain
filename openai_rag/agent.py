from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from openai_rag.config import config

# ══════════════════════════════════════════════════════════════
# RETRIEVER (Herramienta)
# ══════════════════════════════════════════════════════════════

def obtener_retriever():
    """Conecta a la base de datos de OpenAI y retorna el retriever."""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
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
    Busca información en los documentos locales del usuario (vectorizados con OpenAI).
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
# AGENTE CON MEMORIA (LangGraph + OpenAI)
# ══════════════════════════════════════════════════════════════

class EstadoAgente(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def crear_agente_openai():
    """
    Crea el agente RAG usando ChatGPT (gpt-4o-mini).
    """
    modelo = ChatOpenAI(
        model=config.CHAT_MODEL, 
        openai_api_key=config.OPENAI_API_KEY, 
        temperature=0.3
    )
    herramientas = [buscar_en_documentos]
    modelo_con_tools = modelo.bind_tools(herramientas)
    
    checkpointer = MemorySaver()

    def nodo_asistente(estado: EstadoAgente):
        mensajes_historial = estado["messages"]
        
        system_prompt = SystemMessage(content="""Eres un asistente de IA avanzado y amigable, potenciado por OpenAI.
Tu objetivo principal es ayudar al usuario basándote en los documentos locales.

REGLAS IMPORTANTES:
1. Tienes acceso a una herramienta de búsqueda de documentos locales.
2. Si el usuario te pregunta por documentos, reportes, manuales o información técnica, SIEMPRE usa la herramienta 'buscar_en_documentos'.
3. Recuerda el contexto de la conversación (memoria). Si el usuario hace referencia a algo dicho antes, usa el historial.
4. Responde SIEMPRE en español claro, profesional y estructurado.
5. Si la herramienta de búsqueda no devuelve información útil, indícale al usuario que no está en sus documentos.
6. RESPONDE ÚNICA Y EXCLUSIVAMENTE con la información obtenida de la herramienta 'buscar_en_documentos'.
7. Si te preguntan algo que NO está en los documentos recuperados, NO inventes la respuesta ni uses tu conocimiento general. Simplemente responde que no tienes esa información en la base de datos de documentos.
""")
        
        respuesta = modelo_con_tools.invoke([system_prompt] + mensajes_historial)
        return {"messages": [respuesta]}

    def enrutador_herramientas(estado: EstadoAgente):
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

# Instancia única
agente_openai, memoria_checkpointer = crear_agente_openai()
