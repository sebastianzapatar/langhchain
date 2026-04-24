from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import time

from local_rag.ingestion import ingestar_documentos
from local_rag.agent import agente_local

# ══════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Modelo de datos esperado cuando un usuario envía un mensaje al chat."""
    mensaje: str
    thread_id: str = "default_user"  # ID único para recordar la memoria de este usuario

class PasoAgente(BaseModel):
    """Modelo para describir las acciones intermedias del agente (ej. usó la herramienta buscar)."""
    agente: str
    accion: str
    icono: str

class ChatResponse(BaseModel):
    """Modelo de datos que la API devuelve como respuesta al chat."""
    respuesta: str
    thread_id: str
    pasos: list[PasoAgente]
    tiempo_ms: float

class IngestResponse(BaseModel):
    """Modelo de datos que la API devuelve cuando termina de procesar los documentos."""
    mensaje: str
    chunks_procesados: int

# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="📚 Local RAG API",
    description="API para procesar y consultar documentos locales usando Ollama (llama3.1:8b)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingestar", response_model=IngestResponse)
async def endpoint_ingestar():
    """
    ENDPOINT 1: POST /ingestar
    ---------------------------------------------------
    Lee la carpeta documentos/ y actualiza la base de datos vectorial con los contenidos.
    Útil para llamar cuando subes un nuevo PDF y quieres que la IA lo aprenda.
    """
    try:
        total_chunks = ingestar_documentos()
        return IngestResponse(
            mensaje=f"Proceso finalizado. Se guardaron {total_chunks} fragmentos de texto en la base de datos.",
            chunks_procesados=total_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def endpoint_chat(request: ChatRequest):
    """
    ENDPOINT 2: POST /chat
    ---------------------------------------------------
    Chatea con el agente. Recibe el texto del usuario y un 'thread_id'.
    El 'thread_id' es crucial: le dice a LangGraph qué "caja de memoria" 
    abrir para recordar de qué venían hablando.
    """
    inicio = time.time()
    try:
        config_graph = {"configurable": {"thread_id": request.thread_id}}
        
        resultado = agente_local.invoke(
            {"messages": [HumanMessage(content=request.mensaje)]},
            config=config_graph
        )
        
        respuesta = resultado["messages"][-1].content
        
        # Extraer pasos (herramientas usadas)
        pasos = []
        for msg in resultado["messages"]:
            if type(msg).__name__ == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    pasos.append(PasoAgente(
                        agente="herramienta",
                        accion=f"Buscando: {tc.get('args', {}).get('consulta', '')}",
                        icono="🔎"
                    ))
        
        tiempo_ms = (time.time() - inicio) * 1000
        
        return ChatResponse(
            respuesta=respuesta,
            thread_id=request.thread_id,
            pasos=pasos,
            tiempo_ms=round(tiempo_ms, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando Local RAG API en http://localhost:8000")
    print("📄 Swagger Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
