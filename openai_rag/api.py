from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import time

from openai_rag.ingestion import ingestar_documentos
from openai_rag.agent import agente_openai

# ══════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    mensaje: str
    thread_id: str = "default_user"

class PasoAgente(BaseModel):
    agente: str
    accion: str
    icono: str

class ChatResponse(BaseModel):
    respuesta: str
    thread_id: str
    pasos: list[PasoAgente]
    tiempo_ms: float

class IngestResponse(BaseModel):
    mensaje: str
    chunks_procesados: int

# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="☁️ OpenAI RAG API",
    description="API para procesar y consultar documentos usando la nube de OpenAI (gpt-4o-mini)",
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
    Lee la carpeta documentos/ y actualiza la base de datos vectorial en la nube.
    """
    try:
        total_chunks = ingestar_documentos()
        return IngestResponse(
            mensaje=f"Proceso finalizado. Se guardaron {total_chunks} fragmentos de texto en la base de datos de OpenAI.",
            chunks_procesados=total_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def endpoint_chat(request: ChatRequest):
    """
    Chatea con el agente pasándole un thread_id para mantener la memoria conversacional.
    """
    inicio = time.time()
    try:
        config_graph = {"configurable": {"thread_id": request.thread_id}}
        
        resultado = agente_openai.invoke(
            {"messages": [HumanMessage(content=request.mensaje)]},
            config=config_graph
        )
        
        respuesta = resultado["messages"][-1].content
        
        pasos = []
        for msg in resultado["messages"]:
            if type(msg).__name__ == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    pasos.append(PasoAgente(
                        agente="herramienta",
                        accion=f"Buscando en OpenAI Vectors: {tc.get('args', {}).get('consulta', '')}",
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
    print("🚀 Iniciando OpenAI RAG API en http://localhost:8001")
    print("📄 Swagger Docs: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)
