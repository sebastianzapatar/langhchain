"""
============================================================
API REST: Sistema Multi-Agente con FastAPI + LangGraph
============================================================
Expone el sistema multi-agente a través de endpoints REST.

Endpoints:
  POST /chat           → Enviar mensaje al agente (simple o multi-agente)
  POST /chat/stream     → Enviar mensaje con streaming de pasos
  GET  /health          → Estado del servidor
  GET  /agents          → Lista de agentes disponibles

Ejecutar:
  uvicorn api:app --reload --port 8000
============================================================
"""

import os
import json
from datetime import datetime
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Cargar variables de entorno ───────────────────────────────
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ OPENAI_API_KEY no configurada en .env")

# ── Importaciones de LangChain / LangGraph ────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver


# ══════════════════════════════════════════════════════════════
# HERRAMIENTAS (Tools)
# ══════════════════════════════════════════════════════════════

@tool
def calcular_operacion(expresion: str) -> str:
    """Calcula una operación matemática. Recibe una expresión como '2 + 3 * 4'.
    Soporta: suma (+), resta (-), multiplicación (*), división (/), potencia (**).
    """
    try:
        resultado = eval(expresion, {"__builtins__": {}}, {})
        return f"El resultado de {expresion} = {resultado}"
    except Exception as e:
        return f"Error al calcular '{expresion}': {str(e)}"


@tool
def obtener_fecha_actual() -> str:
    """Obtiene la fecha y hora actual del sistema."""
    ahora = datetime.now()
    return f"La fecha y hora actual es: {ahora.strftime('%d/%m/%Y %H:%M:%S')}"


@tool
def buscar_informacion(tema: str) -> str:
    """Busca información sobre un tema específico en la base de conocimiento interna.
    Retorna datos relevantes sobre tecnología, ciencia o programación.
    """
    conocimiento = {
        "python": "Python es un lenguaje de programación de alto nivel, interpretado y multiparadigma. "
                  "Fue creado por Guido van Rossum en 1991. Es popular en IA, ciencia de datos y web.",
        "langchain": "LangChain es un framework para construir aplicaciones con LLMs. "
                     "Permite encadenar prompts, modelos, herramientas y memoria de forma modular.",
        "langgraph": "LangGraph es una extensión de LangChain para construir flujos de trabajo "
                     "con agentes usando grafos de estados. Permite crear sistemas multi-agente.",
        "ia": "La Inteligencia Artificial es una rama de la ciencia de la computación que busca "
              "crear sistemas capaces de realizar tareas que requieren inteligencia humana.",
        "fastapi": "FastAPI es un framework moderno y rápido para construir APIs con Python. "
                   "Está basado en type hints, es asíncrono y genera documentación automática con Swagger.",
        "javascript": "JavaScript es un lenguaje de programación interpretado, orientado a objetos y basado en prototipos. "
                      "Es el lenguaje principal para desarrollo web frontend y también se usa en backend con Node.js.",
        "docker": "Docker es una plataforma de contenedores que permite empaquetar aplicaciones con todas sus "
                  "dependencias, asegurando que se ejecuten de la misma forma en cualquier entorno.",
        "react": "React es una biblioteca de JavaScript para construir interfaces de usuario. "
                 "Fue creada por Facebook y se basa en componentes reutilizables.",
    }

    tema_lower = tema.lower()
    for clave, valor in conocimiento.items():
        if clave in tema_lower or tema_lower in clave:
            return valor

    return f"No se encontró información sobre '{tema}' en la base de conocimiento."


@tool
def analizar_sentimiento(texto: str) -> str:
    """Analiza el sentimiento de un texto dado.
    Retorna si el sentimiento es positivo, negativo o neutro.
    """
    palabras_positivas = ["bueno", "excelente", "genial", "increíble", "feliz",
                          "amor", "éxito", "bien", "mejor", "fantástico", "perfecto"]
    palabras_negativas = ["malo", "terrible", "horrible", "triste", "odio",
                          "fracaso", "peor", "error", "problema", "difícil"]

    texto_lower = texto.lower()
    score_positivo = sum(1 for p in palabras_positivas if p in texto_lower)
    score_negativo = sum(1 for p in palabras_negativas if p in texto_lower)

    if score_positivo > score_negativo:
        sentimiento = "POSITIVO 😊"
    elif score_negativo > score_positivo:
        sentimiento = "NEGATIVO 😞"
    else:
        sentimiento = "NEUTRO 😐"

    return f"Análisis de sentimiento: {sentimiento} (positivo: {score_positivo}, negativo: {score_negativo})"


# Lista de herramientas
todas_las_herramientas = [calcular_operacion, obtener_fecha_actual, buscar_informacion, analizar_sentimiento]


# ══════════════════════════════════════════════════════════════
# CADENA SIMPLE (LangChain)
# ══════════════════════════════════════════════════════════════

def crear_cadena_simple():
    """Crea una cadena LCEL simple: Prompt → LLM → Parser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil y amigable. Responde en español de forma clara y concisa."),
        ("human", "{mensaje}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    return prompt | llm | parser


# ══════════════════════════════════════════════════════════════
# AGENTE REACT (LangGraph Prebuilt)
# ══════════════════════════════════════════════════════════════

def crear_agente_react():
    """Crea un agente ReAct con herramientas."""
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_react_agent(modelo, todas_las_herramientas)


# ══════════════════════════════════════════════════════════════
# SISTEMA MULTI-AGENTE CON SUPERVISOR
# ══════════════════════════════════════════════════════════════

class EstadoMultiAgente(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    siguiente_agente: str
    pasos: list  # Registro de pasos para el frontend


def crear_sistema_multi_agente():
    """Construye y compila el grafo multi-agente con supervisor."""

    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── Nodo Supervisor ──────────────────────────────────────
    def nodo_supervisor(estado: EstadoMultiAgente):
        prompt_supervisor = ChatPromptTemplate.from_messages([
            ("system", """Eres un supervisor que gestiona un equipo de agentes especializados.
Analiza la solicitud del usuario y decide qué agente debe encargarse.

Agentes disponibles:
- "calculadora": Para operaciones matemáticas y cálculos
- "investigador": Para buscar información sobre temas
- "analista": Para analizar sentimientos o textos
- "FINALIZAR": Cuando la tarea ya está completada

Responde SOLO con el nombre del agente (una sola palabra) o FINALIZAR.
Si hay múltiples tareas, elige la primera que aún no se haya completado."""),
            ("human", "{input}")
        ])

        historial = "\n".join([
            f"{'Usuario' if isinstance(m, HumanMessage) else 'Sistema'}: {m.content}"
            for m in estado["messages"]
        ])

        cadena = prompt_supervisor | modelo
        respuesta = cadena.invoke({"input": historial})
        decision = respuesta.content.strip().lower()

        pasos = estado.get("pasos", [])
        pasos.append({
            "agente": "supervisor",
            "accion": f"Delegando a: {decision}",
            "icono": "👔"
        })

        return {
            "siguiente_agente": decision,
            "messages": [AIMessage(content=f"[Supervisor] Delegando a: {decision}")],
            "pasos": pasos
        }

    # ── Agentes especializados ───────────────────────────────
    def nodo_calculadora(estado: EstadoMultiAgente):
        modelo_calc = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        modelo_calc_tools = modelo_calc.bind_tools([calcular_operacion])
        pregunta = estado["messages"][0].content

        respuesta = modelo_calc_tools.invoke([
            SystemMessage(content="Eres un agente calculadora. Usa calcular_operacion para resolver operaciones."),
            HumanMessage(content=pregunta)
        ])

        if respuesta.tool_calls:
            resultados = [calcular_operacion.invoke(tc["args"]) for tc in respuesta.tool_calls]
            contenido = f"[Calculadora] {'; '.join(resultados)}"
        else:
            contenido = f"[Calculadora] {respuesta.content}"

        pasos = estado.get("pasos", [])
        pasos.append({"agente": "calculadora", "accion": contenido, "icono": "🔢"})

        return {"messages": [AIMessage(content=contenido)], "pasos": pasos}

    def nodo_investigador(estado: EstadoMultiAgente):
        modelo_inv = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        modelo_inv_tools = modelo_inv.bind_tools([buscar_informacion])
        pregunta = estado["messages"][0].content

        respuesta = modelo_inv_tools.invoke([
            SystemMessage(content="Eres un agente investigador. Usa buscar_informacion para encontrar datos."),
            HumanMessage(content=pregunta)
        ])

        if respuesta.tool_calls:
            resultados = [buscar_informacion.invoke(tc["args"]) for tc in respuesta.tool_calls]
            contenido = f"[Investigador] {'; '.join(resultados)}"
        else:
            contenido = f"[Investigador] {respuesta.content}"

        pasos = estado.get("pasos", [])
        pasos.append({"agente": "investigador", "accion": contenido, "icono": "🔎"})

        return {"messages": [AIMessage(content=contenido)], "pasos": pasos}

    def nodo_analista(estado: EstadoMultiAgente):
        modelo_an = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        modelo_an_tools = modelo_an.bind_tools([analizar_sentimiento])
        pregunta = estado["messages"][0].content

        respuesta = modelo_an_tools.invoke([
            SystemMessage(content="Eres un agente analista. Usa analizar_sentimiento para analizar textos."),
            HumanMessage(content=pregunta)
        ])

        if respuesta.tool_calls:
            resultados = [analizar_sentimiento.invoke(tc["args"]) for tc in respuesta.tool_calls]
            contenido = f"[Analista] {'; '.join(resultados)}"
        else:
            contenido = f"[Analista] {respuesta.content}"

        pasos = estado.get("pasos", [])
        pasos.append({"agente": "analista", "accion": contenido, "icono": "📊"})

        return {"messages": [AIMessage(content=contenido)], "pasos": pasos}

    def nodo_respuesta_final(estado: EstadoMultiAgente):
        modelo_final = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        historial = "\n".join([m.content for m in estado["messages"]])

        respuesta = modelo_final.invoke([
            SystemMessage(content="Resume los resultados de los agentes en una respuesta clara y completa para el usuario. Responde en español."),
            HumanMessage(content=historial)
        ])

        pasos = estado.get("pasos", [])
        pasos.append({"agente": "respuesta_final", "accion": "Consolidando respuesta", "icono": "✅"})

        return {
            "messages": [AIMessage(content=respuesta.content)],
            "pasos": pasos
        }

    # ── Enrutamiento ─────────────────────────────────────────
    def enrutar_agente(estado: EstadoMultiAgente) -> str:
        siguiente = estado.get("siguiente_agente", "finalizar")
        rutas = {
            "calculadora": "calculadora",
            "investigador": "investigador",
            "analista": "analista",
        }
        return rutas.get(siguiente, "respuesta_final")

    # ── Construir Grafo ──────────────────────────────────────
    grafo = StateGraph(EstadoMultiAgente)

    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("calculadora", nodo_calculadora)
    grafo.add_node("investigador", nodo_investigador)
    grafo.add_node("analista", nodo_analista)
    grafo.add_node("respuesta_final", nodo_respuesta_final)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar_agente)
    grafo.add_edge("calculadora", "supervisor")
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_edge("respuesta_final", END)

    return grafo.compile()


# ══════════════════════════════════════════════════════════════
# AGENTE CON MEMORIA (MemorySaver)
# ══════════════════════════════════════════════════════════════

def crear_agente_con_memoria():
    """Crea un agente con memoria persistente por thread_id."""
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    checkpointer = MemorySaver()

    class EstadoMemoria(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def nodo_asistente(estado: EstadoMemoria):
        system_msg = SystemMessage(content="""Eres un asistente personal con memoria.
Tu trabajo es recordar TODO lo que el usuario te dice: nombre, preferencias, datos importantes.

Reglas:
- Siempre saluda al usuario por su nombre si lo conoces
- Menciona datos previos cuando sea relevante
- Si el usuario te dice algo nuevo, confírmale que lo recordarás
- Responde de forma amigable y concisa en español""")

        mensajes = [system_msg] + estado["messages"]
        modelo_con_tools = modelo.bind_tools(todas_las_herramientas)
        respuesta = modelo_con_tools.invoke(mensajes)
        return {"messages": [respuesta]}

    def debe_usar_tools(estado: EstadoMemoria):
        ultimo = estado["messages"][-1]
        if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
            return "tools"
        return END

    grafo = StateGraph(EstadoMemoria)
    grafo.add_node("asistente", nodo_asistente)
    grafo.add_node("tools", ToolNode(todas_las_herramientas))
    grafo.add_edge(START, "asistente")
    grafo.add_conditional_edges("asistente", debe_usar_tools)
    grafo.add_edge("tools", "asistente")

    return grafo.compile(checkpointer=checkpointer), checkpointer


# ══════════════════════════════════════════════════════════════
# INSTANCIAS (se crean una vez al iniciar el servidor)
# ══════════════════════════════════════════════════════════════

cadena_simple = crear_cadena_simple()
agente_react = crear_agente_react()
sistema_multi_agente = crear_sistema_multi_agente()
agente_memoria, memoria_checkpointer = crear_agente_con_memoria()


# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="🔗 LangChain Multi-Agent API",
    description="API REST para interactuar con agentes de LangChain y LangGraph",
    version="1.0.0"
)

# CORS: permite peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos Pydantic (Request/Response) ──────────────────────

class ChatRequest(BaseModel):
    """Solicitud de chat."""
    mensaje: str
    modo: str = "multi-agente"  # "simple", "react", "multi-agente"


class ChatMemoryRequest(BaseModel):
    """Solicitud de chat con memoria."""
    mensaje: str
    thread_id: str = "default"  # ID de conversación para aislar memoria


class PasoAgente(BaseModel):
    """Un paso en el flujo del multi-agente."""
    agente: str
    accion: str
    icono: str


class ChatResponse(BaseModel):
    """Respuesta del chat."""
    respuesta: str
    modo: str
    pasos: list[PasoAgente] = []
    tiempo_ms: float


class ChatMemoryResponse(BaseModel):
    """Respuesta del chat con memoria."""
    respuesta: str
    thread_id: str
    total_mensajes: int
    pasos: list[PasoAgente] = []
    tiempo_ms: float


class MemoryHistoryResponse(BaseModel):
    """Historial de memoria de un thread."""
    thread_id: str
    total_mensajes: int
    mensajes: list[dict]


class HealthResponse(BaseModel):
    """Estado del servidor."""
    status: str
    version: str
    modelos_disponibles: list[str]


class AgenteInfo(BaseModel):
    """Información de un agente."""
    nombre: str
    descripcion: str
    icono: str
    herramientas: list[str]


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica el estado del servidor."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        modelos_disponibles=["simple", "react", "multi-agente", "memoria"]
    )


@app.get("/agents", response_model=list[AgenteInfo])
async def listar_agentes():
    """Lista los agentes disponibles en el sistema."""
    return [
        AgenteInfo(
            nombre="Supervisor",
            descripcion="Analiza solicitudes y delega a agentes especializados",
            icono="👔",
            herramientas=[]
        ),
        AgenteInfo(
            nombre="Calculadora",
            descripcion="Resuelve operaciones matemáticas",
            icono="🔢",
            herramientas=["calcular_operacion"]
        ),
        AgenteInfo(
            nombre="Investigador",
            descripcion="Busca información en la base de conocimiento",
            icono="🔎",
            herramientas=["buscar_informacion"]
        ),
        AgenteInfo(
            nombre="Analista",
            descripcion="Analiza el sentimiento de textos",
            icono="📊",
            herramientas=["analizar_sentimiento"]
        ),
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal de chat.

    Modos disponibles:
    - "simple": Cadena LCEL básica (Prompt → LLM → Parser)
    - "react": Agente ReAct con herramientas
    - "multi-agente": Sistema multi-agente con supervisor
    """
    import time
    inicio = time.time()

    try:
        if request.modo == "simple":
            # ── Modo Simple: cadena LCEL ──────────────────────
            respuesta = cadena_simple.invoke({"mensaje": request.mensaje})
            pasos = [PasoAgente(agente="cadena_lcel", accion="Prompt → LLM → Parser", icono="⛓️")]

        elif request.modo == "react":
            # ── Modo ReAct: agente con herramientas ───────────
            resultado = agente_react.invoke({
                "messages": [HumanMessage(content=request.mensaje)]
            })
            respuesta = resultado["messages"][-1].content

            # Extraer pasos del rastro de mensajes
            pasos = []
            for msg in resultado["messages"]:
                tipo = type(msg).__name__
                if tipo == "HumanMessage":
                    pasos.append(PasoAgente(agente="usuario", accion=msg.content[:100], icono="👤"))
                elif tipo == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        pasos.append(PasoAgente(
                            agente="react",
                            accion=f"Llamando: {tc['name']}({tc['args']})",
                            icono="🤖"
                        ))
                elif tipo == "ToolMessage":
                    pasos.append(PasoAgente(agente="herramienta", accion=msg.content[:100], icono="🔧"))

        elif request.modo == "multi-agente":
            # ── Modo Multi-Agente: supervisor + especialistas ─
            resultado = sistema_multi_agente.invoke({
                "messages": [HumanMessage(content=request.mensaje)],
                "siguiente_agente": "",
                "pasos": []
            })
            respuesta = resultado["messages"][-1].content

            # Convertir pasos internos a PasoAgente
            pasos = [
                PasoAgente(**p) for p in resultado.get("pasos", [])
            ]

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Modo '{request.modo}' no válido. Usa: simple, react, multi-agente"
            )

        tiempo_ms = (time.time() - inicio) * 1000

        return ChatResponse(
            respuesta=respuesta,
            modo=request.modo,
            pasos=pasos,
            tiempo_ms=round(tiempo_ms, 1)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/memory", response_model=ChatMemoryResponse)
async def chat_with_memory(request: ChatMemoryRequest):
    """
    Chat con MEMORIA conversacional.

    Cada thread_id mantiene su propio historial de conversación.
    El agente recordará todo lo que le hayas dicho en el mismo thread.

    - Usa el mismo thread_id para continuar una conversación
    - Usa un thread_id diferente para iniciar una nueva
    """
    import time
    inicio = time.time()

    try:
        config = {"configurable": {"thread_id": request.thread_id}}

        resultado = agente_memoria.invoke(
            {"messages": [HumanMessage(content=request.mensaje)]},
            config=config
        )

        respuesta = resultado["messages"][-1].content

        # Extraer pasos
        pasos = []
        for msg in resultado["messages"]:
            tipo = type(msg).__name__
            if tipo == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    pasos.append(PasoAgente(
                        agente="memoria",
                        accion=f"Herramienta: {tc['name']}({tc['args']})",
                        icono="🔧"
                    ))

        # Obtener total de mensajes almacenados
        estado = agente_memoria.get_state(config)
        total_mensajes = len(estado.values.get("messages", []))

        pasos.insert(0, PasoAgente(
            agente="memoria",
            accion=f"Thread: {request.thread_id} ({total_mensajes} msgs en memoria)",
            icono="🧠"
        ))

        tiempo_ms = (time.time() - inicio) * 1000

        return ChatMemoryResponse(
            respuesta=respuesta,
            thread_id=request.thread_id,
            total_mensajes=total_mensajes,
            pasos=pasos,
            tiempo_ms=round(tiempo_ms, 1)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{thread_id}/history", response_model=MemoryHistoryResponse)
async def get_memory_history(thread_id: str):
    """
    Obtiene el historial completo de mensajes de un thread.
    Útil para inspeccionar qué recuerda el agente.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        estado = agente_memoria.get_state(config)

        mensajes_raw = estado.values.get("messages", [])
        mensajes = []
        for msg in mensajes_raw:
            mensajes.append({
                "tipo": "usuario" if isinstance(msg, HumanMessage) else "asistente",
                "contenido": msg.content,
                "icono": "👤" if isinstance(msg, HumanMessage) else "🤖"
            })

        return MemoryHistoryResponse(
            thread_id=thread_id,
            total_mensajes=len(mensajes),
            mensajes=mensajes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{thread_id}")
async def clear_memory(thread_id: str):
    """
    Nota: MemorySaver no soporta eliminación directa.
    En producción usarías PostgresSaver con DELETE.
    Este endpoint es informativo.
    """
    return {"message": f"Para limpiar '{thread_id}', reinicia el servidor o usa un nuevo thread_id."}


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor FastAPI en http://localhost:8000")
    print("📄 Docs en: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
