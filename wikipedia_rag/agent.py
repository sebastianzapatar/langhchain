"""
═══════════════════════════════════════════════════════════════
  RAGAgent — Agente conversacional con memoria y RAG
═══════════════════════════════════════════════════════════════
  Responsabilidades:
    1. Buscar documentos relevantes en pgvector (Retrieval)
    2. Generar respuestas basadas en el contexto (Generation)
    3. Mantener memoria conversacional (Memory)
    4. NO alucinar — responder solo con lo que tiene

  Memoria Conversacional (MemorySaver):
  ─────────────────────────────────────
  LangGraph utiliza un sistema de checkpoints para persistir
  el estado de la conversación entre invocaciones.

  ¿Cómo funciona?
    1. Cada conversación tiene un `thread_id` único
    2. El MemorySaver guarda el historial de mensajes por thread
    3. Al invocar el grafo con el mismo thread_id, se carga
       automáticamente todo el historial previo
    4. Diferentes thread_id = conversaciones independientes

  Ejemplo de flujo de memoria:
    thread_id = "usuario-1"

    Invocación 1: "¿Cuál es la capital?"
      → Estado: [HumanMsg("¿Cuál es la capital?"), AIMsg("Bogotá")]
      → MemorySaver GUARDA este estado

    Invocación 2: "¿Cuántos habitantes tiene?"
      → MemorySaver CARGA el estado anterior
      → Estado: [HumanMsg("¿Cuál es la capital?"), AIMsg("Bogotá"),
                  HumanMsg("¿Cuántos habitantes tiene?")]
      → El LLM VE toda la conversación y sabe que "tiene"
        se refiere a Bogotá
      → Estado actualizado se GUARDA de nuevo

  Grafo LangGraph:
    ┌───────┐   ┌────────────┐   ┌────────────────┐
    │ START │──▶│ Recuperar  │──▶│   Responder    │──▶ END
    └───────┘   │ (pgvector) │   │ (OpenAI/Ollama)│
                └────────────┘   └────────────────┘
                                       ▲
                                  MemorySaver
                              (historial por thread)
═══════════════════════════════════════════════════════════════
"""

import time
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from wikipedia_rag.config import Config
from wikipedia_rag.vectorstore import VectorStoreManager
from wikipedia_rag.models import ModelFactory


class RAGAgent:
    """
    Agente conversacional con RAG y memoria.

    Características:
      - Busca documentos relevantes en pgvector antes de responder
      - Mantiene memoria de la conversación (MemorySaver + thread_id)
      - NO alucina: solo responde con información de la base de datos
      - Cita las fuentes (artículos de Wikipedia) en sus respuestas

    Uso:
        config = Config()
        vs_manager = VectorStoreManager(config)
        agente = RAGAgent(config, vs_manager)

        # Hacer una pregunta
        respuesta = agente.preguntar(
            "¿Cuál es la capital de Colombia?",
            thread_id="conversacion-1"
        )

        # Pregunta de seguimiento (usa la memoria)
        respuesta = agente.preguntar(
            "¿Cuántos habitantes tiene?",    # ← sabe que habla de Bogotá
            thread_id="conversacion-1"        # ← mismo thread = misma memoria
        )

        # Otra conversación independiente
        respuesta = agente.preguntar(
            "¿Qué es el café de Colombia?",
            thread_id="conversacion-2"         # ← thread diferente = memoria separada
        )
    """

    # ── Estado del grafo LangGraph ────────────────────────────
    # TypedDict define la estructura de datos que fluye por el grafo.
    # `add_messages` es un reducer que ACUMULA mensajes en lugar
    # de reemplazarlos — esto es lo que permite la memoria.

    class _EstadoRAG(TypedDict):
        # `add_messages` acumula: cada invocación AGREGA mensajes
        # al historial existente en lugar de reemplazarlo.
        # Esto es CLAVE para la memoria conversacional.
        messages: Annotated[list[BaseMessage], add_messages]
        contexto: str       # Documentos recuperados de pgvector
        fuentes: list[str]  # Títulos de los artículos usados
        es_relevante: bool  # ← NUEVO: ¿La pregunta tiene contexto relevante?

    # ══════════════════════════════════════════════════════════
    # ANTI-ALUCINACIÓN POR CÓDIGO
    # ══════════════════════════════════════════════════════════
    # En lugar de confiar en el LLM para que "no alucine",
    # verificamos POR CÓDIGO si la pregunta está relacionada
    # con los datos de Colombia en nuestra base de datos.
    #
    # Estrategia: Validación por contenido
    # ────────────────────────────────────
    # 1. Los documentos recuperados de pgvector SIEMPRE serán
    #    los "más similares", incluso si la pregunta es de otro tema.
    # 2. Verificamos si los documentos contienen palabras clave
    #    del TEMA de la pregunta.
    # 3. Si la pregunta es sobre "Perú" pero los documentos son
    #    sobre "Colombia/Bogotá/Medellín", la rechazamos.
    # ══════════════════════════════════════════════════════════

    # Temas que nuestros documentos de Wikipedia cubren
    TEMAS_PERMITIDOS = {
        "colombia", "colombiano", "colombiana", "colombianos", "colombianas",
        "bogotá", "bogota", "medellín", "medellin", "cartagena",
        "barranquilla", "cali", "café", "cafe",
        "biodiversidad", "fauna", "flora",
        "independencia", "boyacá", "boyaca",
        "cumbia", "vallenato", "salsa",
        "simón bolívar", "simon bolivar",
        "amazonas", "andes", "caribe",
        "peso colombiano", "gdp", "pib",
    }

    # Mensaje de rechazo cuando la pregunta es irrelevante
    MENSAJE_FUERA_DE_CONTEXTO = (
        "⚠️ Tu pregunta no está relacionada con la información que tengo "
        "en mi base de datos. Solo puedo responder preguntas sobre "
        "**Colombia** basándome en artículos de Wikipedia.\n\n"
        "Ejemplos de preguntas que puedo responder:\n"
        "- ¿Cuál es la capital de Colombia?\n"
        "- ¿Qué es el café de Colombia?\n"
        "- ¿Qué biodiversidad tiene Colombia?\n"
        "- ¿Cuándo se independizó Colombia?"
    )

    def __init__(self, config: Config, vs_manager: VectorStoreManager):
        """
        Construye el agente RAG con memoria.

        Args:
            config: Configuración del proyecto.
            vs_manager: Manager del vector store para búsquedas.
        """
        self._config = config
        self._vs_manager = vs_manager
        self._retriever = vs_manager.crear_retriever()

        # ── Crear modelo de chat (OpenAI o Ollama) ───────────
        # ModelFactory decide qué clase usar según config.provider:
        #   openai → ChatOpenAI (gpt-4o-mini, gpt-4o)
        #   ollama → ChatOllama (llama3.2, mistral, etc.)
        self._llm = ModelFactory.crear_llm(config)

        # ══════════════════════════════════════════════════════
        # MEMORIA CONVERSACIONAL — MemorySaver
        # ══════════════════════════════════════════════════════
        # MemorySaver almacena el estado del grafo EN MEMORIA (RAM).
        # Para producción, se puede reemplazar por PostgresSaver
        # (que persiste en PostgreSQL) o SqliteSaver (SQLite).
        #
        # El checkpointer se inyecta al compilar el grafo:
        #   grafo.compile(checkpointer=MemorySaver())
        #
        # Luego, al invocar el grafo, se pasa un `thread_id`:
        #   resultado = app.invoke(input, config={"configurable": {"thread_id": "abc"}})
        #
        # El MemorySaver usa el thread_id como clave para:
        #   - CARGAR el estado previo de esa conversación
        #   - GUARDAR el nuevo estado después de la invocación
        # ══════════════════════════════════════════════════════
        self._memoria = MemorySaver()

        # Compilar el grafo con la memoria como checkpointer
        self._app = self._construir_grafo()

        print("  ✅ Agente RAG compilado")
        print("  📊 Flujo: Recuperar → Validar → Responder")
        print("  🧠 Memoria: MemorySaver (checkpointer)")
        print("  🛡️  Anti-alucinación: validación por contenido + prompt")

    def preguntar(self, pregunta: str, thread_id: str = "default") -> dict:
        """
        Envía una pregunta al agente y recibe una respuesta.

        El agente:
          1. CARGA la memoria de la conversación (thread_id)
          2. Busca documentos relevantes en pgvector
          3. VERIFICA POR CÓDIGO si la pregunta es relevante
          4. Si NO es relevante → responde con mensaje de rechazo
          5. Si SÍ es relevante → genera respuesta con LLM
          6. GUARDA el nuevo estado en la memoria

        Args:
            pregunta: La pregunta del usuario.
            thread_id: Identificador de la conversación.
                       Mismo thread_id = misma memoria.
                       Diferente thread_id = conversación independiente.

        Returns:
            dict con:
              - respuesta: texto de la respuesta
              - fuentes: lista de títulos de Wikipedia usados
              - tiempo_ms: tiempo de respuesta en milisegundos
              - thread_id: ID de la conversación utilizada
        """
        # ── Configuración con thread_id ──────────────────────
        # El thread_id es la CLAVE de la memoria:
        # - Mismo thread_id → carga historial anterior
        # - Diferente thread_id → conversación nueva/separada
        config = {"configurable": {"thread_id": thread_id}}

        inicio = time.time()

        resultado = self._app.invoke(
            {"messages": [HumanMessage(content=pregunta)]},
            config=config  # ← Aquí se pasa el thread_id al MemorySaver
        )

        tiempo_ms = (time.time() - inicio) * 1000

        return {
            "respuesta": resultado["messages"][-1].content,
            "fuentes": resultado.get("fuentes", []),
            "tiempo_ms": tiempo_ms,
            "thread_id": thread_id,
        }

    def obtener_historial(self, thread_id: str) -> list[BaseMessage]:
        """
        Recupera el historial de mensajes de una conversación.

        Esto demuestra cómo la memoria persiste entre invocaciones:
        cada thread_id tiene su propio historial almacenado.

        Args:
            thread_id: ID de la conversación.

        Returns:
            Lista de mensajes (HumanMessage y AIMessage alternados).
        """
        config = {"configurable": {"thread_id": thread_id}}
        estado = self._app.get_state(config)
        return estado.values.get("messages", [])

    def contar_mensajes(self, thread_id: str) -> int:
        """
        Cuenta cuántos mensajes hay en la memoria de un thread.

        Args:
            thread_id: ID de la conversación.

        Returns:
            Número de mensajes almacenados.
        """
        return len(self.obtener_historial(thread_id))

    # ══════════════════════════════════════════════════════════
    # Construcción del grafo LangGraph
    # ══════════════════════════════════════════════════════════

    def _construir_grafo(self) -> object:
        """
        Construye y compila el grafo de LangGraph.

        Arquitectura del grafo (con validación):

          START → recuperar → ¿relevante? ─── SÍ ──→ responder → END
                  (pgvector)       │
                                   └── NO ──→ rechazar → END

        El nodo "recuperar" busca en pgvector Y valida relevancia.
        Si la pregunta NO es relevante, se va directo a "rechazar"
        sin pasar por el LLM → imposible alucinar.
        """
        grafo = StateGraph(self._EstadoRAG)

        # Registrar nodos
        grafo.add_node("recuperar", self._nodo_recuperar)
        grafo.add_node("responder", self._nodo_responder)
        grafo.add_node("rechazar", self._nodo_rechazar)

        # Flujo con bifurcación condicional
        grafo.add_edge(START, "recuperar")

        # ── Routing condicional ──────────────────────────────
        # Después de recuperar, decidimos POR CÓDIGO:
        #   - es_relevante=True  → nodo "responder" (LLM)
        #   - es_relevante=False → nodo "rechazar" (sin LLM)
        grafo.add_conditional_edges(
            "recuperar",
            self._decidir_ruta,
            {"responder": "responder", "rechazar": "rechazar"}
        )

        grafo.add_edge("responder", END)
        grafo.add_edge("rechazar", END)

        # ═══════════════════════════════════════════════════════
        # COMPILAR CON MEMORIA
        # ═══════════════════════════════════════════════════════
        return grafo.compile(checkpointer=self._memoria)

    def _decidir_ruta(self, estado: dict) -> str:
        """
        Decide si la pregunta es relevante o debe ser rechazada.

        Esta es la CLAVE de la anti-alucinación por código:
        si el nodo "recuperar" determinó que la pregunta no es
        relevante, la pregunta se rechaza SIN pasar por el LLM.

        Returns:
            "responder" si es relevante, "rechazar" si no lo es.
        """
        if estado.get("es_relevante", False):
            return "responder"
        return "rechazar"

    def _es_pregunta_relevante(self, pregunta: str, docs: list) -> bool:
        """
        Verifica POR CÓDIGO si la pregunta es sobre temas de Colombia.

        Estrategia de doble verificación:
          1. ¿La pregunta menciona un tema permitido? (Colombia, Bogotá, etc.)
          2. ¿La pregunta NO menciona un país/tema que NO sea Colombia?

        Si la pregunta es genérica (ej: "¿Cuáles son los ríos más grandes?"),
        se marca como relevante porque los documentos recuperados serán
        sobre ríos de Colombia.

        Args:
            pregunta: Texto de la pregunta del usuario.
            docs: Documentos recuperados de pgvector.

        Returns:
            True si la pregunta se considera relevante para Colombia.
        """
        pregunta_lower = pregunta.lower()

        # ── Verificación 1: ¿Menciona un tema de Colombia? ──
        # Si la pregunta menciona directamente un tema permitido,
        # es relevante sin importar qué más diga.
        for tema in self.TEMAS_PERMITIDOS:
            if tema in pregunta_lower:
                return True

        # ── Verificación 2: ¿Menciona un país/tema ajeno? ───
        # Si la pregunta menciona explícitamente otro país o tema
        # que no está en nuestra base de datos, la rechazamos.
        temas_ajenos = [
            "perú", "peru", "argentina", "brasil", "chile", "méxico",
            "mexico", "venezuela", "ecuador", "bolivia", "uruguay",
            "paraguay", "japón", "japon", "china", "estados unidos",
            "españa", "espana", "francia", "alemania", "italia",
            "pizza", "sushi", "receta", "programación", "programacion",
            "inteligencia artificial", "machine learning",
            "bitcoin", "criptomoneda", "nba", "fórmula 1",
        ]

        for tema in temas_ajenos:
            if tema in pregunta_lower:
                return False

        # ── Si no se detectó tema específico, aceptar ────────
        # Preguntas genéricas como "¿Cuáles son los ríos más grandes?"
        # se aceptan — los documentos de Colombia responderán.
        return True

    def _nodo_recuperar(self, estado: dict) -> dict:
        """
        NODO 1: Recuperar documentos de pgvector Y verificar relevancia.

        Proceso:
          1. Verifica POR CÓDIGO si la pregunta es sobre Colombia
          2. Si NO es relevante → marca es_relevante=False (sin buscar)
          3. Si SÍ es relevante → busca documentos en pgvector

        ¿Por qué funciona?
          - Verificamos la pregunta ANTES de enviar al LLM
          - Si la pregunta es sobre Perú/pizza/Japón, la rechazamos
            directamente sin darle oportunidad al LLM de alucinar
        """
        ultimo_mensaje = estado["messages"][-1].content

        # ═══════════════════════════════════════════════════════
        # VERIFICACIÓN DE RELEVANCIA — Anti-alucinación por código
        # ═══════════════════════════════════════════════════════
        docs = self._retriever.invoke(ultimo_mensaje)

        if not self._es_pregunta_relevante(ultimo_mensaje, docs):
            # ❌ Pregunta NO relacionada con Colombia
            # Rechazar directamente sin invocar el LLM
            return {
                "contexto": "",
                "fuentes": [],
                "es_relevante": False,
            }

        if not docs:
            return {"contexto": "", "fuentes": [], "es_relevante": False}

        # ✅ Pregunta relevante — preparar contexto
        contexto = "\n\n---\n\n".join([
            f"[Fuente: {doc.metadata.get('title', 'Desconocida')} | "
            f"URL: {doc.metadata.get('source_url', 'N/A')}]\n"
            f"{doc.page_content}"
            for doc in docs
        ])

        fuentes = list(set(
            doc.metadata.get("title", "Desconocida") for doc in docs
        ))

        return {
            "contexto": contexto,
            "fuentes": fuentes,
            "es_relevante": True,
        }

    def _nodo_rechazar(self, estado: dict) -> dict:
        """
        NODO DE RECHAZO: Responde sin usar el LLM.

        Este nodo se activa cuando la verificación POR CÓDIGO
        determina que la pregunta no es relevante.

        NO se invoca el LLM → imposible alucinar.
        Se genera un AIMessage con el mensaje de rechazo estático.
        """
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=self.MENSAJE_FUERA_DE_CONTEXTO)]
        }

    def _nodo_responder(self, estado: dict) -> dict:
        """
        NODO 2: Generar respuesta con el LLM.

        Este nodo SOLO se ejecuta si la pregunta pasó la
        verificación de relevancia por código.

        Recibe:
          - El contexto de pgvector (documentos relevantes)
          - El historial COMPLETO de mensajes (gracias a la memoria)

        La memoria funciona así:
          - `estado["messages"]` contiene TODOS los mensajes
            acumulados de la conversación (gracias a add_messages)
          - Esto incluye mensajes de invocaciones ANTERIORES
            del mismo thread_id
          - El LLM ve toda la conversación y puede responder
            preguntas de seguimiento
        """
        contexto = estado.get("contexto", "")
        fuentes = estado.get("fuentes", [])

        # ── Prompt para el LLM ───────────────────────────────
        # Simplificado para que funcione bien con modelos pequeños.
        # La anti-alucinación principal ya se hizo POR CÓDIGO
        # en el nodo "recuperar" (verificación de scores).
        system_content = f"""Eres un asistente que responde SOLO usando el contexto de abajo.
Responde en español. Si el contexto no tiene la respuesta, di "No tengo esa información".
NO inventes nada. Usa SOLO lo que está en el contexto.

CONTEXTO:
{contexto if contexto else "No hay información disponible."}

FUENTES: {', '.join(fuentes) if fuentes else "Ninguna"}"""

        # ── Construir mensajes para el LLM ───────────────────
        # estado["messages"] contiene TODO el historial gracias
        # al reducer `add_messages` + el MemorySaver.
        # El LLM ve: SystemMsg + todos los Human/AI anteriores + nuevo Human
        mensajes = [SystemMessage(content=system_content)] + estado["messages"]
        respuesta = self._llm.invoke(mensajes)

        return {"messages": [respuesta]}
