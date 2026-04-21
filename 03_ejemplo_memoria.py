"""
============================================================
EJEMPLO: Memoria Conversacional con LangGraph
============================================================
Este script muestra cómo configurar MEMORIA en un agente
para que recuerde el contexto de conversaciones anteriores.

Conceptos cubiertos:
  1. MemorySaver (Checkpointer) — persiste el estado entre invocaciones
  2. thread_id — identificador de conversación para aislar memorias
  3. Conversación continua — el agente recuerda lo que le dijiste antes
  4. Múltiples hilos — conversaciones independientes en paralelo
  5. Memoria con herramientas — el agente recuerda y actúa
============================================================
"""

import os
import uuid
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

# ── Cargar variables de entorno ───────────────────────────────
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: No se encontró OPENAI_API_KEY en el archivo .env")
    exit(1)

# ── Importaciones ─────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

# ══════════════════════════════════════════════════════════════
# ¿QUÉ ES LA MEMORIA EN LANGGRAPH?
# ══════════════════════════════════════════════════════════════
#
# En LangGraph, la "memoria" se implementa mediante un
# CHECKPOINTER que guarda el estado del grafo después de
# cada ejecución.
#
# MemorySaver guarda en memoria RAM (ideal para desarrollo).
# En producción se usaría SqliteSaver, PostgresSaver, etc.
#
# Cada conversación se identifica con un "thread_id" único.
# El mismo thread_id = misma conversación = misma memoria.
# Diferente thread_id = conversación nueva = sin memoria.
#
# ══════════════════════════════════════════════════════════════

# Importamos el checkpointer de memoria
from langgraph.checkpoint.memory import MemorySaver

print("=" * 60)
print("🧠 EJEMPLO: Memoria Conversacional con LangGraph")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# PARTE 1: Memoria Básica con create_react_agent
# ══════════════════════════════════════════════════════════════
# La forma más sencilla de agregar memoria es pasar un
# checkpointer al compilar el agente.
# ══════════════════════════════════════════════════════════════

def ejemplo_memoria_basica():
    """Demuestra memoria básica con un agente ReAct."""
    print("\n📌 PARTE 1: Memoria Básica con Agente ReAct")
    print("-" * 45)

    # Definimos una herramienta para que el agente pueda usarla
    @tool
    def recordar_nota(nota: str) -> str:
        """Guarda una nota importante para recordar después.
        Útil para almacenar datos del usuario como nombre, preferencias, etc.
        """
        return f"✅ Nota guardada: '{nota}'"

    # ── Paso 1: Crear el modelo ──────────────────────────────
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # ── Paso 2: Crear el checkpointer (MemorySaver) ──────────
    # MemorySaver persiste el estado en memoria RAM.
    # Cada vez que el agente procesa un mensaje, el estado
    # se guarda automáticamente bajo el thread_id.
    memoria = MemorySaver()

    # ── Paso 3: Crear el agente CON checkpointer ─────────────
    # Al pasar checkpointer=memoria, el agente recordará
    # todos los mensajes previos en el mismo thread.
    agente = create_react_agent(
        modelo,
        [recordar_nota],
        checkpointer=memoria
    )

    # ── Paso 4: Definir un thread_id para la conversación ────
    # El thread_id es como un "ID de sesión".
    # Todos los mensajes con el mismo thread_id comparten memoria.
    config_conversacion = {
        "configurable": {
            "thread_id": "conversacion-1"  # ID único de esta conversación
        }
    }

    # ── Paso 5: Primera interacción ──────────────────────────
    print("\n💬 Mensaje 1: 'Hola, me llamo Sebastián y soy ingeniero de software'")
    resultado1 = agente.invoke(
        {"messages": [HumanMessage(content="Hola, me llamo Sebastián y soy ingeniero de software")]},
        config=config_conversacion  # ← mismo thread_id
    )
    print(f"🤖 Respuesta: {resultado1['messages'][-1].content}")

    # ── Paso 6: Segunda interacción (RECUERDA el nombre) ─────
    print("\n💬 Mensaje 2: '¿Cómo me llamo y a qué me dedico?'")
    resultado2 = agente.invoke(
        {"messages": [HumanMessage(content="¿Cómo me llamo y a qué me dedico?")]},
        config=config_conversacion  # ← MISMO thread_id → RECUERDA
    )
    print(f"🤖 Respuesta: {resultado2['messages'][-1].content}")

    # ── Paso 7: Tercera interacción (acumula contexto) ───────
    print("\n💬 Mensaje 3: 'Mi lenguaje favorito es Python. Recuérdalo.'")
    resultado3 = agente.invoke(
        {"messages": [HumanMessage(content="Mi lenguaje favorito es Python. Recuérdalo.")]},
        config=config_conversacion  # ← MISMO thread_id
    )
    print(f"🤖 Respuesta: {resultado3['messages'][-1].content}")

    # ── Paso 8: Verificar que recuerda TODO ──────────────────
    print("\n💬 Mensaje 4: 'Resume todo lo que sabes sobre mí'")
    resultado4 = agente.invoke(
        {"messages": [HumanMessage(content="Resume todo lo que sabes sobre mí")]},
        config=config_conversacion
    )
    print(f"🤖 Respuesta: {resultado4['messages'][-1].content}")

    # Mostramos cuántos mensajes hay en la memoria
    total_msgs = len(resultado4['messages'])
    print(f"\n📊 Total de mensajes en memoria: {total_msgs}")

    return resultado4


# ══════════════════════════════════════════════════════════════
# PARTE 2: Múltiples Hilos (Conversaciones Independientes)
# ══════════════════════════════════════════════════════════════
# Cada thread_id tiene su propia memoria aislada.
# Esto permite manejar múltiples usuarios simultáneamente.
# ══════════════════════════════════════════════════════════════

def ejemplo_multiples_hilos():
    """Demuestra hilos independientes con diferentes memorias."""
    print("\n📌 PARTE 2: Múltiples Hilos (Conversaciones Independientes)")
    print("-" * 55)

    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    memoria = MemorySaver()

    agente = create_react_agent(modelo, [], checkpointer=memoria)

    # ── Hilo 1: Usuario "Sebastián" ──────────────────────────
    config_hilo1 = {"configurable": {"thread_id": "usuario-sebastian"}}

    print("\n🔵 Hilo 1 (Sebastián):")
    r1 = agente.invoke(
        {"messages": [HumanMessage(content="Hola, soy Sebastián. Me gusta Python.")]},
        config=config_hilo1
    )
    print(f"  🤖 {r1['messages'][-1].content}")

    # ── Hilo 2: Usuario "María" ──────────────────────────────
    config_hilo2 = {"configurable": {"thread_id": "usuario-maria"}}

    print("\n🟢 Hilo 2 (María):")
    r2 = agente.invoke(
        {"messages": [HumanMessage(content="Hola, soy María. Me encanta JavaScript.")]},
        config=config_hilo2
    )
    print(f"  🤖 {r2['messages'][-1].content}")

    # ── Verificar aislamiento: preguntar al hilo 1 ───────────
    print("\n🔵 Hilo 1 (preguntando): '¿Cómo me llamo?'")
    r3 = agente.invoke(
        {"messages": [HumanMessage(content="¿Cómo me llamo y cuál es mi lenguaje favorito?")]},
        config=config_hilo1  # ← vuelve al hilo de Sebastián
    )
    print(f"  🤖 {r3['messages'][-1].content}")

    # ── Verificar aislamiento: preguntar al hilo 2 ───────────
    print("\n🟢 Hilo 2 (preguntando): '¿Cómo me llamo?'")
    r4 = agente.invoke(
        {"messages": [HumanMessage(content="¿Cómo me llamo y cuál es mi lenguaje favorito?")]},
        config=config_hilo2  # ← vuelve al hilo de María
    )
    print(f"  🤖 {r4['messages'][-1].content}")

    print("\n✅ Cada hilo mantiene su propia memoria aislada.")


# ══════════════════════════════════════════════════════════════
# PARTE 3: StateGraph Manual con Memoria
# ══════════════════════════════════════════════════════════════
# Configuramos memoria en un grafo personalizado usando
# MemorySaver como checkpointer en .compile().
# ══════════════════════════════════════════════════════════════

def ejemplo_grafo_con_memoria():
    """Demuestra memoria en un grafo personalizado."""
    print("\n📌 PARTE 3: StateGraph Manual con Memoria")
    print("-" * 45)

    # ── Estado ────────────────────────────────────────────────
    class EstadoConversacion(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # ── Modelo con system prompt personalizado ────────────────
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # ── Nodo del asistente ────────────────────────────────────
    def nodo_asistente(estado: EstadoConversacion):
        """
        El asistente recibe TODO el historial de mensajes
        (incluyendo los guardados por el checkpointer).
        El system prompt le indica que recuerde datos del usuario.
        """
        system_msg = SystemMessage(content="""Eres un asistente personal con memoria.
Tu trabajo es recordar TODO lo que el usuario te dice: su nombre, 
preferencias, datos importantes, tareas pendientes, etc.

Reglas:
- Siempre saluda al usuario por su nombre si lo conoces
- Menciona datos previos cuando sea relevante
- Si el usuario te dice algo nuevo, confírmale que lo recordarás
- Responde de forma amigable y concisa""")

        # El historial completo viene del estado (guardado por memoria)
        mensajes = [system_msg] + estado["messages"]
        respuesta = modelo.invoke(mensajes)
        return {"messages": [respuesta]}

    # ── Construir el grafo ────────────────────────────────────
    grafo = StateGraph(EstadoConversacion)
    grafo.add_node("asistente", nodo_asistente)
    grafo.add_edge(START, "asistente")
    grafo.add_edge("asistente", END)

    # ── Compilar CON checkpointer ─────────────────────────────
    # El MemorySaver guarda el estado después de cada .invoke()
    memoria = MemorySaver()
    app = grafo.compile(checkpointer=memoria)

    # ── Función helper para chatear ───────────────────────────
    config = {"configurable": {"thread_id": "mi-sesion"}}

    def chatear(mensaje: str):
        resultado = app.invoke(
            {"messages": [HumanMessage(content=mensaje)]},
            config=config
        )
        respuesta = resultado["messages"][-1].content
        print(f"  👤 {mensaje}")
        print(f"  🤖 {respuesta}\n")
        return resultado

    # ── Conversación con memoria ──────────────────────────────
    print("\n--- Conversación con memoria ---\n")

    chatear("Hola, me llamo Sebastián. Tengo 30 años.")
    chatear("Trabajo como ingeniero de software en Google.")
    chatear("Mi comida favorita es la pizza y me gusta jugar fútbol.")
    chatear("Dame un resumen completo de todo lo que sabes de mí.")

    # ── Ver el historial completo almacenado ──────────────────
    print("📋 Historial completo en memoria:")
    estado_guardado = app.get_state(config)
    for i, msg in enumerate(estado_guardado.values["messages"]):
        tipo = "👤" if isinstance(msg, HumanMessage) else "🤖"
        contenido = msg.content[:80]
        print(f"  {tipo} [{i}] {contenido}...")

    total = len(estado_guardado.values["messages"])
    print(f"\n📊 Total mensajes almacenados: {total}")


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        ejemplo_memoria_basica()
        ejemplo_multiples_hilos()
        ejemplo_grafo_con_memoria()

        print("\n" + "=" * 60)
        print("✅ RESUMEN DE CONCEPTOS DE MEMORIA")
        print("=" * 60)
        print("""
  1. MemorySaver        → Checkpointer que guarda estado en RAM
  2. thread_id          → Identificador único de conversación
  3. checkpointer=      → Parámetro en .compile() o create_react_agent()
  4. config=            → {"configurable": {"thread_id": "mi-id"}}
  5. Aislamiento        → Cada thread_id tiene memoria independiente
  6. get_state()        → Inspeccionar el estado almacenado

  💡 En producción:
     - MemorySaver     → desarrollo (RAM, se pierde al reiniciar)
     - SqliteSaver     → persistencia local
     - PostgresSaver   → producción distribuida
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
