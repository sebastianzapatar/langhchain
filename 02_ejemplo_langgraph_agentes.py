"""
============================================================
EJEMPLO AVANZADO: LangGraph + Agentes con Herramientas
============================================================
Este script muestra cómo construir un sistema multi-agente
usando LangGraph para orquestar el flujo de trabajo.

Conceptos cubiertos:
  1. Definición de herramientas personalizadas (@tool)
  2. Agente ReAct prebuilt (create_react_agent)
  3. Construcción manual de un grafo con StateGraph
  4. Sistema multi-agente con nodos especializados
  5. Enrutamiento condicional entre agentes
============================================================
"""

import os
import json
from datetime import datetime
from typing import Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv

# ── Cargamos las variables de entorno ─────────────────────────
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: No se encontró OPENAI_API_KEY en el archivo .env")
    exit(1)

# ── Importaciones de LangChain ────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

# ── Importaciones de LangGraph ────────────────────────────────
from langgraph.graph import StateGraph, START, END        # Grafo de estados
from langgraph.graph.message import add_messages           # Reducer para mensajes
from langgraph.prebuilt import ToolNode, create_react_agent  # Componentes prebuilt

print("=" * 60)
print("🧠 EJEMPLO AVANZADO: LangGraph + Agentes")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# PARTE 1: Definición de Herramientas (Tools)
# ══════════════════════════════════════════════════════════════
# Las herramientas son funciones que el agente puede invocar.
# Usamos el decorador @tool para registrarlas automáticamente.
# LangChain extrae el nombre, descripción y parámetros del
# docstring y type hints de la función.
# ══════════════════════════════════════════════════════════════

@tool
def calcular_operacion(expresion: str) -> str:
    """Calcula una operación matemática. Recibe una expresión como '2 + 3 * 4'.
    Soporta: suma (+), resta (-), multiplicación (*), división (/), potencia (**).
    """
    try:
        # Evaluamos la expresión de forma segura
        # NOTA: En producción, usar una librería de parsing segura
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
    # Simulamos una base de conocimiento
    conocimiento = {
        "python": "Python es un lenguaje de programación de alto nivel, interpretado y multiparadigma. "
                  "Fue creado por Guido van Rossum en 1991. Es popular en IA, ciencia de datos y web.",
        "langchain": "LangChain es un framework para construir aplicaciones con LLMs. "
                     "Permite encadenar prompts, modelos, herramientas y memoria de forma modular.",
        "langgraph": "LangGraph es una extensión de LangChain para construir flujos de trabajo "
                     "con agentes usando grafos de estados. Permite crear sistemas multi-agente.",
        "ia": "La Inteligencia Artificial es una rama de la ciencia de la computación que busca "
              "crear sistemas capaces de realizar tareas que requieren inteligencia humana.",
    }

    # Buscamos por coincidencia parcial en las claves
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
    # Simulación simple de análisis de sentimiento
    palabras_positivas = ["bueno", "excelente", "genial", "increíble", "feliz", "amor", "éxito", "bien", "mejor"]
    palabras_negativas = ["malo", "terrible", "horrible", "triste", "odio", "fracaso", "peor", "error", "problema"]

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


# Lista de todas las herramientas disponibles
todas_las_herramientas = [calcular_operacion, obtener_fecha_actual, buscar_informacion, analizar_sentimiento]


# ══════════════════════════════════════════════════════════════
# PARTE 2: Agente Simple con create_react_agent (Prebuilt)
# ══════════════════════════════════════════════════════════════
# create_react_agent crea un agente ReAct listo para usar.
# ReAct = Reasoning + Acting: el modelo razona sobre qué
# herramienta usar y luego la ejecuta.
# ══════════════════════════════════════════════════════════════

def ejemplo_agente_react():
    """Demuestra el uso del agente ReAct prebuilt."""
    print("\n📌 PARTE 2: Agente ReAct (Prebuilt)")
    print("-" * 40)

    # Inicializamos el modelo
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Creamos el agente ReAct con las herramientas
    # create_react_agent maneja automáticamente:
    #   - El bucle de razonamiento
    #   - La ejecución de herramientas
    #   - La gestión del estado (mensajes)
    agente = create_react_agent(modelo, todas_las_herramientas)

    # Invocamos el agente con una pregunta
    # El agente decidirá qué herramienta(s) usar
    print("\n🔍 Pregunta: '¿Cuánto es 15 * 7 + 23? Y también dime la fecha actual'")
    resultado = agente.invoke({
        "messages": [HumanMessage(content="¿Cuánto es 15 * 7 + 23? Y también dime la fecha actual")]
    })

    # Mostramos la respuesta final del agente
    respuesta_final = resultado["messages"][-1].content
    print(f"🤖 Respuesta: {respuesta_final}")

    # Mostramos el rastro de todas las acciones del agente
    print("\n📋 Rastro de acciones del agente:")
    for i, msg in enumerate(resultado["messages"]):
        tipo = type(msg).__name__
        contenido = msg.content[:100] if msg.content else "(sin contenido)"
        print(f"  [{i}] {tipo}: {contenido}")

    return resultado


# ══════════════════════════════════════════════════════════════
# PARTE 3: Grafo Personalizado con StateGraph
# ══════════════════════════════════════════════════════════════
# Construimos un grafo manualmente para tener control total
# sobre el flujo de ejecución del agente.
#
# Estructura del grafo:
#   START → agente → {herramientas | END}
#            ↑           ↓
#            └───────────┘
# ══════════════════════════════════════════════════════════════

def ejemplo_grafo_personalizado():
    """Demuestra la construcción manual de un grafo con StateGraph."""
    print("\n📌 PARTE 3: Grafo Personalizado (StateGraph)")
    print("-" * 40)

    # ── Paso 1: Definir el Estado ─────────────────────────────
    # El estado es un TypedDict que contiene los datos compartidos
    # entre todos los nodos del grafo.
    # Annotated[..., add_messages] indica que los mensajes nuevos
    # se AGREGAN a la lista (no la reemplazan).
    class EstadoAgente(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # ── Paso 2: Inicializar el modelo con herramientas ────────
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # bind_tools() le dice al modelo qué herramientas tiene disponibles
    # El modelo incluirá "tool_calls" en su respuesta cuando quiera usar una
    modelo_con_herramientas = modelo.bind_tools(todas_las_herramientas)

    # ── Paso 3: Definir los Nodos (funciones) ─────────────────
    def nodo_agente(estado: EstadoAgente):
        """
        Nodo principal del agente.
        Recibe el estado actual, llama al modelo y retorna
        el nuevo mensaje generado.
        """
        print("  🧠 Nodo Agente: Procesando...")
        mensajes = estado["messages"]
        respuesta = modelo_con_herramientas.invoke(mensajes)
        # Retornamos el mensaje; LangGraph lo agrega al estado automáticamente
        return {"messages": [respuesta]}

    # ToolNode es un nodo prebuilt que ejecuta las herramientas
    # automáticamente cuando encuentra tool_calls en el último mensaje
    nodo_herramientas = ToolNode(todas_las_herramientas)

    # ── Paso 4: Definir la función de enrutamiento ────────────
    def decidir_siguiente_paso(estado: EstadoAgente) -> Literal["herramientas", END]:
        """
        Función de enrutamiento condicional.
        Examina el último mensaje del agente:
          - Si tiene tool_calls → ir al nodo de herramientas
          - Si no → el agente terminó, ir a END
        """
        ultimo_mensaje = estado["messages"][-1]

        # Verificamos si el modelo quiere llamar a alguna herramienta
        if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
            print("  🔧 Decisión: Usar herramientas")
            return "herramientas"
        else:
            print("  ✅ Decisión: Respuesta final")
            return END

    # ── Paso 5: Construir el Grafo ────────────────────────────
    # El grafo define la topología del flujo de ejecución
    grafo = StateGraph(EstadoAgente)

    # Agregamos los nodos
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("herramientas", nodo_herramientas)

    # Definimos las aristas (edges)
    # START → agente: El flujo comienza en el nodo agente
    grafo.add_edge(START, "agente")

    # agente → {herramientas | END}: Arista condicional
    grafo.add_conditional_edges("agente", decidir_siguiente_paso)

    # herramientas → agente: Después de ejecutar herramientas, volver al agente
    grafo.add_edge("herramientas", "agente")

    # ── Paso 6: Compilar y Ejecutar ──────────────────────────
    # .compile() crea un ejecutor optimizado a partir del grafo
    app = grafo.compile()

    print("\n🔍 Pregunta: 'Busca información sobre Python y analiza el sentimiento del texto: Python es excelente y genial'")
    resultado = app.invoke({
        "messages": [HumanMessage(
            content="Busca información sobre Python y luego analiza el sentimiento de este texto: 'Python es un lenguaje excelente y genial para programar'"
        )]
    })

    respuesta_final = resultado["messages"][-1].content
    print(f"\n🤖 Respuesta Final: {respuesta_final}")

    return resultado


# ══════════════════════════════════════════════════════════════
# PARTE 4: Sistema Multi-Agente con Supervisor
# ══════════════════════════════════════════════════════════════
# Creamos un sistema donde un agente "Supervisor" delega
# tareas a agentes especializados:
#   - Agente Calculadora: operaciones matemáticas
#   - Agente Investigador: búsqueda de información
#   - Agente Analista: análisis de sentimiento
#
# Flujo:
#   START → Supervisor → {Calculadora | Investigador | Analista | END}
#                ↑                      ↓
#                └──────────────────────┘
# ══════════════════════════════════════════════════════════════

def ejemplo_multi_agente():
    """Demuestra un sistema multi-agente con supervisión."""
    print("\n📌 PARTE 4: Sistema Multi-Agente con Supervisor")
    print("-" * 40)

    # ── Estado del sistema multi-agente ───────────────────────
    class EstadoMultiAgente(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        siguiente_agente: str  # Indica a qué agente delegar

    # ── Modelo base ──────────────────────────────────────────
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── Nodo Supervisor ──────────────────────────────────────
    def nodo_supervisor(estado: EstadoMultiAgente):
        """
        El Supervisor analiza la solicitud del usuario y decide
        a qué agente especializado delegar la tarea.
        """
        print("  👔 Supervisor: Analizando solicitud...")

        prompt_supervisor = ChatPromptTemplate.from_messages([
            ("system", """Eres un supervisor que gestiona un equipo de agentes especializados.
Tu trabajo es analizar la solicitud del usuario y decidir qué agente debe encargarse.

Agentes disponibles:
- "calculadora": Para operaciones matemáticas y cálculos
- "investigador": Para buscar información sobre temas
- "analista": Para analizar sentimientos o textos
- "FINALIZAR": Cuando la tarea ya está completada y puedes dar la respuesta final

Responde SOLO con el nombre del agente (una sola palabra) o FINALIZAR.
Si hay múltiples tareas, elige la primera que aún no se haya completado."""),
            ("human", "{input}")
        ])

        # Formateamos el historial de mensajes para el supervisor
        historial = "\n".join([
            f"{'Usuario' if isinstance(m, HumanMessage) else 'Sistema'}: {m.content}"
            for m in estado["messages"]
        ])

        cadena = prompt_supervisor | modelo
        respuesta = cadena.invoke({"input": historial})
        decision = respuesta.content.strip().lower()

        print(f"  👔 Supervisor decide: {decision}")

        return {
            "siguiente_agente": decision,
            "messages": [AIMessage(content=f"[Supervisor] Delegando a: {decision}")]
        }

    # ── Nodo Agente Calculadora ──────────────────────────────
    def nodo_calculadora(estado: EstadoMultiAgente):
        """Agente especializado en cálculos matemáticos."""
        print("  🔢 Agente Calculadora: Procesando...")

        modelo_calc = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        modelo_calc_tools = modelo_calc.bind_tools([calcular_operacion])

        # Extraemos la solicitud original
        pregunta_original = estado["messages"][0].content

        respuesta = modelo_calc_tools.invoke([
            SystemMessage(content="Eres un agente calculadora. Usa la herramienta calcular_operacion para resolver operaciones matemáticas."),
            HumanMessage(content=pregunta_original)
        ])

        # Si el modelo quiere usar herramientas, las ejecutamos
        if respuesta.tool_calls:
            resultados = []
            for tc in respuesta.tool_calls:
                resultado_tool = calcular_operacion.invoke(tc["args"])
                resultados.append(resultado_tool)
            contenido = f"[Calculadora] {'; '.join(resultados)}"
        else:
            contenido = f"[Calculadora] {respuesta.content}"

        return {"messages": [AIMessage(content=contenido)]}

    # ── Nodo Agente Investigador ─────────────────────────────
    def nodo_investigador(estado: EstadoMultiAgente):
        """Agente especializado en búsqueda de información."""
        print("  🔎 Agente Investigador: Buscando...")

        pregunta_original = estado["messages"][0].content

        modelo_inv = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        modelo_inv_tools = modelo_inv.bind_tools([buscar_informacion])

        respuesta = modelo_inv_tools.invoke([
            SystemMessage(content="Eres un agente investigador. Usa la herramienta buscar_informacion para encontrar datos relevantes."),
            HumanMessage(content=pregunta_original)
        ])

        if respuesta.tool_calls:
            resultados = []
            for tc in respuesta.tool_calls:
                resultado_tool = buscar_informacion.invoke(tc["args"])
                resultados.append(resultado_tool)
            contenido = f"[Investigador] {'; '.join(resultados)}"
        else:
            contenido = f"[Investigador] {respuesta.content}"

        return {"messages": [AIMessage(content=contenido)]}

    # ── Nodo Agente Analista ─────────────────────────────────
    def nodo_analista(estado: EstadoMultiAgente):
        """Agente especializado en análisis de texto."""
        print("  📊 Agente Analista: Analizando...")

        pregunta_original = estado["messages"][0].content

        modelo_an = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        modelo_an_tools = modelo_an.bind_tools([analizar_sentimiento])

        respuesta = modelo_an_tools.invoke([
            SystemMessage(content="Eres un agente analista. Usa la herramienta analizar_sentimiento para analizar textos."),
            HumanMessage(content=pregunta_original)
        ])

        if respuesta.tool_calls:
            resultados = []
            for tc in respuesta.tool_calls:
                resultado_tool = analizar_sentimiento.invoke(tc["args"])
                resultados.append(resultado_tool)
            contenido = f"[Analista] {'; '.join(resultados)}"
        else:
            contenido = f"[Analista] {respuesta.content}"

        return {"messages": [AIMessage(content=contenido)]}

    # ── Nodo de respuesta final ──────────────────────────────
    def nodo_respuesta_final(estado: EstadoMultiAgente):
        """Genera la respuesta final consolidada."""
        print("  📝 Generando respuesta final...")

        modelo_final = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        historial = "\n".join([m.content for m in estado["messages"]])

        respuesta = modelo_final.invoke([
            SystemMessage(content="Resume los resultados obtenidos por los agentes en una respuesta clara y completa para el usuario."),
            HumanMessage(content=historial)
        ])

        return {"messages": [AIMessage(content=f"[Respuesta Final] {respuesta.content}")]}

    # ── Función de enrutamiento del supervisor ────────────────
    def enrutar_agente(estado: EstadoMultiAgente) -> str:
        """Determina el siguiente nodo basándose en la decisión del supervisor."""
        siguiente = estado.get("siguiente_agente", "finalizar")

        rutas = {
            "calculadora": "calculadora",
            "investigador": "investigador",
            "analista": "analista",
        }
        return rutas.get(siguiente, "respuesta_final")

    # ── Construir el Grafo Multi-Agente ──────────────────────
    grafo = StateGraph(EstadoMultiAgente)

    # Nodos
    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("calculadora", nodo_calculadora)
    grafo.add_node("investigador", nodo_investigador)
    grafo.add_node("analista", nodo_analista)
    grafo.add_node("respuesta_final", nodo_respuesta_final)

    # Aristas
    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar_agente)

    # Después de cada agente especializado → volver al supervisor
    grafo.add_edge("calculadora", "supervisor")
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")

    # La respuesta final termina el flujo
    grafo.add_edge("respuesta_final", END)

    # Compilar
    app = grafo.compile()

    # ── Ejecutar ─────────────────────────────────────────────
    print("\n🔍 Pregunta: 'Busca información sobre LangChain'")
    resultado = app.invoke({
        "messages": [HumanMessage(content="Busca información sobre LangChain")],
        "siguiente_agente": ""
    })

    # Mostramos la respuesta final
    respuesta_final = resultado["messages"][-1].content
    print(f"\n🤖 {respuesta_final}")

    # Mostramos el flujo completo
    print("\n📋 Flujo completo de ejecución:")
    for i, msg in enumerate(resultado["messages"]):
        tipo = "👤" if isinstance(msg, HumanMessage) else "🤖"
        contenido = msg.content[:120]
        print(f"  {tipo} [{i}] {contenido}")

    return resultado


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        # Ejecutamos los tres ejemplos en orden
        ejemplo_agente_react()
        ejemplo_grafo_personalizado()
        ejemplo_multi_agente()

        print("\n" + "=" * 60)
        print("✅ RESUMEN DE CONCEPTOS AVANZADOS")
        print("=" * 60)
        print("""
  1. @tool              → Definir herramientas que el agente puede usar
  2. create_react_agent → Agente ReAct prebuilt (rápido de implementar)
  3. StateGraph         → Grafo personalizado con control total
  4. ToolNode           → Nodo prebuilt para ejecutar herramientas
  5. Conditional Edges  → Enrutamiento dinámico entre nodos
  6. Multi-Agente       → Supervisor que delega a agentes especializados

  💡 Diagrama del Sistema Multi-Agente:
     Usuario → [Supervisor] → [Calculadora]
                    ↕         → [Investigador]
                    ↕         → [Analista]
                    ↓
              [Respuesta Final]
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Verifica que tu OPENAI_API_KEY sea válida en el archivo .env")
