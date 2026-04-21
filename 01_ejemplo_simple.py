"""
============================================================
EJEMPLO SIMPLE: LangChain + OpenAI
============================================================
Este script muestra los fundamentos de LangChain conectado
con OpenAI usando el patrón LCEL (LangChain Expression Language).

Conceptos cubiertos:
  1. Invocación directa del modelo (ChatOpenAI)
  2. Uso de PromptTemplates para estructurar mensajes
  3. Cadenas (Chains) con el operador pipe (|)
  4. OutputParsers para formatear la respuesta
============================================================
"""

import os
from dotenv import load_dotenv

# ── 1. Cargamos las variables de entorno (.env) ──────────────
# Esto busca un archivo .env en el directorio actual
# y carga OPENAI_API_KEY automáticamente
load_dotenv()

# Verificamos que la API key esté configurada
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: No se encontró OPENAI_API_KEY en el archivo .env")
    print("   Copia .env.example a .env y agrega tu API key")
    exit(1)

# ── 2. Importamos los componentes de LangChain ───────────────
from langchain_openai import ChatOpenAI                    # Modelo de OpenAI
from langchain_core.prompts import ChatPromptTemplate      # Plantillas de prompts
from langchain_core.output_parsers import StrOutputParser  # Parser de salida a string

print("=" * 60)
print("🔗 EJEMPLO SIMPLE: LangChain + OpenAI")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# PARTE 1: Invocación Directa del Modelo
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 1: Invocación directa del modelo")
print("-" * 40)

# Inicializamos el modelo de OpenAI
# - model: el modelo de OpenAI a utilizar (gpt-4o-mini es económico y rápido)
# - temperature: controla la creatividad (0 = determinista, 1 = muy creativo)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Invocamos el modelo directamente con un string
# .invoke() envía el prompt y espera la respuesta completa
respuesta = llm.invoke("¿Qué es LangChain en una frase?")

# La respuesta es un objeto AIMessage, accedemos al contenido con .content
print(f"🤖 Respuesta: {respuesta.content}")


# ══════════════════════════════════════════════════════════════
# PARTE 2: Usando PromptTemplates
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 2: Usando PromptTemplates")
print("-" * 40)

# Los PromptTemplates permiten crear prompts reutilizables con variables
# Las variables se definen con {nombre_variable}
prompt = ChatPromptTemplate.from_messages([
    # system: define el rol/personalidad del asistente
    ("system", "Eres un profesor experto en {materia}. Explica de forma clara y concisa."),
    # human: el mensaje del usuario
    ("human", "{pregunta}")
])

# Formateamos el prompt con valores específicos
mensaje_formateado = prompt.format_messages(
    materia="inteligencia artificial",
    pregunta="¿Qué es un transformer?"
)

# Invocamos el modelo con el prompt formateado
respuesta = llm.invoke(mensaje_formateado)
print(f"🤖 Respuesta: {respuesta.content}")


# ══════════════════════════════════════════════════════════════
# PARTE 3: Cadenas con LCEL (LangChain Expression Language)
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 3: Cadenas con LCEL (pipe operator |)")
print("-" * 40)

# LCEL usa el operador | (pipe) para conectar componentes
# El flujo es: Prompt → Modelo → Parser
#
# Esto crea un pipeline reutilizable:
#   1. El prompt formatea la entrada
#   2. El modelo genera la respuesta
#   3. El parser extrae solo el texto (string)

# Definimos el prompt
prompt_explicar = ChatPromptTemplate.from_template(
    "Explica el concepto de '{concepto}' como si le hablaras a un "
    "estudiante de primer semestre de ingeniería. Máximo 3 oraciones."
)

# Definimos el parser (convierte AIMessage → string)
parser = StrOutputParser()

# Construimos la cadena con el operador pipe
# prompt_explicar → llm → parser
cadena = prompt_explicar | llm | parser

# Invocamos la cadena pasando las variables del prompt
resultado = cadena.invoke({"concepto": "redes neuronales"})

# El resultado ya es un string limpio gracias al parser
print(f"🤖 Respuesta: {resultado}")


# ══════════════════════════════════════════════════════════════
# PARTE 4: Cadena con múltiples pasos
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 4: Cadena con múltiples pasos")
print("-" * 40)

# Podemos encadenar múltiples operaciones
# Aquí creamos una cadena que:
#   1. Genera una explicación
#   2. Luego crea un ejemplo práctico basado en esa explicación

# Primera cadena: generar explicación
prompt_paso1 = ChatPromptTemplate.from_template(
    "Explica brevemente qué es {tema}. Máximo 2 oraciones."
)
cadena_paso1 = prompt_paso1 | llm | parser

# Segunda cadena: generar ejemplo basado en la explicación anterior
prompt_paso2 = ChatPromptTemplate.from_template(
    "Basándote en esta explicación: '{explicacion}'\n\n"
    "Genera un ejemplo práctico y sencillo en Python. Solo el código, sin explicación."
)
cadena_paso2 = prompt_paso2 | llm | parser

# Ejecutamos ambas cadenas secuencialmente
print("Paso 1 - Explicación:")
explicacion = cadena_paso1.invoke({"tema": "una función lambda en Python"})
print(f"  📝 {explicacion}")

print("\nPaso 2 - Ejemplo práctico:")
ejemplo = cadena_paso2.invoke({"explicacion": explicacion})
print(f"  💻 {ejemplo}")


# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("✅ RESUMEN DE CONCEPTOS")
print("=" * 60)
print("""
  1. ChatOpenAI      → Conexión con modelos de OpenAI
  2. PromptTemplate  → Plantillas reutilizables de prompts
  3. LCEL (|)        → Encadenar componentes con el pipe operator
  4. OutputParser    → Formatear la salida del modelo
  5. .invoke()       → Ejecutar una cadena o modelo

  💡 Diagrama del flujo LCEL:
     Input → [Prompt] → [LLM] → [Parser] → Output
""")
