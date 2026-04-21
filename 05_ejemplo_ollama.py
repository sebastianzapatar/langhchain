#!/usr/bin/env python3
"""
============================================================
EJEMPLO SIMPLE: LangChain + Ollama (100% Local)
============================================================
Este script muestra cómo usar LangChain con Ollama para
ejecutar modelos de IA localmente, SIN necesidad de API
keys ni conexión a internet.

Conceptos cubiertos:
  1. Invocación directa de ChatOllama
  2. PromptTemplates con modelos locales
  3. Cadenas LCEL (pipe operator |)
  4. Chat interactivo local
  5. Streaming de respuestas (token por token)

Requisitos:
  - Ollama corriendo: docker compose up -d
    (o instalado localmente: https://ollama.ai)
  - Modelo descargado: ollama pull llama3.2:1b

Ejecutar:
  python 05_ejemplo_ollama.py
============================================================
"""

# ═══════════════════════════════════════════════════════════
# DIFERENCIA CLAVE: OpenAI vs Ollama
# ═══════════════════════════════════════════════════════════
#
#  ┌────────────────────────────────────────────────────────┐
#  │  OpenAI (cloud)          │  Ollama (local)            │
#  ├──────────────────────────┼────────────────────────────┤
#  │  from langchain_openai   │  from langchain_ollama     │
#  │    import ChatOpenAI     │    import ChatOllama       │
#  │                          │                            │
#  │  llm = ChatOpenAI(       │  llm = ChatOllama(         │
#  │    model="gpt-4o-mini",  │    model="llama3.2:1b",    │
#  │    temperature=0.7       │    temperature=0.7,        │
#  │  )                       │    base_url="http://       │
#  │                          │      localhost:11434"      │
#  │  # Requiere API key      │  )                        │
#  │  # Tiene costo por token │  # Gratis, privado        │
#  │  # Mejor calidad         │  # Sin internet           │
#  └──────────────────────────┴────────────────────────────┘
#
#  El resto del código (prompts, cadenas, parsers) es
#  EXACTAMENTE IGUAL. LangChain abstrae el proveedor.
# ═══════════════════════════════════════════════════════════

# ── Importaciones ─────────────────────────────────────────
# NOTA: No necesitamos load_dotenv() ni API keys
# Ollama corre localmente, no requiere autenticación
from langchain_ollama import ChatOllama                    # ← En vez de ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate      # Mismo que con OpenAI
from langchain_core.output_parsers import StrOutputParser  # Mismo que con OpenAI


# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN DE OLLAMA
# ═══════════════════════════════════════════════════════════
# Ollama corre un servidor HTTP local en el puerto 11434.
# Los modelos se descargan una vez y quedan en disco.
#
# Para ver los modelos disponibles:
#   ollama list
#   (o: docker exec rag-ollama ollama list)
#
# Para descargar un modelo nuevo:
#   ollama pull llama3.2:3b
#   ollama pull mistral:7b
# ═══════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434"
MODELO = "llama3.2:1b"   # Modelo pequeño, rápido (~1.3 GB)

print("=" * 60)
print("🏠 EJEMPLO SIMPLE: LangChain + Ollama (Local)")
print("=" * 60)
print(f"  Servidor: {OLLAMA_URL}")
print(f"  Modelo:   {MODELO}")
print(f"  💰 Costo: $0.00 (gratis, local)")
print()


# ══════════════════════════════════════════════════════════════
# PARTE 1: Invocación Directa del Modelo
# ══════════════════════════════════════════════════════════════
print("📌 PARTE 1: Invocación directa del modelo")
print("-" * 40)

# Inicializamos el modelo de Ollama
# Comparación con OpenAI:
#   OpenAI:  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
#   Ollama:  llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
llm = ChatOllama(
    model=MODELO,
    temperature=0.7,
    base_url=OLLAMA_URL,  # Solo necesario si no es localhost:11434
)

# .invoke() funciona EXACTAMENTE IGUAL que con OpenAI
# Envía el prompt al modelo local y espera la respuesta
respuesta = llm.invoke("¿Qué es LangChain en una frase?")

# La respuesta es un AIMessage, igual que con OpenAI
print(f"🤖 Respuesta: {respuesta.content}")


# ══════════════════════════════════════════════════════════════
# PARTE 2: Usando PromptTemplates
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 2: Usando PromptTemplates")
print("-" * 40)

# Los PromptTemplates son IDÉNTICOS entre OpenAI y Ollama
# LangChain abstrae el proveedor — el prompt no cambia
prompt = ChatPromptTemplate.from_messages([
    # system: define el ROL del asistente
    ("system", "Eres un profesor experto en {materia}. Responde en máximo 3 oraciones."),
    # human: la pregunta del usuario
    ("human", "{pregunta}")
])

# Formateamos el prompt con valores
mensaje_formateado = prompt.format_messages(
    materia="programación",
    pregunta="¿Qué es una API REST?"
)

# Invocamos el modelo local — mismo método que OpenAI
respuesta = llm.invoke(mensaje_formateado)
print(f"🤖 Respuesta: {respuesta.content}")


# ══════════════════════════════════════════════════════════════
# PARTE 3: Cadenas con LCEL (pipe operator |)
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 3: Cadenas con LCEL (pipe operator |)")
print("-" * 40)

# LCEL funciona EXACTAMENTE IGUAL con Ollama
# El flujo es: Prompt → Modelo Local → Parser
#
# Con OpenAI:  cadena = prompt | ChatOpenAI() | parser
# Con Ollama:  cadena = prompt | ChatOllama() | parser
#
# ¡El cambio es SOLO el modelo, todo lo demás es idéntico!

prompt_explicar = ChatPromptTemplate.from_template(
    "Explica el concepto de '{concepto}' como si le hablaras a un "
    "estudiante de primer semestre de ingeniería. Máximo 3 oraciones."
)

parser = StrOutputParser()

# Cadena LCEL: prompt → modelo local → parser
cadena = prompt_explicar | llm | parser

resultado = cadena.invoke({"concepto": "bases de datos vectoriales"})
print(f"🤖 Respuesta: {resultado}")


# ══════════════════════════════════════════════════════════════
# PARTE 4: Streaming de respuestas (token por token)
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 4: Streaming (respuesta token por token)")
print("-" * 40)

# Una ventaja de los modelos locales: streaming sin latencia de red.
# .stream() envía la respuesta token por token en tiempo real.
#
# Esto es útil para:
#   - Mostrar la respuesta mientras se genera
#   - Dar sensación de velocidad al usuario
#   - No esperar a que termine toda la generación

prompt_stream = ChatPromptTemplate.from_template(
    "Escribe 3 datos curiosos sobre {tema}. Sé breve."
)

cadena_stream = prompt_stream | llm | parser

print("🤖 Respuesta (streaming): ", end="", flush=True)

# .stream() retorna un generador que produce tokens uno a uno
for token in cadena_stream.stream({"tema": "Colombia"}):
    print(token, end="", flush=True)

print()  # Salto de línea al final


# ══════════════════════════════════════════════════════════════
# PARTE 5: Chat interactivo
# ══════════════════════════════════════════════════════════════
print("\n📌 PARTE 5: Chat interactivo con Ollama")
print("-" * 40)

# Creamos un chat simple donde puedes hacer preguntas
# al modelo local. Sin API key, sin costo, sin internet.

prompt_chat = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente amigable que responde en español. "
     "Sé conciso: máximo 3-4 oraciones por respuesta."),
    ("human", "{pregunta}")
])

cadena_chat = prompt_chat | llm | parser

print("  💬 Escribe una pregunta (o 'salir' para terminar)")
print()

while True:
    try:
        pregunta = input("  👤 Tú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  👋 ¡Hasta luego!")
        break

    if not pregunta:
        continue

    if pregunta.lower() == "salir":
        print("\n  👋 ¡Hasta luego!")
        break

    # Streaming de la respuesta
    print("  🤖 Ollama: ", end="", flush=True)
    for token in cadena_chat.stream({"pregunta": pregunta}):
        print(token, end="", flush=True)
    print("\n")


# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("✅ RESUMEN: OpenAI vs Ollama")
print("=" * 60)
print("""
  Lo que CAMBIA:
    - Import: langchain_openai → langchain_ollama
    - Clase:  ChatOpenAI       → ChatOllama
    - Config: API key          → base_url (localhost)

  Lo que NO cambia (¡todo lo demás!):
    - PromptTemplates
    - Cadenas LCEL (pipe operator |)
    - OutputParsers
    - .invoke(), .stream(), .batch()
    - Toda la lógica de la aplicación

  ┌─────────────────────────────────────────────────┐
  │  LangChain abstrae el proveedor del modelo.     │
  │  Cambia UNA línea y toda tu app funciona        │
  │  con OpenAI, Ollama, Anthropic, Google, etc.    │
  └─────────────────────────────────────────────────┘

  💡 Para usar otro modelo, cambia MODELO al inicio:
     MODELO = "llama3.2:3b"    # Mejor calidad
     MODELO = "mistral:7b"     # Bueno en español
     MODELO = "gemma2:9b"      # Excelente calidad
""")
