"""
============================================================
EJEMPLO: LangChain + Memoria + Base de Datos Vectorizada
         (PostgreSQL con pgvector)
============================================================
Este script muestra cómo conectar LangChain a una base de
datos vectorizada en PostgreSQL (pgvector) con memoria
conversacional para crear un sistema RAG completo.

Conceptos cubiertos:
  1. Embeddings — Convertir texto en vectores numéricos
  2. pgvector   — Extensión de PostgreSQL para vectores
  3. VectorStore — Almacenar y buscar documentos por similitud
  4. RAG        — Retrieval-Augmented Generation
  5. Memoria    — Conversación persistente con MemorySaver

Requisitos:
  - PostgreSQL con extensión pgvector instalada
  - pip install langchain langchain-openai langchain-postgres
  - pip install psycopg[binary] (driver PostgreSQL moderno)

Base de datos:
  docker run --name pgvector -d \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=vectordb \
    -p 5432:5432 \
    pgvector/pgvector:pg16
============================================================
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

# ── Cargar variables de entorno ───────────────────────────────
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: No se encontró OPENAI_API_KEY en el archivo .env")
    exit(1)

# ── Importaciones de LangChain ────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# LangGraph para memoria
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# PostgreSQL + pgvector
from langchain_postgres import PGVector

print("=" * 60)
print("🗄️  EJEMPLO: LangChain + PostgreSQL pgvector + Memoria")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE CONEXIÓN
# ══════════════════════════════════════════════════════════════

# Cadena de conexión a PostgreSQL
# Formato: postgresql+psycopg://usuario:contraseña@host:puerto/base_datos
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/vectordb"
)

# Nombre de la colección de vectores (como una "tabla" en la DB)
COLLECTION_NAME = "documentos_ejemplo"


# ══════════════════════════════════════════════════════════════
# ¿QUÉ SON LOS EMBEDDINGS?
# ══════════════════════════════════════════════════════════════
#
# Un embedding es una representación numérica (vector) de un
# texto. Textos similares tienen vectores cercanos en el
# espacio vectorial.
#
# Ejemplo:
#   "gato" → [0.2, 0.8, 0.1, ...]   (1536 dimensiones)
#   "perro" → [0.3, 0.7, 0.2, ...]   ← cercano a "gato"
#   "avión" → [0.9, 0.1, 0.8, ...]   ← lejano de "gato"
#
# Esto permite buscar documentos por SIGNIFICADO, no solo
# por palabras exactas.
#
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# PARTE 1: Crear Embeddings y VectorStore
# ══════════════════════════════════════════════════════════════

def crear_vectorstore():
    """
    Crea una conexión al vector store en PostgreSQL usando pgvector.
    Si la colección no existe, la crea automáticamente.
    """
    print("\n📌 Creando conexión al VectorStore (PostgreSQL + pgvector)...")

    # ── Paso 1: Crear el modelo de embeddings ────────────────
    # OpenAI text-embedding-3-small: 1536 dimensiones, rápido y económico
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Modelo de embeddings
    )

    print("  ✅ Modelo de embeddings: text-embedding-3-small (1536 dims)")

    # ── Paso 2: Conectar al VectorStore en PostgreSQL ────────
    # PGVector se conecta a PostgreSQL y crea las tablas necesarias
    # automáticamente (incluyendo la extensión pgvector si es posible).
    vectorstore = PGVector(
        embeddings=embeddings,           # Modelo para generar vectores
        collection_name=COLLECTION_NAME, # Nombre de la colección
        connection=DATABASE_URL,         # Cadena de conexión PostgreSQL
        use_jsonb=True,                  # Usar JSONB para metadata
    )

    print(f"  ✅ VectorStore conectado: {COLLECTION_NAME}")
    print(f"  📍 Base de datos: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

    return vectorstore, embeddings


# ══════════════════════════════════════════════════════════════
# PARTE 2: Ingestar Documentos
# ══════════════════════════════════════════════════════════════

def ingestar_documentos(vectorstore):
    """
    Inserta documentos de ejemplo en la base de datos vectorizada.
    Cada documento se convierte en un vector y se almacena.
    """
    print("\n📌 PARTE 2: Ingesta de Documentos")
    print("-" * 45)

    # ── Documentos de ejemplo ─────────────────────────────────
    # Cada Document tiene:
    #   - page_content: el texto del documento
    #   - metadata: información adicional (filtrable)
    documentos = [
        Document(
            page_content="Python es un lenguaje de programación de alto nivel, interpretado "
                         "y multiparadigma. Fue creado por Guido van Rossum y lanzado en 1991. "
                         "Es conocido por su sintaxis clara y legible.",
            metadata={"categoria": "lenguajes", "autor": "wikipedia", "año": 2024}
        ),
        Document(
            page_content="FastAPI es un framework moderno y de alto rendimiento para construir "
                         "APIs con Python 3.7+. Está basado en type hints estándar de Python. "
                         "Genera documentación automática con Swagger UI.",
            metadata={"categoria": "frameworks", "autor": "documentacion", "año": 2024}
        ),
        Document(
            page_content="PostgreSQL es un sistema de gestión de bases de datos relacional y "
                         "orientado a objetos. Es de código abierto y conocido por su robustez, "
                         "extensibilidad y cumplimiento de estándares SQL.",
            metadata={"categoria": "bases_datos", "autor": "wikipedia", "año": 2024}
        ),
        Document(
            page_content="pgvector es una extensión de PostgreSQL que agrega soporte para "
                         "almacenamiento y búsqueda de vectores. Permite realizar búsquedas "
                         "por similitud coseno, distancia euclidiana y producto interno.",
            metadata={"categoria": "bases_datos", "autor": "documentacion", "año": 2024}
        ),
        Document(
            page_content="LangChain es un framework de código abierto para construir aplicaciones "
                         "con modelos de lenguaje grande (LLMs). Permite encadenar componentes como "
                         "prompts, modelos, herramientas y memoria de forma modular.",
            metadata={"categoria": "ia", "autor": "documentacion", "año": 2024}
        ),
        Document(
            page_content="Los embeddings son representaciones vectoriales de texto que capturan "
                         "el significado semántico. Textos similares producen vectores cercanos "
                         "en el espacio vectorial, permitiendo búsquedas por similitud.",
            metadata={"categoria": "ia", "autor": "tutorial", "año": 2024}
        ),
        Document(
            page_content="Docker es una plataforma de contenedores que permite empaquetar "
                         "aplicaciones con todas sus dependencias en contenedores. Esto garantiza "
                         "que la aplicación se ejecute de la misma forma en cualquier entorno.",
            metadata={"categoria": "devops", "autor": "documentacion", "año": 2024}
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) es una técnica que combina "
                         "recuperación de información con generación de texto. Primero busca "
                         "documentos relevantes y luego los usa como contexto para el LLM.",
            metadata={"categoria": "ia", "autor": "tutorial", "año": 2024}
        ),
        Document(
            page_content="React es una biblioteca de JavaScript para construir interfaces de "
                         "usuario. Fue creada por Facebook y se basa en componentes reutilizables "
                         "que gestionan su propio estado.",
            metadata={"categoria": "frontend", "autor": "wikipedia", "año": 2024}
        ),
        Document(
            page_content="Kubernetes es una plataforma de orquestación de contenedores de código "
                         "abierto. Automatiza el despliegue, escalamiento y gestión de aplicaciones "
                         "containerizadas en clusters de servidores.",
            metadata={"categoria": "devops", "autor": "documentacion", "año": 2024}
        ),
    ]

    # ── Insertar documentos en la base de datos ──────────────
    print(f"\n  📄 Insertando {len(documentos)} documentos en PostgreSQL...")

    # add_documents convierte cada texto en un vector y lo almacena
    ids = vectorstore.add_documents(documentos)

    print(f"  ✅ {len(ids)} documentos insertados correctamente")
    print(f"  📊 IDs generados: {ids[:3]}...")

    # Mostrar las categorías insertadas
    categorias = set(doc.metadata["categoria"] for doc in documentos)
    print(f"  📂 Categorías: {', '.join(categorias)}")

    return ids


# ══════════════════════════════════════════════════════════════
# PARTE 3: Búsqueda por Similitud
# ══════════════════════════════════════════════════════════════

def ejemplo_busqueda_similitud(vectorstore):
    """
    Demuestra cómo buscar documentos similares a una consulta.
    La búsqueda se hace por cercanía de vectores, no por palabras exactas.
    """
    print("\n📌 PARTE 3: Búsqueda por Similitud (Similarity Search)")
    print("-" * 55)

    # ── Búsqueda 1: Consulta general ─────────────────────────
    consulta = "¿Qué es pgvector y para qué sirve?"
    print(f"\n  🔍 Consulta: \"{consulta}\"")

    # similarity_search convierte la consulta en un vector y busca
    # los documentos más cercanos en el espacio vectorial
    resultados = vectorstore.similarity_search(
        consulta,
        k=3  # Retornar los 3 más similares
    )

    print(f"  📊 Resultados encontrados: {len(resultados)}")
    for i, doc in enumerate(resultados, 1):
        print(f"\n  📄 Resultado {i}:")
        print(f"     Categoría: {doc.metadata.get('categoria', 'N/A')}")
        print(f"     Texto: {doc.page_content[:100]}...")

    # ── Búsqueda 2: Con scores de similitud ──────────────────
    print(f"\n  {'─' * 45}")
    consulta2 = "frameworks para construir APIs web"
    print(f"\n  🔍 Consulta con scores: \"{consulta2}\"")

    # similarity_search_with_score retorna tuplas (Document, score)
    resultados_con_score = vectorstore.similarity_search_with_score(
        consulta2,
        k=3
    )

    for i, (doc, score) in enumerate(resultados_con_score, 1):
        print(f"\n  📄 Resultado {i}: (score: {score:.4f})")
        print(f"     Texto: {doc.page_content[:80]}...")

    # ── Búsqueda 3: Con filtrado por metadata ────────────────
    print(f"\n  {'─' * 45}")
    consulta3 = "herramientas de inteligencia artificial"
    print(f"\n  🔍 Consulta con filtro (solo categoría 'ia'): \"{consulta3}\"")

    resultados_filtrados = vectorstore.similarity_search(
        consulta3,
        k=3,
        filter={"categoria": "ia"}  # Solo documentos de categoría "ia"
    )

    for i, doc in enumerate(resultados_filtrados, 1):
        print(f"\n  📄 Resultado {i}:")
        print(f"     Categoría: {doc.metadata.get('categoria')}")
        print(f"     Texto: {doc.page_content[:80]}...")


# ══════════════════════════════════════════════════════════════
# PARTE 4: Retriever (como interfaz de LangChain)
# ══════════════════════════════════════════════════════════════

def ejemplo_retriever(vectorstore):
    """
    Un Retriever es la interfaz estándar de LangChain para buscar
    documentos. Se puede usar dentro de cadenas LCEL.
    """
    print("\n📌 PARTE 4: Retriever (interfaz estándar)")
    print("-" * 45)

    # ── Crear retriever desde el vectorstore ──────────────────
    # as_retriever() convierte el VectorStore en un Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",       # Tipo de búsqueda
        search_kwargs={"k": 3}          # Parámetros de búsqueda
    )

    print("  ✅ Retriever creado con search_type='similarity', k=3")

    # ── Usar el retriever ────────────────────────────────────
    consulta = "¿Cómo funciona la contenedorización?"
    print(f"\n  🔍 Consulta: \"{consulta}\"")

    docs = retriever.invoke(consulta)

    for i, doc in enumerate(docs, 1):
        print(f"  📄 [{i}] ({doc.metadata.get('categoria')}): {doc.page_content[:80]}...")

    return retriever


# ══════════════════════════════════════════════════════════════
# PARTE 5: RAG — Retrieval-Augmented Generation
# ══════════════════════════════════════════════════════════════

def ejemplo_rag(vectorstore):
    """
    RAG combina la búsqueda vectorial con generación de texto.
    1. El usuario hace una pregunta
    2. Se buscan documentos relevantes en la base vectorizada
    3. Los documentos se pasan como contexto al LLM
    4. El LLM genera una respuesta basada en los documentos
    """
    print("\n📌 PARTE 5: RAG (Retrieval-Augmented Generation)")
    print("-" * 50)

    # ── Componentes ──────────────────────────────────────────
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # ── Prompt con contexto ──────────────────────────────────
    # El prompt incluye los documentos recuperados como {contexto}
    prompt_rag = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente experto que responde preguntas
basándose ÚNICAMENTE en el contexto proporcionado.

Reglas:
- Usa SOLO la información del contexto para responder
- Si el contexto no contiene la respuesta, di "No tengo información suficiente"
- Cita la fuente (categoría) cuando sea posible
- Responde en español de forma clara y concisa

Contexto:
{contexto}"""),
        ("human", "{pregunta}")
    ])

    # ── Función para formatear documentos ────────────────────
    def formatear_docs(docs):
        """Convierte una lista de Documents en un string formateado."""
        return "\n\n".join([
            f"[{doc.metadata.get('categoria', 'N/A')}] {doc.page_content}"
            for doc in docs
        ])

    # ── Cadena RAG con LCEL ──────────────────────────────────
    # 1. Retriever busca documentos relevantes
    # 2. formatear_docs los convierte en texto
    # 3. El prompt inyecta el contexto
    # 4. El modelo genera la respuesta
    cadena_rag = (
        {
            "contexto": retriever | formatear_docs,
            "pregunta": RunnablePassthrough()
        }
        | prompt_rag
        | modelo
        | StrOutputParser()
    )

    print("  ✅ Cadena RAG construida: Retriever → Prompt → LLM → Parser\n")

    # ── Hacer preguntas ──────────────────────────────────────
    preguntas = [
        "¿Qué es pgvector y cómo funciona?",
        "¿Cuáles son las ventajas de usar Docker?",
        "¿Cómo funciona RAG con LangChain?",
    ]

    for pregunta in preguntas:
        print(f"  ❓ {pregunta}")
        respuesta = cadena_rag.invoke(pregunta)
        print(f"  💬 {respuesta}\n")
        print(f"  {'─' * 45}\n")

    return cadena_rag


# ══════════════════════════════════════════════════════════════
# PARTE 6: RAG con Memoria Conversacional
# ══════════════════════════════════════════════════════════════

def ejemplo_rag_con_memoria(vectorstore):
    """
    Combina RAG + Memoria usando LangGraph.
    El agente puede:
    - Buscar documentos en la base vectorizada
    - Recordar el contexto de la conversación anterior
    - Responder preguntas de seguimiento
    """
    print("\n📌 PARTE 6: RAG con Memoria Conversacional (LangGraph)")
    print("-" * 55)

    # ── Componentes ──────────────────────────────────────────
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # ── Estado del grafo ─────────────────────────────────────
    class EstadoRAG(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        contexto: str  # Documentos recuperados

    # ── Nodo 1: Recuperar documentos ─────────────────────────
    def nodo_recuperar(estado: EstadoRAG):
        """Busca documentos relevantes basados en el último mensaje."""
        # Obtener la última pregunta del usuario
        ultimo_mensaje = estado["messages"][-1].content

        # Buscar documentos similares
        docs = retriever.invoke(ultimo_mensaje)

        # Formatear documentos como contexto
        contexto = "\n\n".join([
            f"[{doc.metadata.get('categoria', 'N/A')}] {doc.page_content}"
            for doc in docs
        ])

        return {"contexto": contexto}

    # ── Nodo 2: Generar respuesta ────────────────────────────
    def nodo_responder(estado: EstadoRAG):
        """Genera una respuesta usando el contexto recuperado y la memoria."""
        system_msg = SystemMessage(content=f"""Eres un asistente con acceso a una base de conocimiento.
Usa el siguiente contexto para responder. Si el contexto no contiene
la información necesaria, usa tu conocimiento general.

IMPORTANTE: Recuerdas toda la conversación previa con el usuario.
Si el usuario hace preguntas de seguimiento, usa el contexto anterior.

Contexto de la base de conocimiento:
{estado.get('contexto', 'Sin contexto disponible')}

Responde en español, de forma clara y concisa.""")

        mensajes = [system_msg] + estado["messages"]
        respuesta = modelo.invoke(mensajes)

        return {"messages": [respuesta]}

    # ── Construir grafo ──────────────────────────────────────
    grafo = StateGraph(EstadoRAG)
    grafo.add_node("recuperar", nodo_recuperar)
    grafo.add_node("responder", nodo_responder)
    grafo.add_edge(START, "recuperar")
    grafo.add_edge("recuperar", "responder")
    grafo.add_edge("responder", END)

    # Compilar con memoria
    memoria = MemorySaver()
    app = grafo.compile(checkpointer=memoria)

    print("  ✅ Grafo RAG con memoria compilado")
    print("  📊 Flujo: Recuperar → Responder (con historial)\n")

    # ── Configuración con thread_id ──────────────────────────
    config = {"configurable": {"thread_id": "rag-conversacion-1"}}

    # ── Conversación con memoria ─────────────────────────────
    def chatear(pregunta: str):
        resultado = app.invoke(
            {"messages": [HumanMessage(content=pregunta)]},
            config=config
        )
        respuesta = resultado["messages"][-1].content
        print(f"  👤 {pregunta}")
        print(f"  🤖 {respuesta}\n")
        return resultado

    # Pregunta 1: Sobre pgvector
    chatear("¿Qué es pgvector y qué tipos de búsqueda soporta?")

    # Pregunta 2: Seguimiento (usa la memoria)
    chatear("¿Y cómo se conecta eso con LangChain?")

    # Pregunta 3: Sobre Docker
    chatear("¿Qué ventajas tiene Docker para desplegar aplicaciones?")

    # Pregunta 4: Pregunta de seguimiento que requiere memoria
    chatear("De todo lo que hemos hablado, ¿cuál es la mejor combinación de tecnologías para un proyecto de IA?")

    # ── Mostrar historial almacenado ─────────────────────────
    estado_final = app.get_state(config)
    total_msgs = len(estado_final.values.get("messages", []))
    print(f"  📊 Total mensajes en memoria: {total_msgs}")

    return app


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        # Crear VectorStore
        vectorstore, embeddings = crear_vectorstore()

        # Verificar si hay documentos existentes
        print("\n  🔍 Verificando documentos existentes...")
        test_results = vectorstore.similarity_search("test", k=1)

        if not test_results:
            print("  📝 No hay documentos. Ingresando documentos de ejemplo...")
            ingestar_documentos(vectorstore)
        else:
            print(f"  ✅ La colección ya tiene documentos. Saltando ingesta.")
            print(f"     (Para reingestar, elimina la colección en PostgreSQL)")

        # Ejecutar ejemplos
        ejemplo_busqueda_similitud(vectorstore)
        ejemplo_retriever(vectorstore)
        ejemplo_rag(vectorstore)
        ejemplo_rag_con_memoria(vectorstore)

        print("\n" + "=" * 60)
        print("✅ RESUMEN DE CONCEPTOS")
        print("=" * 60)
        print("""
  1. Embeddings         → Vectores numéricos que representan texto
  2. pgvector           → Extensión de PostgreSQL para vectores
  3. VectorStore        → Almacén de documentos con vectores
  4. Retriever          → Interfaz estándar para buscar documentos
  5. RAG                → Búsqueda + Generación de texto
  6. Memoria + RAG      → Conversación con contexto y documentos

  📦 Componentes clave:
     - OpenAIEmbeddings  → Genera vectores de texto
     - PGVector          → Conecta LangChain con PostgreSQL
     - as_retriever()    → Convierte VectorStore en Retriever
     - MemorySaver       → Memoria conversacional persistente

  🗄️ Base de datos:
     PostgreSQL + pgvector almacena los vectores nativamente.
     Soporta: similitud coseno, distancia euclidiana, producto interno.
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        if "connection" in str(e).lower() or "connect" in str(e).lower():
            print("\n💡 Asegúrate de tener PostgreSQL con pgvector corriendo:")
            print("   docker run --name pgvector -d \\")
            print("     -e POSTGRES_PASSWORD=postgres \\")
            print("     -e POSTGRES_DB=vectordb \\")
            print("     -p 5432:5432 \\")
            print("     pgvector/pgvector:pg16")
