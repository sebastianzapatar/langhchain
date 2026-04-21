"""
============================================================
Wikipedia → pgvector → Agente RAG (Colombia)
============================================================
Este script:
  1. Descarga artículos de Wikipedia sobre Colombia (en español)
  2. Los divide en fragmentos (chunks) manejables
  3. Los almacena como vectores en PostgreSQL (pgvector)
  4. Crea un agente conversacional que SOLO responde con la
     información de la base de datos — NO alucina

Requisitos:
  - PostgreSQL con extensión pgvector corriendo
  - pip install langchain langchain-openai langchain-postgres
  - pip install langchain-community wikipedia psycopg[binary]
  - OPENAI_API_KEY configurada en .env

Levantar PostgreSQL:
  docker run --name pgvector -d \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=vectordb \
    -p 5432:5432 \
    pgvector/pgvector:pg16

Ejecutar:
  python 05_wikipedia_rag.py
============================================================
"""

import os
import sys
import time
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

# ── Cargar variables de entorno ───────────────────────────────
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: No se encontró OPENAI_API_KEY en el archivo .env")
    sys.exit(1)

# ── Importaciones ─────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/vectordb"
)

COLLECTION_NAME = "wikipedia_colombia"

# Temas de Wikipedia a buscar (en español)
WIKIPEDIA_QUERIES = [
    "Colombia",
    "Historia de Colombia",
    "Geografía de Colombia",
    "Economía de Colombia",
    "Cultura de Colombia",
    "Biodiversidad de Colombia",
    "Bogotá",
    "Medellín",
    "Cartagena de Indias",
    "Café de Colombia",
]

print("=" * 60)
print("🇨🇴 Wikipedia → pgvector → Agente RAG sobre Colombia")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# PASO 1: Descargar artículos de Wikipedia
# ══════════════════════════════════════════════════════════════

def descargar_wikipedia():
    """
    Descarga artículos de Wikipedia sobre Colombia usando
    WikipediaLoader de LangChain Community.
    """
    print("\n📥 PASO 1: Descargando artículos de Wikipedia...")
    print("-" * 50)

    todos_los_docs = []
    titulos_vistos = set()  # Evitar duplicados

    for query in WIKIPEDIA_QUERIES:
        print(f"\n  🔍 Buscando: \"{query}\"...")
        try:
            loader = WikipediaLoader(
                query=query,
                lang="es",                  # Wikipedia en español
                load_max_docs=2,            # Máximo 2 artículos por búsqueda
                doc_content_chars_max=15000  # Máximo 15K caracteres por artículo
            )
            docs = loader.load()

            for doc in docs:
                titulo = doc.metadata.get("title", "Sin título")
                if titulo not in titulos_vistos:
                    titulos_vistos.add(titulo)
                    todos_los_docs.append(doc)
                    chars = len(doc.page_content)
                    print(f"    ✅ \"{titulo}\" ({chars:,} caracteres)")
                else:
                    print(f"    ⏭️  \"{titulo}\" (ya descargado, saltando)")

        except Exception as e:
            print(f"    ⚠️  Error buscando \"{query}\": {e}")
            continue

    print(f"\n  📊 Total artículos descargados: {len(todos_los_docs)}")
    total_chars = sum(len(d.page_content) for d in todos_los_docs)
    print(f"  📝 Total caracteres: {total_chars:,}")

    return todos_los_docs


# ══════════════════════════════════════════════════════════════
# PASO 2: Dividir en chunks
# ══════════════════════════════════════════════════════════════

def dividir_en_chunks(documentos):
    """
    Divide los documentos largos en fragmentos más pequeños
    para mejor precisión en la búsqueda vectorial.
    """
    print("\n✂️  PASO 2: Dividiendo documentos en chunks...")
    print("-" * 50)

    # RecursiveCharacterTextSplitter: divide por párrafos, oraciones, etc.
    # chunk_size: tamaño máximo de cada fragmento en caracteres
    # chunk_overlap: solapamiento entre fragmentos para no perder contexto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )

    chunks = splitter.split_documents(documentos)

    print(f"  📄 Documentos originales: {len(documentos)}")
    print(f"  📦 Chunks generados: {len(chunks)}")
    print(f"  📏 Tamaño promedio: {sum(len(c.page_content) for c in chunks) // len(chunks)} caracteres")

    # Mostrar ejemplo
    if chunks:
        print(f"\n  📋 Ejemplo de chunk (primero):")
        print(f"     Título: {chunks[0].metadata.get('title', 'N/A')}")
        print(f"     Texto: {chunks[0].page_content[:150]}...")

    return chunks


# ══════════════════════════════════════════════════════════════
# PASO 3: Almacenar en pgvector
# ══════════════════════════════════════════════════════════════

def almacenar_en_pgvector(chunks):
    """
    Convierte cada chunk en un vector (embedding) y lo almacena
    en PostgreSQL usando pgvector.
    """
    print("\n🗄️  PASO 3: Almacenando en PostgreSQL (pgvector)...")
    print("-" * 50)

    # Crear modelo de embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("  ✅ Modelo de embeddings: text-embedding-3-small")

    # Conectar al vectorstore
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    print(f"  ✅ Conectado a colección: {COLLECTION_NAME}")

    # Insertar en lotes para evitar timeouts
    batch_size = 50
    total_insertados = 0

    print(f"\n  📤 Insertando {len(chunks)} chunks en lotes de {batch_size}...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = vectorstore.add_documents(batch)
        total_insertados += len(ids)
        print(f"    ✅ Lote {i // batch_size + 1}: {len(ids)} chunks insertados ({total_insertados}/{len(chunks)})")

    print(f"\n  🎉 Total insertados: {total_insertados} chunks")
    return vectorstore


# ══════════════════════════════════════════════════════════════
# PASO 4: Conectar al vectorstore existente
# ══════════════════════════════════════════════════════════════

def conectar_vectorstore():
    """Conecta al vectorstore existente en PostgreSQL."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    return vectorstore


# ══════════════════════════════════════════════════════════════
# PASO 5: Crear el Agente RAG con Memoria
# ══════════════════════════════════════════════════════════════

def crear_agente_rag(vectorstore):
    """
    Crea un agente conversacional con:
    - Búsqueda vectorial en pgvector (RAG)
    - Memoria conversacional (MemorySaver)
    - Prompt estricto anti-alucinación
    """
    print("\n🤖 PASO 5: Creando agente RAG con memoria...")
    print("-" * 50)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Top 5 documentos más relevantes
    )
    modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── Estado del grafo ─────────────────────────────────────
    class EstadoRAG(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        contexto: str
        fuentes: list[str]

    # ── Nodo 1: Recuperar documentos relevantes ──────────────
    def nodo_recuperar(estado: EstadoRAG):
        """Busca documentos similares a la pregunta del usuario."""
        ultimo_mensaje = estado["messages"][-1].content

        docs = retriever.invoke(ultimo_mensaje)

        if not docs:
            return {
                "contexto": "",
                "fuentes": []
            }

        contexto = "\n\n---\n\n".join([
            f"[Fuente: {doc.metadata.get('title', 'Desconocida')}]\n{doc.page_content}"
            for doc in docs
        ])

        fuentes = list(set(
            doc.metadata.get("title", "Desconocida") for doc in docs
        ))

        return {
            "contexto": contexto,
            "fuentes": fuentes
        }

    # ── Nodo 2: Generar respuesta (SIN alucinaciones) ────────
    def nodo_responder(estado: EstadoRAG):
        """
        Genera una respuesta basada ÚNICAMENTE en el contexto recuperado.
        Si no hay información suficiente, lo dice explícitamente.
        """
        contexto = estado.get("contexto", "")
        fuentes = estado.get("fuentes", [])

        # PROMPT ANTI-ALUCINACIÓN: muy estricto
        system_content = f"""Eres un asistente experto sobre Colombia. Tu conocimiento proviene 
EXCLUSIVAMENTE de artículos de Wikipedia almacenados en una base de datos.

═══ REGLAS ESTRICTAS ═══
1. SOLO puedes responder con información del CONTEXTO proporcionado abajo.
2. Si el contexto NO contiene la información necesaria para responder, 
   debes decir EXACTAMENTE: "⚠️ No tengo información suficiente en mi base 
   de datos para responder esta pregunta. La información que tengo proviene 
   de artículos de Wikipedia sobre Colombia y sus temas relacionados."
3. NUNCA inventes datos, fechas, nombres o estadísticas.
4. Si solo tienes información parcial, responde lo que puedas y aclara 
   qué parte no tienes.
5. Cita las fuentes (artículos de Wikipedia) de donde sacas la información.
6. Responde en español, de forma clara y bien estructurada.
7. Recuerdas la conversación previa con el usuario.

═══ CONTEXTO DE LA BASE DE DATOS ═══
{contexto if contexto else "⚠️ No se encontraron documentos relevantes en la base de datos."}

═══ FUENTES DISPONIBLES ═══
{', '.join(fuentes) if fuentes else "Ninguna"}"""

        mensajes = [SystemMessage(content=system_content)] + estado["messages"]
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

    print("  ✅ Agente RAG compilado con memoria")
    print("  📊 Flujo: Pregunta → Recuperar (pgvector) → Responder (GPT-4o-mini)")
    print("  🛡️  Modo anti-alucinación activado")

    return app


# ══════════════════════════════════════════════════════════════
# PASO 6: Chat interactivo
# ══════════════════════════════════════════════════════════════

def chat_interactivo(agente):
    """
    Inicia un chat interactivo con el agente RAG.
    El usuario puede hacer preguntas sobre Colombia y el agente
    responde SOLO con información de la base de datos.
    """
    print("\n" + "=" * 60)
    print("💬 CHAT INTERACTIVO — Pregunta sobre Colombia")
    print("=" * 60)
    print("  📝 Escribe tu pregunta y presiona Enter")
    print("  🔚 Escribe 'salir' para terminar")
    print("  🔄 Escribe 'nuevo' para iniciar nueva conversación")
    print("-" * 60)

    config = {"configurable": {"thread_id": "colombia-chat-1"}}
    thread_counter = 1

    while True:
        try:
            pregunta = input("\n  👤 Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 ¡Hasta luego!")
            break

        if not pregunta:
            continue

        if pregunta.lower() == "salir":
            print("\n  👋 ¡Hasta luego!")
            break

        if pregunta.lower() == "nuevo":
            thread_counter += 1
            config = {"configurable": {"thread_id": f"colombia-chat-{thread_counter}"}}
            print("  🔄 Nueva conversación iniciada")
            continue

        # Enviar pregunta al agente
        inicio = time.time()
        try:
            resultado = agente.invoke(
                {"messages": [HumanMessage(content=pregunta)]},
                config=config
            )

            respuesta = resultado["messages"][-1].content
            tiempo = (time.time() - inicio) * 1000

            print(f"\n  🤖 Agente: {respuesta}")

            # Mostrar fuentes si están disponibles
            fuentes = resultado.get("fuentes", [])
            if fuentes:
                print(f"\n  📚 Fuentes: {', '.join(fuentes)}")
            print(f"  ⏱️  Tiempo: {tiempo:.0f}ms")

        except Exception as e:
            print(f"\n  ❌ Error: {e}")


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wikipedia → pgvector → Agente RAG sobre Colombia"
    )
    parser.add_argument(
        "--solo-chat",
        action="store_true",
        help="Solo iniciar el chat (sin descargar/ingestar Wikipedia)"
    )
    parser.add_argument(
        "--solo-ingesta",
        action="store_true",
        help="Solo descargar e ingestar (sin chat interactivo)"
    )
    parser.add_argument(
        "--reingestar",
        action="store_true",
        help="Forzar re-ingesta aunque ya existan documentos"
    )

    args = parser.parse_args()

    try:
        if args.solo_chat:
            # ── Modo solo chat ───────────────────────────────
            print("\n  🔌 Conectando al vectorstore existente...")
            vectorstore = conectar_vectorstore()

            # Verificar que hay documentos
            test = vectorstore.similarity_search("Colombia", k=1)
            if not test:
                print("  ❌ No hay documentos en la base de datos.")
                print("  💡 Ejecuta primero sin --solo-chat para ingestar Wikipedia.")
                sys.exit(1)

            print(f"  ✅ Base de datos lista con documentos")
            agente = crear_agente_rag(vectorstore)
            chat_interactivo(agente)

        elif args.solo_ingesta:
            # ── Modo solo ingesta ────────────────────────────
            documentos = descargar_wikipedia()
            if not documentos:
                print("  ❌ No se pudieron descargar artículos.")
                sys.exit(1)
            chunks = dividir_en_chunks(documentos)
            almacenar_en_pgvector(chunks)
            print("\n  ✅ Ingesta completada. Usa --solo-chat para chatear.")

        else:
            # ── Modo completo ────────────────────────────────
            # Verificar si ya hay datos
            vectorstore = conectar_vectorstore()
            test = vectorstore.similarity_search("Colombia", k=1)

            if test and not args.reingestar:
                print(f"\n  ✅ La colección '{COLLECTION_NAME}' ya tiene documentos.")
                print("  ⏭️  Saltando ingesta. Usa --reingestar para forzar.")
            else:
                # Descargar e ingestar
                documentos = descargar_wikipedia()
                if not documentos:
                    print("  ❌ No se pudieron descargar artículos de Wikipedia.")
                    sys.exit(1)
                chunks = dividir_en_chunks(documentos)
                vectorstore = almacenar_en_pgvector(chunks)

            # Crear agente y chatear
            agente = crear_agente_rag(vectorstore)
            chat_interactivo(agente)

    except KeyboardInterrupt:
        print("\n\n  👋 Interrumpido por el usuario.")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

        if "connect" in str(e).lower() or "connection" in str(e).lower():
            print("\n  💡 ¿PostgreSQL está corriendo? Levántalo con:")
            print("     docker run --name pgvector -d \\")
            print("       -e POSTGRES_PASSWORD=postgres \\")
            print("       -e POSTGRES_DB=vectordb \\")
            print("       -p 5432:5432 \\")
            print("       pgvector/pgvector:pg16")
