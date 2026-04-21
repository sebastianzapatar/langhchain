#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Wikipedia RAG — Punto de entrada (CLI)
═══════════════════════════════════════════════════════════════
  Pipeline: Wikipedia → pgvector → Agente RAG sobre Colombia

  Modos de ejecución:
    python -m wikipedia_rag                             # Completo (OpenAI)
    python -m wikipedia_rag --provider ollama            # Completo (Ollama local)
    python -m wikipedia_rag --solo-ingesta               # Solo descargar y guardar
    python -m wikipedia_rag --solo-chat                  # Solo chatear
    python -m wikipedia_rag --solo-chat --provider ollama # Chat con Ollama
    python -m wikipedia_rag --reingestar                 # Forzar re-descarga
    python -m wikipedia_rag --demo-memoria               # Demo de memoria

  Proveedores:
    openai (default): Requiere OPENAI_API_KEY, alta calidad, de pago
    ollama:           Modelos locales, gratis, privado

  Requisitos:
    - PostgreSQL con pgvector corriendo (docker)
    - OPENAI_API_KEY en .env (solo para provider=openai)
    - pip install -r requirements.txt
═══════════════════════════════════════════════════════════════
"""

import sys
import argparse

from wikipedia_rag.config import Config
from wikipedia_rag.scraper import WikipediaScraper
from wikipedia_rag.vectorstore import VectorStoreManager
from wikipedia_rag.agent import RAGAgent


def ejecutar_ingesta(config: Config) -> VectorStoreManager:
    """
    Ejecuta el pipeline de ingesta: Wikipedia → chunks → pgvector.

    Returns:
        VectorStoreManager conectado y con los documentos ingresados.
    """
    # 1. Descargar artículos de Wikipedia
    scraper = WikipediaScraper(config)
    documentos = scraper.descargar_articulos()

    if not documentos:
        print("  ❌ No se pudieron descargar artículos de Wikipedia.")
        sys.exit(1)

    # 2. Dividir en chunks
    chunks = scraper.dividir_en_chunks(documentos)

    # 3. Almacenar en pgvector
    vs_manager = VectorStoreManager(config)
    total = vs_manager.ingestar_documentos(chunks)
    print(f"\n  ✅ Ingesta completada: {total} chunks en pgvector")

    return vs_manager


def demo_memoria(config: Config, vs_manager: VectorStoreManager) -> None:
    """
    Demostración explícita del sistema de memoria.

    Esta función muestra paso a paso cómo funciona la memoria
    conversacional con MemorySaver y thread_id.
    """
    print("\n" + "=" * 60)
    print("🧠 DEMOSTRACIÓN DE MEMORIA CONVERSACIONAL")
    print("=" * 60)

    agente = RAGAgent(config, vs_manager)

    # ══════════════════════════════════════════════════════════
    # THREAD 1: Conversación con memoria
    # ══════════════════════════════════════════════════════════
    thread_id = "demo-memoria-1"
    print(f"\n  📌 Thread ID: \"{thread_id}\"")
    print(f"     (todas las preguntas con este ID comparten memoria)\n")

    # Pregunta 1
    print("  ─── Pregunta 1 ───")
    r1 = agente.preguntar("¿Cuál es la capital de Colombia?", thread_id)
    print(f"  👤 ¿Cuál es la capital de Colombia?")
    print(f"  🤖 {r1['respuesta'][:200]}...")
    print(f"  📊 Mensajes en memoria: {agente.contar_mensajes(thread_id)}")
    # Ahora hay 2 mensajes: [HumanMsg, AIMsg]

    # Pregunta 2 — SEGUIMIENTO (usa la memoria)
    print("\n  ─── Pregunta 2 (seguimiento) ───")
    r2 = agente.preguntar("¿Cuántos habitantes tiene?", thread_id)
    print(f"  👤 ¿Cuántos habitantes tiene?")
    print(f"     ↑ El agente SABE que se refiere a Bogotá gracias a la memoria")
    print(f"  🤖 {r2['respuesta'][:200]}...")
    print(f"  📊 Mensajes en memoria: {agente.contar_mensajes(thread_id)}")
    # Ahora hay 4 mensajes: [Human1, AI1, Human2, AI2]

    # Pregunta 3 — SEGUIMIENTO
    print("\n  ─── Pregunta 3 (seguimiento) ───")
    r3 = agente.preguntar("¿Y qué más me puedes contar de esa ciudad?", thread_id)
    print(f"  👤 ¿Y qué más me puedes contar de esa ciudad?")
    print(f"     ↑ 'esa ciudad' = Bogotá (contexto de la memoria)")
    print(f"  🤖 {r3['respuesta'][:200]}...")
    print(f"  📊 Mensajes en memoria: {agente.contar_mensajes(thread_id)}")
    # Ahora hay 6 mensajes

    # ══════════════════════════════════════════════════════════
    # THREAD 2: Conversación INDEPENDIENTE
    # ══════════════════════════════════════════════════════════
    thread_id_2 = "demo-memoria-2"
    print(f"\n\n  📌 Thread ID: \"{thread_id_2}\" (conversación NUEVA)")
    print(f"     (este thread NO tiene memoria del anterior)\n")

    r4 = agente.preguntar("¿Qué tipo de biodiversidad tiene?", thread_id_2)
    print(f"  👤 ¿Qué tipo de biodiversidad tiene?")
    print(f"     ↑ Sin contexto previo — no sabe de qué se habla")
    print(f"  🤖 {r4['respuesta'][:200]}...")
    print(f"  📊 Mensajes en thread 1: {agente.contar_mensajes(thread_id)}")
    print(f"  📊 Mensajes en thread 2: {agente.contar_mensajes(thread_id_2)}")

    # ══════════════════════════════════════════════════════════
    # Mostrar historial almacenado
    # ══════════════════════════════════════════════════════════
    print(f"\n\n  {'═' * 50}")
    print(f"  📜 HISTORIAL ALMACENADO EN MEMORIA")
    print(f"  {'═' * 50}")

    historial = agente.obtener_historial(thread_id)
    print(f"\n  Thread \"{thread_id}\" ({len(historial)} mensajes):")
    for msg in historial:
        tipo = "👤 Human" if msg.type == "human" else "🤖 AI"
        texto = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"    {tipo}: {texto}")

    print(f"\n  💡 La memoria permite preguntas de seguimiento:")
    print(f"     'tiene' → se refiere al último contexto (Bogotá)")
    print(f"     'esa ciudad' → referencia anafórica resuelta por la memoria")


def chat_interactivo(agente: RAGAgent) -> None:
    """
    Chat interactivo en terminal.

    Comandos:
      - 'salir'    → Terminar
      - 'nuevo'    → Nueva conversación (nuevo thread_id)
      - 'historial' → Ver mensajes en memoria
      - 'memoria'  → Ver cuántos mensajes tiene el thread actual
    """
    print("\n" + "=" * 60)
    print("💬 CHAT INTERACTIVO — Pregunta sobre Colombia")
    print("=" * 60)
    print("  📝 Escribe tu pregunta y presiona Enter")
    print("  🔚 'salir'     → Terminar")
    print("  🔄 'nuevo'     → Nueva conversación (limpia memoria)")
    print("  📜 'historial' → Ver mensajes en memoria")
    print("  🧠 'memoria'   → Ver estado de la memoria")
    print("-" * 60)

    thread_id = "chat-interactivo-1"
    thread_counter = 1
    print(f"  📌 Thread ID: \"{thread_id}\"\n")

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

        if pregunta.lower() == "nuevo":
            thread_counter += 1
            thread_id = f"chat-interactivo-{thread_counter}"
            print(f"  🔄 Nueva conversación iniciada")
            print(f"  📌 Thread ID: \"{thread_id}\" (memoria limpia)\n")
            continue

        if pregunta.lower() == "historial":
            historial = agente.obtener_historial(thread_id)
            print(f"\n  📜 Historial del thread \"{thread_id}\":")
            if not historial:
                print("     (vacío)")
            for msg in historial:
                tipo = "👤" if msg.type == "human" else "🤖"
                texto = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"     {tipo} {texto}")
            print()
            continue

        if pregunta.lower() == "memoria":
            total = agente.contar_mensajes(thread_id)
            print(f"\n  🧠 Thread: \"{thread_id}\"")
            print(f"     Mensajes en memoria: {total}")
            print(f"     (cada pregunta+respuesta = 2 mensajes)\n")
            continue

        # ── Enviar pregunta al agente ────────────────────────
        try:
            resultado = agente.preguntar(pregunta, thread_id)

            print(f"\n  🤖 Agente: {resultado['respuesta']}")

            if resultado["fuentes"]:
                print(f"\n  📚 Fuentes: {', '.join(resultado['fuentes'])}")

            msgs = agente.contar_mensajes(thread_id)
            print(f"  ⏱️  {resultado['tiempo_ms']:.0f}ms | 🧠 {msgs} msgs en memoria\n")

        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="🇨🇴 Wikipedia → pgvector → Agente RAG sobre Colombia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m wikipedia_rag                              # OpenAI (default)
  python -m wikipedia_rag --provider ollama             # Ollama local
  python -m wikipedia_rag --provider ollama --modelo llama3.2:3b
  python -m wikipedia_rag --solo-ingesta               # Solo ingestar
  python -m wikipedia_rag --solo-chat                   # Solo chatear
  python -m wikipedia_rag --solo-chat --provider ollama # Chat con Ollama
  python -m wikipedia_rag --demo-memoria               # Demo de memoria
  python -m wikipedia_rag --reingestar                 # Re-descargar
        """
    )
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                        help="Proveedor de modelos: 'openai' (cloud) o 'ollama' (local)")
    parser.add_argument("--modelo", type=str, default=None,
                        help="Modelo de chat a usar (ej: llama3.2:1b, gpt-4o-mini)")
    parser.add_argument("--embedding", type=str, default=None,
                        help="Modelo de embeddings (ej: nomic-embed-text)")
    parser.add_argument("--solo-chat", action="store_true",
                        help="Solo iniciar el chat (sin descargar)")
    parser.add_argument("--solo-ingesta", action="store_true",
                        help="Solo descargar e ingestar (sin chat)")
    parser.add_argument("--reingestar", action="store_true",
                        help="Forzar re-ingesta de Wikipedia")
    parser.add_argument("--demo-memoria", action="store_true",
                        help="Ejecutar demo del sistema de memoria")

    args = parser.parse_args()

    print("=" * 60)
    print("🇨🇴 Wikipedia → pgvector → Agente RAG sobre Colombia")
    print("=" * 60)

    try:
        # ── Configurar proveedor ────────────────────────────
        config = Config(provider=args.provider)

        # Aplicar defaults de Ollama si es necesario
        config.aplicar_defaults_ollama()

        # Sobreescribir modelo si se pasó por CLI
        if args.modelo:
            config.chat_model = args.modelo
        if args.embedding:
            config.embedding_model = args.embedding

        config.validate()

        # Mostrar configuración activa
        from wikipedia_rag.models import ModelFactory
        ModelFactory.imprimir_info(config)

        if args.solo_ingesta:
            ejecutar_ingesta(config)

        elif args.solo_chat or args.demo_memoria:
            print("\n  🔌 Conectando al vectorstore existente...")
            vs_manager = VectorStoreManager(config)

            if not vs_manager.tiene_documentos():
                print("  ❌ No hay documentos. Ejecuta primero sin --solo-chat.")
                sys.exit(1)

            print("  ✅ Documentos encontrados en la base de datos\n")

            if args.demo_memoria:
                demo_memoria(config, vs_manager)
            else:
                agente = RAGAgent(config, vs_manager)
                chat_interactivo(agente)

        else:
            # Modo completo
            print("\n  🔌 Verificando datos existentes...")
            vs_manager = VectorStoreManager(config)

            if vs_manager.tiene_documentos() and not args.reingestar:
                print(f"  ✅ '{config.collection_name}' ya tiene documentos.")
                print("  ⏭️  Saltando ingesta. Usa --reingestar para forzar.\n")
            else:
                vs_manager = ejecutar_ingesta(config)

            agente = RAGAgent(config, vs_manager)
            chat_interactivo(agente)

    except KeyboardInterrupt:
        print("\n\n  👋 Interrumpido por el usuario.")
    except ValueError as e:
        print(f"\n  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

        if "connect" in str(e).lower():
            print("\n  💡 ¿PostgreSQL está corriendo?")
            print("     docker run --name pgvector -d \\")
            print("       -e POSTGRES_PASSWORD=postgres \\")
            print("       -e POSTGRES_DB=vectordb \\")
            print("       -p 5432:5432 \\")
            print("       pgvector/pgvector:pg16")


if __name__ == "__main__":
    main()
