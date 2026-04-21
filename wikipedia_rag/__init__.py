"""
═══════════════════════════════════════════════════════════════
  Wikipedia RAG — Paquete principal
═══════════════════════════════════════════════════════════════
  Pipeline completo para crear un agente conversacional que
  responde preguntas sobre Colombia basándose ÚNICAMENTE en
  artículos de Wikipedia almacenados en PostgreSQL (pgvector).

  Arquitectura:
    ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
    │  Scraper    │──▶│ VectorStore │──▶│   Agente     │
    │ (Wikipedia) │   │  (pgvector) │   │ (RAG+Memoria)│
    └─────────────┘   └─────────────┘   └──────────────┘

  Módulos:
    - config.py      → Configuración centralizada
    - scraper.py     → Descarga y chunking de Wikipedia
    - vectorstore.py → Conexión y operaciones con pgvector
    - agent.py       → Agente RAG con memoria conversacional
    - main.py        → Punto de entrada (CLI)
═══════════════════════════════════════════════════════════════
"""

from wikipedia_rag.config import Config
from wikipedia_rag.models import ModelFactory
from wikipedia_rag.scraper import WikipediaScraper
from wikipedia_rag.vectorstore import VectorStoreManager
from wikipedia_rag.agent import RAGAgent

__all__ = ["Config", "ModelFactory", "WikipediaScraper", "VectorStoreManager", "RAGAgent"]
