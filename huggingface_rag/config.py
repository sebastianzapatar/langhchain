import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración para la integración de HuggingFace."""
    
    # Modelo medio recomendado para buen balance entre rendimiento y calidad
    # Qwen2.5-3B-Instruct o microsoft/Phi-3-mini-4k-instruct
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    
    # Modelo de embeddings rápido y ligero para ejecutar localmente
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Configuraciones de generación
    TEMPERATURE = 0.1
    MAX_NEW_TOKENS = 512
    
    # Dispositivo a usar: 'cpu', 'cuda', 'mps' (para Mac Apple Silicon)
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

config = Config()
