"""
Fábrica para inicializar modelos locales de HuggingFace.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from huggingface_rag.config import config
import torch

class ModelFactory:
    
    @staticmethod
    def crear_llm():
        """
        Descarga y carga el modelo LLM en memoria.
        Esto puede tardar un poco la primera vez y consumirá RAM/VRAM.
        """
        print(f"📥 Cargando modelo LLM local: {config.LLM_MODEL}")
        print(f"⚙️  Usando dispositivo: {config.DEVICE}")
        
        # Para modelos de la familia Llama/Phi/Qwen es mejor usar bfloat16 si está disponible
        dtype = torch.bfloat16 if config.DEVICE in ["cuda", "mps"] else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL, 
            torch_dtype=dtype,
            device_map=config.DEVICE
        )
        
        # Crear el pipeline de transformers
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            do_sample=True, # Necesario para temperature > 0
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Envolverlo en la interfaz de LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ LLM cargado exitosamente.")
        return llm

    @staticmethod
    def crear_embeddings():
        """
        Descarga y carga el modelo de embeddings local.
        """
        print(f"📥 Cargando embeddings locales: {config.EMBEDDING_MODEL}")
        model_kwargs = {'device': config.DEVICE}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("✅ Embeddings cargados exitosamente.")
        return embeddings
