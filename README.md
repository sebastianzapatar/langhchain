# 🔗 LangChain & LangGraph — Ejemplos y Presentación

Proyecto educativo con ejemplos prácticos de **LangChain** conectado con **OpenAI** y un sistema avanzado de **agentes con LangGraph**.

## 📁 Estructura del Proyecto

```
langhchain/
├── 01_ejemplo_simple.py          # Ejemplo básico: LangChain + OpenAI
├── 02_ejemplo_langgraph_agentes.py  # Ejemplo avanzado: LangGraph + Multi-Agente
├── presentacion.html             # Presentación interactiva en HTML
├── requirements.txt              # Dependencias del proyecto
├── .env.example                  # Template de variables de entorno
└── README.md                     # Este archivo
```

## 🚀 Configuración Rápida

### 1. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar API Key de OpenAI
```bash
cp .env.example .env
# Edita .env y agrega tu API key real:
# OPENAI_API_KEY=sk-tu-api-key-aqui
```

### 4. Ejecutar los ejemplos
```bash
# Ejemplo simple
python 01_ejemplo_simple.py

# Ejemplo avanzado con agentes
python 02_ejemplo_langgraph_agentes.py
```

### 5. Ver la presentación
Abre `presentacion.html` en tu navegador.

## 📚 Contenido

### `01_ejemplo_simple.py`
- Invocación directa del modelo (`ChatOpenAI`)
- Uso de `PromptTemplates` con variables
- Cadenas LCEL con el operador pipe (`|`)
- `OutputParsers` para formatear respuestas
- Cadenas multi-paso

### `02_ejemplo_langgraph_agentes.py`
- Herramientas personalizadas con `@tool`
- Agente ReAct prebuilt (`create_react_agent`)
- Grafo personalizado con `StateGraph`
- Sistema Multi-Agente con **Supervisor** que delega a:
  - 🔢 Agente Calculadora
  - 🔎 Agente Investigador
  - 📊 Agente Analista

### `presentacion.html`
- ¿Qué es LangChain?
- Arquitectura y componentes
- Diagramas interactivos
- Cuándo usar / cuándo NO usar
- Ejemplos de código con syntax highlighting
- Sistema Multi-Agente con diagrama visual
- Comparativa vs alternativas
