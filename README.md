# Asistente turístico de Tenerife

Asistente conversacional que responde preguntas sobre Tenerife a partir de una guía turística en PDF, citando las páginas de las que sale cada dato, y que consulta la previsión meteorológica de cualquier punto de la isla. Usa Google Gemini a través de LangChain.

## Funcionalidades

| Componente | Implementación |
| --- | --- |
| LLM | Gemini (`gemini-3.5-flash-lite`) con `init_chat_model`; API key en `.env`; parámetros de generación expuestos en `MODEL_PARAMS` |
| RAG | `PyPDFLoader` → chunking por secciones de la guía (zona y lugar) → embeddings `gemini-embedding-001` → `InMemoryVectorStore` persistido en disco; respuestas citadas como *(Guía, pág. N)* |
| Herramientas | `search_tenerife_guide` (la guía como herramienta) y `get_weather(fecha, location)` contra la API de Open-Meteo, ambas definidas con JSON Schema y validadas con `jsonschema` |
| Gestión de errores y log | Los fallos de las herramientas vuelven al modelo de forma estructurada y cada intento queda en `logs/tool_calls.log` |
| Diálogo multiturno | Historial con recorte por tokens estimados con `tiktoken` |
| Streaming | La respuesta se imprime token a token |
| Varias herramientas | El modelo decide qué herramienta usar y puede combinar ambas en un mismo turno |
| Evaluación | hit@k del retriever para dos estrategias de chunking; selección de herramientas, contenido de la respuesta y latencia del asistente, con gráficos |
| Chat interactivo | Conversación libre dentro del notebook, con ejemplos de uso, `reset` para reiniciar y mensajes de error comprensibles |

## Estructura

```
.
├── notebooks/
│   └── asistente_tenerife.ipynb   # Notebook principal (ejecutar de arriba abajo)
├── data/
│   ├── TENERIFE.pdf               # Guía turística
│   └── index/                     # Índices vectoriales generados (ignorado por git)
├── logs/
│   └── tool_calls.log             # Intentos de llamadas a herramientas (generado)
├── informe.md                     # Diseño, decisiones técnicas, resultados y limitaciones
├── requirements.txt
├── .env.example
└── .gitignore
```

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

1. Crea una API key gratuita en [Google AI Studio](https://aistudio.google.com/apikey).
2. Copia la plantilla de variables de entorno y pega la clave:

   ```bash
   cp .env.example .env
   ```

## Ejecución

```bash
jupyter notebook notebooks/asistente_tenerife.ipynb
```

La primera ejecución calcula los embeddings de la guía y los guarda en `data/index/`. Las siguientes reutilizan ese índice. Si cambias la guía, borra la carpeta `data/index/`.

## Notas sobre la cuota de Gemini

- La capa gratuita permite 20 peticiones diarias por modelo y el notebook completo hace unas 45, así que hace falta activar la facturación del proyecto en AI Studio y comprar crédito. Comprados minimos creditos para el testeo del notebook.
- `gemini-3.5-flash-lite` usa valores de muestreo fijos: ignora `temperature` y `top_p` (LangChain muestra un aviso) y solo aplica `max_tokens`.
- Si el modelo deja de estar disponible, cambia `GENERATION_MODEL` en la sección 0.2 del notebook.

## Problemas frecuentes

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| `429` o `RESOURCE_EXHAUSTED` | Cuota agotada (capa gratuita) o límite por minuto | Activada facturación del proyecto o aumenta `RATE_LIMIT_PAUSE_SECONDS` |
| `404` o modelo no encontrado | Modelo retirado o renombrado | Cambia `GENERATION_MODEL` en la sección 0.2 |
| Error al crear `ENCODER` | `tiktoken` descarga su tokenizador la primera vez | Ejecuta la celda con conexión a internet |
| Respuestas que citan secciones antiguas tras cambiar la guía | Índice vectorial desactualizado | Borra `data/index/` y vuelve a ejecutar |
