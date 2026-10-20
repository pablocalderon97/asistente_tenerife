# Asistente turístico de Tenerife: informe técnico

## 1. Objetivo

Construir un prototipo conversacional reproducible que responda preguntas sobre Tenerife usando la guía `data/TENERIFE.pdf` como única fuente de recomendaciones, que mantenga el contexto entre turnos y que consulte la previsión meteorológica mediante una llamada de función.

## 2. Diseño de la solución

```
                    ┌──────────────────────────────────────────────┐
 Usuario ──────────▶│ run_turn (bucle de tool calling)             │
                    │  1. instrucciones + fecha de hoy             │
                    │  2. historial con recorte por tokens         │
                    │  3. pregunta                                 │
                    └───────────────┬──────────────────────────────┘
                                    │ streaming
                                    ▼
                           Gemini (LangChain)
                     ┌──────────────┴──────────────┐
            tool_call│                             │tool_call
                     ▼                             ▼
          search_tenerife_guide               get_weather
          InMemoryVectorStore                 Open-Meteo
          (chunks por sección)                (destinos de la guía + geocoding)
                     │                             │
                     └──────────┬──────────────────┘
                                ▼
          execute_tool_call: validación JSON Schema, captura de errores,
          ToolExecution y log en logs/tool_calls.log
                                │
                                ▼
                 respuesta final citada → memoria (solo pregunta y respuesta)
```

El sistema tiene tres capas:

1. **Conocimiento (RAG).** La guía se carga con `PyPDFLoader`, se limpia, se agrupa por zonas y lugares, se trocea por secciones, se vectoriza con `gemini-embedding-001` y se indexa en `InMemoryVectorStore`, que se persiste en disco.
2. **Herramientas.** La búsqueda en la guía se expone como herramienta (`search_tenerife_guide`) junto a `get_weather`. Un ejecutor común valida argumentos, ejecuta, captura errores y registra cada intento.
3. **Conversación.** `run_turn` implementa el bucle modelo → herramientas → modelo con streaming, y `add_to_history` mantiene el historial dentro de un presupuesto de tokens.

## 3. Decisiones técnicas

### 3.1. Proveedor, modelo y parámetros

- **Gemini**, usado a través de `init_chat_model` de LangChain. La capa gratuita limita el uso a 20 peticiones diarias por modelo, insuficientes para ejecutar el notebook completo (unas 45 llamadas), por lo que se activó la facturación del proyecto. El coste de una ejecución completa es de céntimos.
- **`gemini-3.5-flash-lite`**: rápido, barato y con soporte de llamadas a funciones.
- **Parámetros expuestos en `MODEL_PARAMS`:** `temperature=0.2` y `top_p=0.9`, pensados para respuestas factuales y estables, y `max_tokens=1024`, suficiente para respuestas con citas. Este modelo usa valores de muestreo fijos: LangChain avisa de que ignora `temperature` y `top_p`, así que en la práctica solo `max_tokens` tiene efecto. Se mantienen en la configuración para que quede explícito qué se pidió y para que surtan efecto si se cambia a un modelo que sí los admita.
- **`max_retries=6` y pausas entre llamadas**: reintentos automáticos ante errores de límite de peticiones.

### 3.2. Análisis del documento

La guía tiene 25 páginas de texto y es muy visual. La extracción por página muestra tres problemas:

- **Texto muy irregular por página**, entre 0 y 2.429 caracteres.
- **Páginas que son solo pies de foto.** La página 2 únicamente contiene *"Auditorio de Tenerife"* y *"Plaza de España"*, sin mencionar Santa Cruz.
- **Frases que continúan en la página siguiente.** La página 19 empieza por *"También abre durante el día como restaurante…"* sin nombrar el local.

Además, el texto incluye muchos restos de enlaces (`[vídeo - ubicación]`, `(ubicación)`, `[web]`, `[Tripadvisor]`) sin valor semántico.

A cambio, la guía tiene una estructura muy clara: encabezados de zona (`ZONA NORTE`, `ZONA SUR`, `OTRAS VAINAS`), una viñeta `•` por lugar y viñetas de segundo y tercer nivel (`o`, `-`, `▪`) para rutas, puntos de interés y restaurantes.

### 3.3. Chunking

Se implementaron dos estrategias, ambas con `RecursiveCharacterTextSplitter`:

- **Por páginas (referencia):** cada página limpia se trocea por separado.
- **Por secciones (elegida):**
  1. Se eliminan los restos de enlaces con una expresión regular.
  2. Cada viñeta `•` abre una sección (un lugar) que hereda la zona del último encabezado. Así se agrupan textos de un mismo lugar repartidos en varias páginas.
  3. Cada sección se trocea y a cada chunk se le antepone una cabecera como `[Zona Norte > La Orotava]`.

La cabecera garantiza que cualquier fragmento, por corto que sea, quede ligado a su lugar tanto en el embedding como en el contexto que ve el modelo. Cada chunk guarda la zona, el lugar y el rango de páginas de su sección (`page_label`, por ejemplo `"24-25"`), que es lo que se cita.

Con 1.000 caracteres y 150 de solape, la guía produce 24 chunks y la mayoría de lugares caben en uno o dos. La evaluación (sección 4.1) compara esta configuración con la estrategia por páginas.

### 3.4. Vector store

`InMemoryVectorStore` persistido con `dump()`/`load()`. Los embeddings de la guía se calculan una sola vez y no consumen cuota en cada ejecución. Con unas decenas de chunks, la búsqueda exhaustiva en memoria es instantánea y no justifica un vector store externo.

### 3.5. La guía como herramienta

En lugar de recuperar siempre, el modelo decide cuándo buscar. Ventajas:

- No se gasta contexto en turnos que no lo necesitan (saludos, resúmenes de la conversación).
- La descripción del parámetro `query` exige una búsqueda **autocontenida**, de modo que el modelo reescribe preguntas como *"¿y qué tiempo hará allí?"* o *"¿dónde como a la vuelta?"* usando el historial.

El flujo RAG fijo (recuperar siempre y luego generar) se mantiene en el notebook como línea base.

### 3.6. Función `get_weather`

- **Esquema JSON** con `fecha` (obligatoria, `YYYY-MM-DD`) y `location` (opcional, por defecto Santa Cruz de Tenerife). El tiempo en Tenerife varía mucho entre norte, sur y cumbres, y la guía condiciona varios planes a él: ver el atardecer en Benijo *"si el día está despejado"* o subir al Teide de noche *"cuando esté despejado"*.
- **API real (Open-Meteo)**, sin clave, con el bloque de previsión diaria.
- **Destinos de la guía con coordenadas fijas.** El Teide, Anaga, Benijo y Punta de Teno no son municipios y el geocodificador no los resuelve; el resto de lugares se geocodifica y se comprueba que caiga dentro de Tenerife.
- **Log en fichero:** cada intento de llamada se añade como una línea JSON (hora, herramienta, argumentos, éxito y salida resumida) a `logs/tool_calls.log`.
- **Validación en dos niveles:** `jsonschema` comprueba tipos y campos obligatorios; la función comprueba formato de fecha, que no sea pasada, que esté dentro de los 16 días de previsión y que la ubicación esté en Tenerife.
- **Errores estructurados:** cualquier excepción se devuelve al modelo como `{"ok": false, "error": ...}` para que la explique sin inventar datos, y queda registrada en el log.
- **Salida compacta:** temperaturas, probabilidad de lluvia, viento y el código meteorológico WMO que devuelve Open-Meteo.
- **Fecha actual en las instrucciones.** Sin ella, el modelo no puede convertir *"el sábado"* en una fecha.

Los esquemas no incluyen `additionalProperties` porque Gemini no lo admite.

### 3.7. Historial multiturno

- **Solo se guardan pregunta y respuesta final.** Los chunks recuperados y las salidas de herramientas viven dentro del turno. El historial crece mucho más despacio y nunca queda un mensaje de herramienta huérfano al recortar.
- **Recorte por tokens estimados con `tiktoken`**, eliminando turnos completos desde el más antiguo. La ventana de Gemini es muy grande; el límite responde al coste y a la cuota de tokens por minuto.

### 3.8. Streaming y coordinación de herramientas

- **Streaming:** `model.stream()` imprime el texto según llega; los fragmentos se suman para reconstruir el mensaje completo, incluidas las llamadas a herramientas.
- **Varias herramientas:** el modelo elige entre la guía y el tiempo y puede usar ambas en el mismo turno (por ejemplo, *"quiero ir a Masca el sábado, ¿qué dice la guía y qué tiempo hará?"*). El bucle se programa de forma explícita para poder observar y registrar cada paso.

### 3.9. Experiencia de usuario

- **Chat interactivo** al final del notebook, con historial y streaming: muestra ejemplos de preguntas al empezar, permite reiniciar la conversación con `reset` y terminar con `salir`.
- **Mensajes de error comprensibles:** los errores de las herramientas (fecha fuera de rango, lugar fuera de Tenerife) llegan al modelo de forma estructurada para que los explique en lenguaje natural, y si falla la llamada al modelo el chat muestra un aviso con la causa probable en lugar de detenerse.
- **Citas visibles** en cada dato que sale de la guía y una traza breve de las herramientas usadas en cada turno.

## 4. Resultados

### 4.1. Retrieval

18 consultas formuladas, siempre que es posible, sin las palabras exactas de la guía. hit@k es la proporción de consultas con algún chunk relevante entre los `k` primeros.

| Configuración | Chunks | hit@1 | hit@3 | hit@5 |
| --- | --- | --- | --- | --- |
| paginas 500/80 | 47 | 61% (11/18) | 89% (16/18) | 94% (17/18) |
| secciones 1000/150 | 24 | 78% (14/18) | 100% (18/18) | 100% (18/18) |

El chunking por secciones supera a la estrategia por páginas en todas las métricas: recupera el fragmento correcto en primera posición en el 78% de las consultas, frente al 61%, y lo sitúa siempre entre los tres primeros. Como el asistente recupera cuatro fragmentos por búsqueda (`RAG_K = 4`), en este conjunto el contexto que recibe el modelo siempre incluye la respuesta. La mejora es coherente con el análisis del documento: los fragmentos por página que solo contienen pies de foto pierden la referencia al lugar, mientras que la cabecera de zona y lugar la conserva.

Dos matices limitan la conclusión. Primero, la comparación cambia a la vez la estrategia y el tamaño del chunk (24 fragmentos frente a 47); con chunks más grandes, los cinco primeros resultados cubren una parte mayor de la guía y es más probable que contengan la palabra clave, así que parte de la mejora puede deberse al tamaño. Segundo, con 18 consultas cada acierto equivale a 5,6 puntos, por lo que las diferencias deben leerse como una tendencia.

### 4.2. Asistente completo

| Métrica | Valor |
| --- | --- |
| Selección correcta de herramienta | 89% (8/9) |
| Palabra clave esperada en la respuesta | 100% (8/8) |
| Latencia media por turno | 5,95 s |

La palabra clave se evalúa sobre 8 casos porque el saludo no tiene una palabra clave esperada.

El único caso marcado como fallo es la consulta del tiempo en La Orotava a 40 días. Además de `get_weather`, que devolvió el error de fecha fuera de rango, el modelo consultó la guía, presumiblemente para ofrecer una alternativa como indican sus instrucciones. La respuesta fue correcta, por lo que se trata de una llamada adicional que no empeora el resultado, aunque sí aumenta el coste y la latencia. La métrica es estricta: exige exactamente las herramientas esperadas.

La latencia crece con el número de llamadas al modelo: 2,8 s para el saludo, que no usa herramientas; entre 4 y 8 s para las consultas con una herramienta; y 8,4 s para la consulta que combina guía y previsión. La comprobación por palabra clave es laxa: verifica que la respuesta contenga el dato o la mención esperada, no que todo su contenido sea correcto.

### 4.3. Function calling

En la evaluación, `get_weather` se invocó en tres consultas. En la previsión para mañana en Puerto de la Cruz y en la consulta combinada sobre Masca, la llamada fue correcta y el modelo calculó la fecha a partir de expresiones relativas (*"mañana"*, *"el sábado"*). En la consulta a 40 días, la herramienta devolvió un error estructurado de fecha fuera de rango y el modelo lo explicó al usuario sin inventar datos meteorológicos. En la consulta combinada, el modelo coordinó las dos herramientas en el mismo turno.

### 4.4. Casos límite

Ante la pregunta por el precio de la entrada al Siam Park, un dato que la guía no contiene, el asistente indicó que no tenía información suficiente en lugar de inventar un precio. Ante una fecha fuera del horizonte de previsión, explicó la limitación en lugar de responder con datos ficticios.

## 5. Limitaciones

- **Evaluación aproximada.** hit@k y `keyword_hit` se basan en palabras clave: una respuesta puede contenerlas y ser incorrecta. Los conjuntos (18 consultas de retrieval, 9 casos del asistente) sirven para comparar configuraciones, no para estimar tasas con precisión. No se evalúan expresiones de fecha ambiguas, preguntas en otros idiomas ni intentos de inyección de instrucciones.
- **No determinismo.** El modelo fija sus parámetros de muestreo, así que no es posible reducir la variabilidad bajando la temperatura; la selección de herramientas puede variar entre ejecuciones.
- **Tokens estimados.** `tiktoken` no es el tokenizador de Gemini; el recorte de memoria es aproximado.
- **Memoria sin contexto recuperado.** Si el usuario pide un detalle de una respuesta antigua, el asistente vuelve a buscar en la guía.
- **Contenido de la guía.** No incluye precios ni horarios y algunas indicaciones son tentativas (*"creo que actualmente…"*); el asistente solo puede repetir lo que dice.
- **Citas por sección.** Cada chunk cita el rango de páginas de su sección, no la página exacta de cada frase.
- **Chunking específico del documento.** Las reglas de estructura (encabezados de zona y viñetas) dependen de esta guía; otro documento necesitaría ajustarlas.
- **Dependencias externas.** La capa gratuita de Gemini (20 peticiones diarias por modelo) no permite ejecutar el notebook completo, lo que obliga a activar la facturación; Open-Meteo solo ofrece 16 días de previsión y las coordenadas de los destinos que no son municipios son aproximadas.
- **Interfaz.** El chat vive dentro del notebook; no hay interfaz web.

## 6. Mejoras futuras

- **Evaluación más fiel:** etiquetar a mano el chunk correcto de cada pregunta, ampliar los conjuntos y añadir un LLM como juez de fidelidad a la fuente.
- **Recuperación híbrida y reranking:** combinar búsqueda semántica con búsqueda por palabras clave para nombres propios (restaurantes, playas, miradores).
- **Filtros por metadatos:** aprovechar la zona y el lugar de cada chunk para acotar la búsqueda cuando el usuario menciona una zona concreta.
- **Resumen de turnos antiguos** en lugar de eliminarlos, y contador de tokens del proveedor para el recorte.
- **Interfaz web** con Gradio o Streamlit reutilizando `run_turn`.
- **Observabilidad:** trazas por turno con herramientas, argumentos, tokens y latencia en una herramienta especializada.
