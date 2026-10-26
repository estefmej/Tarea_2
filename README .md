# Tarea 2 – Del documento y el audio a datos estructurados

**Pregunta:** ¿qué tan bien convierten los modelos abiertos (OCR, ASR y un LLM local) documentos y audios
peruanos en datos estructurados confiables, y dónde fallan?

**Dashboard público:** [completar con el enlace de share.streamlit.io]

Pipeline de extracción (OCR + ASR), evaluación con referencias manuales (CER/WER), comparación de modelos y
extracción estructurada con un LLM local, sobre un corpus peruano de documentos y audios.

## Estructura del repositorio

```
├── 01_extraccion.ipynb      # Parte 1: corpus, OCR, transcripción, limpieza y EDA
├── 02_evaluacion.ipynb      # Parte 2: CER/WER, comparación de modelos, LLM y auditoría
├── app.py                   # Parte 3: dashboard Streamlit (solo lee los CSV)
├── requirements.txt         # Dependencias del dashboard (streamlit, pandas, plotly)
├── requirements-notebooks.txt  # Dependencias exactas para reejecutar los notebooks
├── README.md                # Este reporte
├── .gitignore               # Excluye datos/raw/ (audios, imágenes originales) y pesos de modelos
├── evidencias_ia/           # Capturas de la conversación con el asistente de IA (01.png, 02.png, ...)
└── datos/
    ├── documentos.csv       # Corpus OCR (25 documentos)
    ├── transcripciones.csv  # Corpus ASR (22 clips) + columna adicional `condicion`
    ├── fuentes.csv          # Fuente, licencia y acceso a cada archivo
    ├── atribucion.csv       # Atribución requerida por las licencias
    ├── muestra_eval.csv     # Ids de la muestra de evaluación (10 documentos + 10 clips)
    ├── resultados_eval.csv  # CER/WER, tiempo y RTF por caso y modelo
    ├── extraccion_llm.csv   # Campos extraídos por el LLM (id, modalidad, campos_json, modelo_llm, fecha)
    └── auditoria_llm.csv    # Auditoría humana de 20 extracciones
```

## Fuentes y licencias

Cada documento y audio tiene su URL, licencia y procedimiento de recuperación en `datos/fuentes.csv`, y la
atribución correspondiente en `datos/atribucion.csv`. Los archivos originales (`datos/raw/`) no se versionan
por tamaño y licencia; se recuperan siguiendo `fuentes.csv`.

Fecha de extracción del corpus: [completar, p. ej. octubre de 2026]; el detalle por archivo está en
`datos/fuentes.csv`.

## Cómo ejecutar

### Dashboard (Parte 3)

```bash
pip install -r requirements.txt
streamlit run app.py
```

El dashboard no ejecuta OCR, ASR ni LLM: muestra los resultados precalculados en `datos/*.csv`. Las imágenes y
audios solo se muestran si están en `datos/ejemplos/` con permiso de redistribución y sin datos personales; el
resto se consulta como texto y metadatos.

### Notebooks (Partes 1 y 2)

```bash
pip install -r requirements-notebooks.txt
```

Requisitos externos (no se instalan con pip):

- **Tesseract OCR 5.5** para Windows (UB Mannheim). El notebook descarga el modelo `spa.traineddata` del
  repositorio oficial `tesseract-ocr/tessdata` en `datos/raw/tessdata/`.
- **Ollama** con el modelo `qwen2.5:1.5b` (`ollama pull qwen2.5:1.5b`).

Luego ejecutar en orden `01_extraccion.ipynb` y `02_evaluacion.ipynb`. Los notebooks se entregan ya ejecutados,
con salidas visibles en GitHub. Entorno usado: Python 3.14.7, CPU (sin GPU).

## Parte 2 – Evaluación

### Muestra de evaluación

- **Documentos (10):** doc_001, doc_002, doc_004, doc_005, doc_006, doc_007, doc_008, doc_017, doc_018,
  doc_022, con los tres tipos (impreso, foto y manuscrito). En **doc_001, doc_002 y doc_022** se evaluó un
  **recorte** de la imagen (horarios, primer párrafo y título del manuscrito, respectivamente), porque la
  transcripción manual cubre solo ese fragmento; ambos OCR leen exactamente el mismo recorte.
- **Audio (10 clips):** clip_003, 006, 007, 009, 012, 014, 015, 016, 017, 020. Se evaluaron los **primeros 15 s**
  de cada clip (150 s en total), con autorización del profesor. Ambos ASR procesan el mismo fragmento.
- Las referencias (`texto_referencia`) se transcribieron manualmente mirando la imagen o escuchando el audio.
  Las configuraciones de los modelos se fijaron antes de calcular las métricas.

### Normalización

Referencia e hipótesis se normalizan igual: minúsculas, sin puntuación, sin marcas `[HABLANTE_X]` y con los
nombres reales reemplazados por `PERSONA_1` antes de calcular CER/WER.

### Resultados

| Modalidad | Modelo | Error medio | RTF medio |
|---|---|---|---|
| Documento | PaddleOCR 3.7 PP-OCRv6 (config. Parte 1, engine transformers, CPU) | CER = [completar] | – |
| Documento | Tesseract 5.5 (spa, psm 3) | CER = [completar] | – |
| Audio | faster-whisper small (CPU, int8) – línea base | WER = 0.088 | 0.13 |
| Audio | wav2vec2-large-xlsr-53-spanish | WER = 0.261 | 0.29 |

**Decisión.** Para audio elegiría **faster-whisper small**: menor WER, unas 7 veces más rápido que tiempo real
en CPU, y añade puntuación y mayúsculas. wav2vec2 (CTC, sin modelo de lenguaje) une palabras ("corazónconoces")
y falla más con nombres propios. Para documentos: [completar con el modelo de menor CER y su costo/latencia].

**Casos difíciles.** El peor clip fue clip_003 (entrevista con varios hablantes): WER 0.27 con Whisper y 0.70
con wav2vec2, por el cambio de hablante y los nombres propios. En documentos: [completar: peor tipo, por
ejemplo el manuscrito doc_022, y la causa].

### Extracción con LLM y auditoría

Modelo `qwen2.5:1.5b` (Ollama local, temperatura 0, salida JSON). Campos: `modalidad`, `tema` y `resumen`
siempre; en documentos `emisor`, `fecha` y `monto` solo si aparecen; en audios `numero_hablantes` y
`citas_clave` si se pueden sustentar; `null` para datos ausentes. El código descarta cualquier campo no
permitido.

Auditoría humana de 20 extracciones al azar (10 documentos + 10 audios): **[X]/20 completamente correctas**.
Errores típicos: [completar].

## Hallazgos principales

1. **Audio:** faster-whisper small supera a wav2vec2 en precisión (WER 0.088 frente a 0.261) y en velocidad
   (RTF 0.13 frente a 0.31). La brecha crece en la entrevista con varios hablantes (0.27 frente a 0.70): sin modelo
   de lenguaje, wav2vec2 une palabras y escribe por sonido.
2. **Documentos y LLM:** PaddleOCR supera a Tesseract en fotos e impresos, pero ambos fallan en manuscritos. Cuando
   el OCR es ilegible, el LLM no devuelve null: alucina contenido (fechas, descripciones), y solo [X]/20
   extracciones fueron completamente correctas.

## Privacidad y ética

- Los nombres reales detectados se seudonimizan (`PERSONA_1`) en CSV, métricas y salidas. También se aplica a
  las variantes mal transcritas por los ASR.
- Antes de publicar se eliminaron del notebook las salidas con audio incrustado y la lista de nombres
  candidatos de la Parte 1.
- No se versionan audios originales ni pesos de modelos (`.gitignore`).

## Limitaciones

- Fragmentos de 15 s por clip: con 30 a 40 palabras por clip, un par de errores mueve mucho el WER.
- El corpus de audio es casi todo "limpio" (9 de 10 clips de la muestra); no incluye ruido ni música, así que
  el análisis de casos difíciles en audio se limita a varios hablantes.
- Las cifras escritas con números frente a palabras ("70" / "setenta") cuentan como error.
- La seudonimización se aplica antes del WER, así que un nombre mal reconocido no se cuenta como error.

## Uso de IA

Se usó un asistente de IA (Claude) para apoyar el desarrollo; las capturas completas están en
`evidencias_ia/`. Las transcripciones de referencia, las etiquetas de condición y la auditoría del LLM se
hicieron manualmente.
