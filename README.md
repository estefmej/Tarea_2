# Tarea 2 – Del documento y el audio a datos estructurados

**Pregunta:** ¿qué tan bien convierten los modelos abiertos (OCR, ASR y un LLM local) documentos y audios
peruanos en datos estructurados confiables, y dónde fallan?

**Dashboard público:** https://tarea2-ocr-asr.streamlit.app

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

Contenido del dashboard: indicadores clave (documentos procesados, minutos de audio, CER y WER por modelo y
concordancia del LLM con la auditoría humana), búsqueda de palabras en los textos OCR y ASR, filtros por tipo de
documento, modelo, condición y duración del audio, cuatro gráficos interactivos de Plotly, tablas descargables en
CSV, un explorador de documentos (texto OCR, campos del LLM, referencia y CER de cada modelo) y un explorador de
audios (transcripción con marcas de hablante, campos del LLM y comparación de WER entre modelos).

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
| Documento | PaddleOCR 3.7 PP-OCRv6 (config. Parte 1, engine transformers, CPU) | CER = 0.312 | – (2.8 s/doc) |
| Documento | Tesseract 5.5 (spa, psm 3) | CER = 0.534 | – (1.1 s/doc) |
| Audio | faster-whisper small (CPU, int8) – línea base | WER = 0.088 | 0.13 |
| Audio | wav2vec2-large-xlsr-53-spanish | WER = 0.261 | 0.31 |

**Decisión.** Para audio elegiría **faster-whisper small**: menor WER, unas 7 veces más rápido que tiempo real
en CPU, y añade puntuación y mayúsculas. wav2vec2 (CTC, sin modelo de lenguaje) une palabras ("corazónconoces")
y falla más con nombres propios. Para documentos elegiría **PaddleOCR**: menor CER en los tres tipos (foto 0.31,
impreso 0.45, manuscrito 0.07), a cambio de ser unas 2.6 veces más lento que Tesseract (2.8 s frente a 1.1 s por documento).

**Casos difíciles.** El peor clip fue clip_003 (entrevista con varios hablantes): WER 0.27 con Whisper y 0.70
con wav2vec2, por el cambio de hablante y los nombres propios. En documentos, el peor tipo fueron los **impresos**
(CER 0.45 y 0.68): periódicos del siglo XIX con tipografía antigua y columnas mezcladas, y billetes con fuentes
decorativas. El peor caso individual fue **doc_007** (foto de la placa "Calle Espaderos"): PaddleOCR leyó las dos
palabras en orden inverso (CER 0.80) y Tesseract no detectó texto (CER 1.0); el CER penaliza el orden de lectura
además del reconocimiento. En el manuscrito (doc_022) solo se evaluó el título, con letra grande: PaddleOCR lo leyó casi completo
(CER 0.07) y Tesseract no (0.62); el cuerpo del manuscrito no se evaluó.

### Extracción con LLM y auditoría

Modelo `qwen2.5:1.5b` (Ollama local, temperatura 0, salida JSON). Campos: `modalidad`, `tema` y `resumen`
siempre; en documentos `emisor`, `fecha` y `monto` solo si aparecen; en audios `numero_hablantes` y
`citas_clave` si se pueden sustentar; `null` para datos ausentes. El código descarta cualquier campo no
permitido.

Auditoría humana de 20 extracciones al azar (10 documentos + 10 audios): **7/20 completamente correctas (35 %)**,
3/10 en documentos y 4/10 en audios. Criterio: `numero_hablantes` en null no se penaliza, pero sí un número inventado,
igual que omitir emisor, fecha o monto cuando aparecen literalmente. Errores típicos: omite campos presentes
("Municipalidad de Lima", "El Comercio", "Banco del Perú"), alucina contenido cuando el OCR es ilegible
(manuscritos doc_021 y doc_028), confunde entrevistador y entrevistado, e inventa `numero_hablantes`.

## Hallazgos principales

1. **Audio:** faster-whisper small supera a wav2vec2 en precisión (WER 0.088 frente a 0.261) y en velocidad
   (RTF 0.13 frente a 0.31). La brecha crece en la entrevista con varios hablantes (0.27 frente a 0.70): sin modelo
   de lenguaje, wav2vec2 une palabras y escribe por sonido.
2. **Documentos y LLM:** PaddleOCR supera a Tesseract en fotos e impresos, pero ambos fallan en manuscritos. Cuando
   el OCR es ilegible, el LLM no devuelve null: alucina contenido (fechas, descripciones), y solo [X]/20
   extracciones fueron completamente correctas.

## Privacidad y ética

- Los nombres reales detectados se seudonimizan (`PERSONA_1`, `PERSONA_2`, ...) en CSV, métricas, extracciones del
  LLM y salidas, incluidos entrevistados, personas históricas y las variantes mal transcritas por los ASR.
- Antes de publicar se eliminaron del notebook las salidas con audio incrustado y la lista de nombres
  candidatos de la Parte 1.
- No se versionan audios originales ni pesos de modelos (`.gitignore`). El dashboard solo muestra imágenes o
  audios con permiso de redistribución y sin datos personales; las entrevistas se consultan únicamente como texto.

## Limitaciones

- Fragmentos de 15 s por clip: con 30 a 40 palabras por clip, un par de errores mueve mucho el WER.
- El corpus de audio es casi todo "limpio" (9 de 10 clips de la muestra); no incluye ruido ni música, así que
  el análisis de casos difíciles en audio se limita a varios hablantes.
- Muestras pequeñas: 10 casos por modalidad y un solo manuscrito (evaluado solo en su título), así que las cifras
  por grupo son orientativas.
- Las cifras escritas con números frente a palabras ("70" / "setenta") cuentan como error.
- La seudonimización se aplica antes del WER, así que un nombre mal reconocido no se cuenta como error.

## Uso de IA

Se usó un asistente de IA (Claude) para apoyar el desarrollo; las capturas completas están en
`evidencias_ia/`. Las transcripciones de referencia, las etiquetas de condición y la auditoría del LLM se
hicieron manualmente.
