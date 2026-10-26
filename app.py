"""Tarea 2 – Parte 3: dashboard de resultados (solo lee los CSV generados por los notebooks)."""
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Documentos y audios peruanos a datos", page_icon="📜", layout="wide")

DATOS = Path(__file__).parent / "datos"
EJEMPLOS = DATOS / "ejemplos"   # solo archivos con permiso de redistribución y sin datos personales


@st.cache_data
def leer(nombre):
    ruta = DATOS / nombre
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


docs = leer("documentos.csv")
aud = leer("transcripciones.csv")
res = leer("resultados_eval.csv")
llm = leer("extraccion_llm.csv")
audit = leer("auditoria_llm.csv")
fuentes = leer("fuentes.csv")


def ejemplo_publicable(id_, extensiones):
    for ext in extensiones:
        ruta = EJEMPLOS / f"{id_}{ext}"
        if ruta.exists():
            return ruta
    return None


def campos_llm(id_):
    if llm.empty:
        return None
    fila = llm[llm["id"] == id_]
    return json.loads(fila.iloc[0]["campos_json"]) if len(fila) else None


def con_hablantes(texto):
    return re.sub(r"\[(HABLANTE_\d+)\]", r"\n\n**\1:**", str(texto)).strip()


def contiene(serie, consulta):
    if not consulta:
        return pd.Series(True, index=serie.index)
    return serie.fillna("").astype(str).str.contains(re.escape(consulta), case=False)


def columnas(df, nombres):
    return [c for c in nombres if c in df.columns]


# ---------------- Encabezado ----------------
st.title("¿Qué tan bien convierten los modelos abiertos documentos y audios peruanos en datos estructurados?")
st.write(
    "Se extrajo texto de documentos (impresos, fotos y manuscritos) con OCR y de audios con ASR, se midió el error "
    "contra transcripciones manuales (CER y WER) comparando dos modelos por modalidad, y se extrajeron campos "
    "estructurados con un LLM local cuya salida se auditó a mano. Todos los textos están seudonimizados."
)

# ---------------- Filtros ----------------
st.sidebar.header("Filtros")
consulta = st.sidebar.text_input("Buscar palabras en textos OCR y ASR")
tipos = sorted(docs["tipo"].dropna().unique()) if "tipo" in docs else []
sel_tipos = st.sidebar.multiselect("Tipo de documento", tipos, default=tipos)
modelos = sorted(res["modelo"].dropna().unique()) if not res.empty else []
sel_modelos = st.sidebar.multiselect("Modelo evaluado", modelos, default=modelos)
condiciones = sorted(aud["condicion"].dropna().unique()) if "condicion" in aud else []
sel_cond = st.sidebar.multiselect("Condición del audio", condiciones, default=condiciones)
if "duracion_seg" in aud and len(aud):
    dmin, dmax = float(aud["duracion_seg"].min()), float(aud["duracion_seg"].max())
    rango = st.sidebar.slider("Duración del audio (s)", dmin, dmax, (dmin, dmax))
else:
    rango = None

docs_f = docs[docs["tipo"].isin(sel_tipos) & contiene(docs["texto_ocr"], consulta)] if len(docs) else docs
mask_aud = contiene(aud["texto_asr"], consulta) if len(aud) else pd.Series(dtype=bool)
if condiciones:
    mask_aud &= aud["condicion"].isin(sel_cond)
if rango:
    mask_aud &= aud["duracion_seg"].between(*rango)
aud_f = aud[mask_aud] if len(aud) else aud
res_f = res[res["modelo"].isin(sel_modelos)] if len(res) else res
if len(res_f):   # los gráficos de error también respetan la búsqueda, el tipo y los filtros de audio
    res_f = res_f[res_f["id"].isin(set(docs_f["id"]) | set(aud_f["id"]))]

# ---------------- Indicadores ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Documentos procesados", len(docs))
c2.metric("Minutos de audio transcritos", f"{aud['duracion_seg'].sum() / 60:.1f}" if len(aud) else "–")
if len(audit) and "correcto" in audit:
    c3.metric("Concordancia LLM vs. auditoría humana",
              f"{audit['correcto'].mean():.0%}", f"{int(audit['correcto'].sum())} de {len(audit)} casos",
              delta_color="off")

if len(res_f):
    resumen = res_f.groupby(["modalidad", "modelo", "metrica"])["valor"].mean().reset_index()
    cols = st.columns(len(resumen))
    for col, (_, r) in zip(cols, resumen.iterrows()):
        muestra = "10 documentos" if r["modalidad"] == "documento" else "10 clips"
        col.metric(f"{r['metrica']} – {r['modelo']}", f"{r['valor']:.3f}", muestra, delta_color="off")

tab_res, tab_doc, tab_aud, tab_met = st.tabs(["Resultados", "Documentos", "Audios", "Fuentes y método"])

# ---------------- Resultados ----------------
with tab_res:
    g1, g2 = st.columns(2)
    for col, (mod, met) in zip((g1, g2), (("documento", "CER"), ("audio", "WER"))):
        datos = res_f[res_f["modalidad"] == mod].groupby(["grupo", "modelo"])["valor"].mean().reset_index()
        if len(datos):
            fig = px.bar(datos, x="grupo", y="valor", color="modelo", barmode="group",
                         labels={"valor": f"{met} medio", "grupo": "Tipo" if mod == "documento" else "Condición"},
                         title=f"{met} por {'tipo de documento' if mod == 'documento' else 'condición de audio'}")
            col.plotly_chart(fig, width="stretch")
    g3, g4 = st.columns(2)
    if len(docs_f):
        fig = px.histogram(docs_f, x="tipo", color="tipo", title="Documentos por tipo")
        g3.plotly_chart(fig, width="stretch")
    if len(aud_f):
        fig = px.histogram(aud_f, x="duracion_seg", nbins=15, color="condicion" if condiciones else None,
                           labels={"duracion_seg": "Duración (s)"}, title="Duración de los audios")
        g4.plotly_chart(fig, width="stretch")

# ---------------- Documentos ----------------
with tab_doc:
    st.subheader(f"{len(docs_f)} documentos con los filtros actuales")
    tabla = docs_f[columnas(docs_f, ["id", "tipo", "fuente", "modelo_ocr", "texto_ocr"])]
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.download_button("Descargar tabla de documentos (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                       "documentos_filtrados.csv", "text/csv")

    if len(docs_f):
        sel = st.selectbox("Elige un documento", docs_f["id"].tolist())
        fila = docs_f[docs_f["id"] == sel].iloc[0]
        izq, der = st.columns(2)
        with izq:
            st.markdown(f"**Tipo:** {fila.get('tipo', '–')}  \n**Fuente:** {fila.get('fuente', '–')}")
            imagen = ejemplo_publicable(sel, (".jpg", ".jpeg", ".png"))
            if imagen:
                st.image(str(imagen), width="stretch")
            else:
                st.caption("Imagen no publicada (licencia o datos personales); se consulta como texto.")
            st.text_area("Texto OCR", str(fila.get("texto_ocr", "")), height=300, disabled=True)
        with der:
            campos = campos_llm(sel)
            st.markdown("**Campos extraídos por el LLM**")
            if campos:
                st.json(campos)
            else:
                st.caption("Sin extracción para este documento.")
            ref = fila.get("texto_referencia")
            if isinstance(ref, str) and ref.strip():
                st.markdown("**Texto de referencia (transcripción manual)**")
                st.write(ref)
                filas = res[(res["id"] == sel) & (res["metrica"] == "CER")]
                for _, r in filas.iterrows():
                    st.metric(f"CER – {r['modelo']}", f"{r['valor']:.3f}")
                    with st.expander(f"Lectura de {r['modelo']}"):
                        st.write(r["hipotesis"])
            else:
                st.caption("Este documento no forma parte de la muestra de evaluación (sin CER).")
    else:
        st.info("Ningún documento coincide con la búsqueda. Prueba otra palabra o amplía los filtros.")

# ---------------- Audios ----------------
with tab_aud:
    st.subheader(f"{len(aud_f)} clips con los filtros actuales")
    tabla = aud_f[columnas(aud_f, ["id", "condicion", "duracion_seg", "idioma", "n_hablantes", "modelo_asr", "texto_asr"])]
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.download_button("Descargar tabla de audios (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                       "audios_filtrados.csv", "text/csv")

    if len(aud_f):
        sel = st.selectbox("Elige un clip", aud_f["id"].tolist())
        fila = aud_f[aud_f["id"] == sel].iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Duración", f"{float(fila.get('duracion_seg', 0)):.0f} s")
        m2.metric("Hablantes detectados", str(fila.get("n_hablantes", "–")))
        m3.metric("Condición", str(fila.get("condicion", "–")))
        audio = ejemplo_publicable(sel, (".wav", ".mp3", ".ogg"))
        if audio:
            st.audio(str(audio))
        else:
            st.caption("Audio no publicado (licencia o voz de personas); se consulta como texto.")
        izq, der = st.columns(2)
        with izq:
            st.markdown("**Transcripción ASR**")
            with st.container(height=350):
                st.markdown(con_hablantes(fila.get("texto_asr", "")))
        with der:
            campos = campos_llm(sel)
            st.markdown("**Campos extraídos por el LLM**")
            if campos:
                st.json(campos)
            else:
                st.caption("Sin extracción para este clip.")
        filas = res[(res["id"] == sel) & (res["metrica"] == "WER")]
        if len(filas):
            st.markdown("**Comparación de modelos sobre los primeros 15 s**")
            st.write("Referencia manual: " + str(fila.get("texto_referencia", "")))
            st.dataframe(filas[["modelo", "valor", "rtf", "hipotesis"]].rename(
                columns={"valor": "WER", "rtf": "RTF", "hipotesis": "transcripción"}),
                width="stretch", hide_index=True)
    else:
        st.info("Ningún clip coincide con la búsqueda. Prueba otra palabra o amplía los filtros.")

# ---------------- Fuentes y método ----------------
with tab_met:
    st.markdown(
        "- **Muestra de evaluación:** 10 documentos (impreso, foto, manuscrito) y 10 clips, con transcripción manual. "
        "En doc_001, doc_002 y doc_022 se evaluó un recorte; en audio, los primeros 15 s de cada clip.\n"
        "- **Modelos:** PaddleOCR PP-OCRv6 frente a Tesseract 5.5; faster-whisper small frente a wav2vec2 XLSR español.\n"
        "- **LLM:** qwen2.5:1.5b (Ollama), salida JSON; auditoría humana de 20 extracciones.\n"
        "- **Privacidad:** nombres reales reemplazados por `PERSONA_n`; imágenes y audios solo se muestran si están "
        "en `datos/ejemplos/` con permiso de redistribución."
    )
    if len(fuentes):
        st.subheader("Fuentes y licencias")
        st.dataframe(fuentes, width="stretch", hide_index=True)
    if len(audit):
        st.subheader("Auditoría humana del LLM")
        st.dataframe(audit[columnas(audit, ["id", "modalidad", "correcto", "error"])],
                     width="stretch", hide_index=True)
