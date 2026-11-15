"""
Alerta temprana de deserción estudiantil con redes neuronales
App de Streamlit · ACA 2 · Aprendizaje Automático · Especialización en IA · CUN · Grupo 7

Carga el ensamble de redes exportado a ONNX (GRU híbrida, MLP profundo y
FT-Transformer), el codificador entrenado y la base SQLite del proyecto.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nucleo import codificador as cdf
from nucleo import preprocesamiento as pp

RAIZ = Path(__file__).parent
DB = RAIZ / "data" / "desercion.db"
MOD = RAIZ / "modelo"
CLASES = pp.ORDEN_CLASES
COLORES = {"Deserción": "#C0392B", "Matriculado": "#E08A1E", "Graduado": "#1A4F8B"}
ETAPAS = cdf.ETAPAS
ENLACES = json.loads((RAIZ / "enlaces.json").read_text(encoding="utf-8")) if (RAIZ / "enlaces.json").exists() else {}

st.set_page_config(page_title="Alerta temprana de deserción · Redes neuronales", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.tarjeta {border-radius: 14px; padding: 16px 18px; border: 1px solid #E3E8EC; background: #FFFFFF;}
.kpi {font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
.kpi-l {font-size: .82rem; color: #5B6770;}
.riesgo {border-radius: 14px; padding: 18px 20px; color: white;}
.riesgo h3 {margin: 0 0 4px 0; color: white;}
.nota {font-size: .82rem; color: #5B6770;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- recursos
@st.cache_resource
def cargar_recursos():
    meta = json.loads((MOD / "metadatos.json").read_text(encoding="utf-8"))
    cod = cdf.CodificadorTabular.cargar(MOD / "codificador.json")
    sesiones = {nombre: ort.InferenceSession(str(MOD / archivo), providers=["CPUExecutionProvider"])
                for nombre, archivo in meta["archivos"].items()}
    return meta, cod, sesiones


@st.cache_data
def consultar(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def consulta_libre(sql: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


META, COD, SESIONES = cargar_recursos()


def grafico(fig):
    fig.update_layout(separators=",.", font=dict(family="sans-serif"))
    st.plotly_chart(fig, use_container_width=True)
PRINCIPAL = META["modelo_principal"]
REF = META["referencia"]


def predecir(df: pd.DataFrame, etapa, modelo: str = PRINCIPAL) -> np.ndarray:
    t = COD.transformar(df, etapa)
    return SESIONES[modelo].run(None, {k: t[k] for k in ("cat", "est", "seq", "mask")})[0]


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def porc(x, d=1):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def tabla_es(df, d=3):
    """Tabla con coma decimal."""
    return df.style.format(precision=d, decimal=",", thousands=".")


def nivel_riesgo(p_des: float) -> tuple[str, str]:
    if p_des >= 0.60:
        return "Alto", "#C0392B"
    if p_des >= 0.35:
        return "Medio", "#E08A1E"
    return "Bajo", "#3F8F3A"


def registrar(fila: dict) -> bool:
    try:
        con = sqlite3.connect(DB)
        con.execute("""CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT DEFAULT (datetime('now', 'localtime')),
            origen TEXT, etapa TEXT, prob_desercion REAL, prob_matriculado REAL, prob_graduado REAL,
            clase TEXT, nivel_riesgo TEXT, entrada_json TEXT)""")
        cols = ", ".join(fila)
        con.execute(f"INSERT INTO predicciones ({cols}) VALUES ({', '.join('?' * len(fila))})", tuple(fila.values()))
        con.commit()
        con.close()
        return True
    except sqlite3.Error:
        return False


def metrica(meta_prueba, modelo, etapa, clave):
    for r in meta_prueba:
        if r["modelo"] == modelo and r["etapa"] == ETAPAS[etapa]:
            return r[clave]
    return None


# --------------------------------------------------------------------------- barra lateral
with st.sidebar:
    if (RAIZ / "assets" / "logo_cun.png").exists():
        st.image(str(RAIZ / "assets" / "logo_cun.png"), use_container_width=True)
    st.markdown("### Alerta temprana de deserción")
    st.caption("ACA 2 · Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN")
    modelo_sel = st.selectbox("Red neuronal", list(META["archivos"]), index=list(META["archivos"]).index(PRINCIPAL),
                              help="La GRU híbrida es el modelo principal. Las otras dos se incluyen para comparar.")
    st.markdown("**Grupo 7**  \nBonny Darhyll Brayan Galindo Santos  \nAnternol de los Reyes Bedoya Naranjo  \n"
                "Andrés Alberto López Herrera  \nCamilo José Rodríguez Ayubi")
    st.caption("Docente: Dr. Hamilton Rivera Flor · Noviembre de 2026")
    if ENLACES.get("repo"):
        st.markdown(f"[Repositorio en GitHub]({ENLACES['repo']})")
    if ENLACES.get("video"):
        st.markdown(f"[Video explicativo]({ENLACES['video']})")

st.title("🎓 ¿Quién está en riesgo de abandonar sus estudios?")
st.markdown("Una red neuronal recurrente que lee la trayectoria de cada estudiante **semestre a semestre** y estima, "
            "desde el día de la matrícula, la probabilidad de que deserte, siga matriculado o se gradúe.")

m2 = {k: metrica(META["prueba"], PRINCIPAL, 2, k) for k in ("Exactitud", "F1 macro", "ROC-AUC (OvR macro)",
                                                             "Sensibilidad Deserción")}
k1, k2, k3, k4 = st.columns(4)
for col, val, lab in [(k1, porc(m2['Exactitud']), "exactitud en 885 estudiantes nunca vistos"),
                      (k2, num(m2['F1 macro']), "F1 macro (equilibrio entre las 3 clases)"),
                      (k3, num(m2['ROC-AUC (OvR macro)']), "ROC-AUC: capacidad de ordenar por riesgo"),
                      (k4, porc(m2['Sensibilidad Deserción']), "de quienes desertaron fueron detectados")]:
    col.markdown(f"<div class='tarjeta'><div class='kpi'>{val}</div><div class='kpi-l'>{lab}</div></div>",
                 unsafe_allow_html=True)
st.write("")

tab_ind, tab_lote, tab_desemp, tab_como, tab_sql = st.tabs(
    ["🎯 Evaluar un estudiante", "📋 Evaluación por lotes", "📊 Desempeño del modelo", "🧠 Cómo funciona",
     "🗄️ Base de datos SQL"])

# --------------------------------------------------------------------------- formulario
NOMBRE_PROG = {v: k for k, v in pp.PROGRAMAS.items()}
NOMBRE_MODO = {v: k for k, v in pp.MODO_INGRESO.items()}
NOMBRE_CIVIL = {v: k for k, v in pp.ESTADO_CIVIL.items()}
NOMBRE_PREVIA = {v: k for k, v in pp.FORMACION_PREVIA.items()}
NIVELES = list(pp.CODIGO_NIVEL)
OCUPACIONES = [v for v in pp.GRUPO_OCUPACION.values()]
COHORTES = consultar("SELECT * FROM cohortes ORDER BY cohorte_id")

CAMPOS_SEM = [("inscritas", "Unidades inscritas", 0, 26), ("evaluaciones", "Evaluaciones presentadas", 0, 45),
              ("aprobadas", "Unidades aprobadas", 0, 26), ("nota", "Promedio (0-20)", 0.0, 20.0),
              ("convalidadas", "Unidades convalidadas", 0, 20), ("sin_evaluacion", "Unidades sin evaluación", 0, 12)]


def _codigo_a_nivel(c):
    return pp.COD_A_NIVEL.get(int(c), "Media") if pd.notna(c) else "Media"


def cargar_caso(est_id: int):
    f = consultar("SELECT * FROM v_estudiantes_plano WHERE estudiante_id = ?", (est_id,)).iloc[0]
    ss = st.session_state
    ss["programa"] = pp.PROGRAMAS.get(int(f.programa), list(NOMBRE_PROG)[0])
    ss["modo"] = pp.MODO_INGRESO.get(int(f.modo_ingreso), list(NOMBRE_MODO)[0])
    ss["civil"] = pp.ESTADO_CIVIL.get(int(f.estado_civil), "Soltero(a)")
    ss["previa"] = pp.FORMACION_PREVIA.get(int(f.formacion_previa), "Secundaria completa")
    ss["edad"] = int(f.edad_ingreso)
    ss["admision"] = float(f.nota_admision)
    ss["nota_previa"] = float(f.nota_formacion_previa)
    ss["orden"] = int(f.orden_opcion)
    ss["jornada"] = "Diurna" if f.jornada_diurna == 1 else "Nocturna"
    ss["genero"] = "Hombre" if f.genero_masculino == 1 else "Mujer"
    for k in ("matricula_al_dia", "deudor", "becado", "desplazado", "necesidades_especiales", "internacional"):
        ss[k] = bool(f[k])
    ss["edu_madre"] = _codigo_a_nivel(f.educacion_madre)
    ss["edu_padre"] = _codigo_a_nivel(f.educacion_padre)
    ss["ocu_madre"] = pp.grupo_ocupacion(f.ocupacion_madre)
    ss["ocu_padre"] = pp.grupo_ocupacion(f.ocupacion_padre)
    ss["cohorte"] = int(f.cohorte_id)
    ss["info"] = ETAPAS[2]
    for s in (1, 2):
        for c, _, lo, hi in CAMPOS_SEM:
            v = f[f"s{s}_{c}"]
            v = REF[f"s{s}_{c}"] if pd.isna(v) else v
            ss[f"s{s}_{c}"] = float(v) if isinstance(lo, float) else int(round(v))
    ss["real"] = f.resultado


def al_cambiar_caso():
    sel = st.session_state.get("caso_sel")
    if sel and sel in META["casos_ejemplo"]:
        cargar_caso(META["casos_ejemplo"][sel])
    else:
        st.session_state.pop("real", None)


if "programa" not in st.session_state:
    cargar_caso(META["casos_ejemplo"]["Riesgo medio"])
    st.session_state["caso_sel"] = "Riesgo medio"

with tab_ind:
    st.markdown("Completa los datos o carga un **caso real del conjunto de prueba** (estudiantes que el modelo nunca vio).")
    st.selectbox("Cargar un caso de ejemplo", ["Riesgo alto", "Riesgo medio", "Riesgo bajo", "— Datos propios —"],
                 key="caso_sel", on_change=al_cambiar_caso)

    with st.expander("1 · Datos de ingreso y contexto", expanded=False):
        a, b, c = st.columns(3)
        a.selectbox("Programa", list(NOMBRE_PROG), key="programa")
        a.selectbox("Modalidad de ingreso", list(NOMBRE_MODO), key="modo")
        a.slider("Orden de preferencia del programa (1 = primera opción)", 0, 9, key="orden")
        b.slider("Edad al ingresar", 17, 70, key="edad")
        b.slider("Nota de admisión (95-190)", 95.0, 190.0, step=0.1, key="admision")
        b.selectbox("Formación previa", list(NOMBRE_PREVIA), key="previa")
        b.slider("Nota de la formación previa (95-190)", 95.0, 190.0, step=0.1, key="nota_previa")
        c.radio("Jornada", ["Diurna", "Nocturna"], horizontal=True, key="jornada")
        c.radio("Género", ["Mujer", "Hombre"], horizontal=True, key="genero")
        c.selectbox("Estado civil", list(NOMBRE_CIVIL), key="civil")
        d, e, f = st.columns(3)
        d.selectbox("Nivel educativo de la madre", NIVELES, key="edu_madre")
        d.selectbox("Nivel educativo del padre", NIVELES, key="edu_padre")
        e.selectbox("Ocupación de la madre", OCUPACIONES, key="ocu_madre")
        e.selectbox("Ocupación del padre", OCUPACIONES, key="ocu_padre")
        f.checkbox("Vive fuera de su ciudad de origen", key="desplazado")
        f.checkbox("Necesidades educativas especiales", key="necesidades_especiales")
        f.checkbox("Estudiante internacional", key="internacional")
        st.selectbox("Cohorte de ingreso (contexto económico del año)", COHORTES["cohorte_id"].tolist(), key="cohorte",
                     format_func=lambda i: (lambda r: f"Cohorte {i}: desempleo {r.tasa_desempleo:.1f} %, inflación "
                                            f"{r.tasa_inflacion:.1f} %, PIB {r.pib:+.2f}")(
                         COHORTES.set_index("cohorte_id").loc[i]))

    with st.expander("2 · Situación financiera", expanded=True):
        a, b, c = st.columns(3)
        a.checkbox("Matrícula al día", key="matricula_al_dia")
        b.checkbox("Tiene deudas con la institución", key="deudor")
        c.checkbox("Tiene beca", key="becado")

    with st.expander("3 · Trayectoria académica", expanded=True):
        info = st.radio("¿Qué información hay disponible?", list(ETAPAS.values()), horizontal=True, key="info")
        etapa = list(ETAPAS.values()).index(info)
        cols_s = st.columns(2)
        for s in (1, 2):
            with cols_s[s - 1]:
                st.markdown(f"**{'1.er' if s == 1 else '2.º'} semestre**" + ("" if etapa >= s else " · *aún no ocurre*"))
                c1, c2 = st.columns(2)
                for j, (c, lab, lo, hi) in enumerate(CAMPOS_SEM):
                    (c1 if j % 2 == 0 else c2).number_input(lab, min_value=lo, max_value=hi, key=f"s{s}_{c}",
                                                            step=0.1 if isinstance(lo, float) else 1,
                                                            disabled=etapa < s)

    ss = st.session_state
    fila = {
        "estado_civil": NOMBRE_CIVIL[ss.civil], "modo_ingreso": NOMBRE_MODO[ss.modo], "orden_opcion": ss.orden,
        "programa": NOMBRE_PROG[ss.programa], "jornada_diurna": int(ss.jornada == "Diurna"),
        "formacion_previa": NOMBRE_PREVIA[ss.previa], "nota_formacion_previa": ss.nota_previa,
        "educacion_madre": pp.CODIGO_NIVEL[ss.edu_madre], "educacion_padre": pp.CODIGO_NIVEL[ss.edu_padre],
        "ocupacion_madre": pp.CODIGO_OCUPACION[ss.ocu_madre], "ocupacion_padre": pp.CODIGO_OCUPACION[ss.ocu_padre],
        "nota_admision": ss.admision, "desplazado": int(ss.desplazado),
        "necesidades_especiales": int(ss.necesidades_especiales), "deudor": int(ss.deudor),
        "matricula_al_dia": int(ss.matricula_al_dia), "genero_masculino": int(ss.genero == "Hombre"),
        "becado": int(ss.becado), "edad_ingreso": ss.edad, "internacional": int(ss.internacional),
        "sin_registro_curricular": 0,
    }
    coh = COHORTES.set_index("cohorte_id").loc[ss.cohorte]
    fila.update({"tasa_desempleo": coh.tasa_desempleo, "tasa_inflacion": coh.tasa_inflacion, "pib": coh.pib})
    for s in (1, 2):
        for c, *_ in CAMPOS_SEM:
            fila[f"s{s}_{c}"] = ss[f"s{s}_{c}"]
    df_est = pd.DataFrame([fila])

    avisos = []
    for s in range(1, etapa + 1):
        if fila[f"s{s}_aprobadas"] > fila[f"s{s}_inscritas"]:
            avisos.append(f"S{s}: hay más unidades aprobadas que inscritas.")
        if fila[f"s{s}_aprobadas"] > 0 and fila[f"s{s}_nota"] < 10:
            avisos.append(f"S{s}: con unidades aprobadas el promedio debería ser de al menos 10/20.")
    for a in avisos:
        st.warning(a)

    # ---------------- resultado
    prob = predecir(df_est, etapa, modelo_sel)[0]
    p_des = float(prob[0])
    nivel, color = nivel_riesgo(p_des)
    clase = CLASES[int(prob.argmax())]
    st.divider()
    r1, r2 = st.columns([1, 1.4])
    with r1:
        st.markdown(f"<div class='riesgo' style='background:{color}'><h3>Riesgo de deserción: {nivel}</h3>"
                    f"<div style='font-size:2.4rem;font-weight:700'>{porc(p_des, 0)}</div>"
                    f"<div>Resultado más probable: <b>{clase}</b> · {modelo_sel} · {ETAPAS[etapa].lower()}</div></div>",
                    unsafe_allow_html=True)
        if "real" in ss and ss.get("caso_sel") in META["casos_ejemplo"]:
            st.caption(f"Resultado real de este estudiante: **{ss.real}**")
        fig = go.Figure(go.Bar(x=prob, y=CLASES, orientation="h", marker_color=[COLORES[c] for c in CLASES],
                               text=[porc(v) for v in prob], textposition="outside"))
        fig.update_layout(height=200, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 1.12], tickformat=".0%"),
                          yaxis=dict(autorange="reversed"))
        grafico(fig)
    with r2:
        st.markdown("**Cómo evoluciona la alerta a medida que avanza el estudiante**")
        evol = np.vstack([predecir(df_est, e, modelo_sel)[0] for e in range(etapa + 1)])
        dfe = pd.DataFrame(evol, columns=CLASES)
        dfe["Etapa"] = [ETAPAS[e] for e in range(etapa + 1)]
        fig = go.Figure()
        for c in CLASES:
            fig.add_trace(go.Scatter(x=dfe["Etapa"], y=dfe[c], mode="lines+markers+text", name=c,
                                     line=dict(color=COLORES[c], width=3), marker=dict(size=10),
                                     text=[porc(v, 0) for v in dfe[c]], textposition="top center"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                          legend=dict(orientation="h", y=-0.2))
        grafico(fig)
        st.caption("Es la misma red leyendo más información: primero solo la matrícula, luego el 1.er semestre y "
                   "después el año completo.")

    # ---------------- explicación por oclusión
    st.markdown("#### ¿Qué factores empujan el riesgo de este estudiante?")
    grupos = {
        "Matrícula al día": ["matricula_al_dia"], "Deudas con la institución": ["deudor"], "Beca": ["becado"],
        "Programa": ["programa"], "Edad de ingreso": ["edad_ingreso"], "Modalidad de ingreso": ["modo_ingreso"],
        "Nota de admisión": ["nota_admision"], "Género": ["genero_masculino"],
        "Familia (padres)": ["educacion_madre", "educacion_padre", "ocupacion_madre", "ocupacion_padre"],
    }
    if etapa >= 1:
        grupos["Rendimiento 1.er semestre"] = [f"s1_{c}" for c, *_ in CAMPOS_SEM]
    if etapa >= 2:
        grupos["Rendimiento 2.º semestre"] = [f"s2_{c}" for c, *_ in CAMPOS_SEM]
    filas_alt = []
    for g, cols in grupos.items():
        alt = df_est.copy()
        for c in cols:
            alt[c] = REF[c]
        filas_alt.append(alt)
    p_alt = predecir(pd.concat(filas_alt, ignore_index=True), etapa, modelo_sel)[:, 0]
    contrib = pd.DataFrame({"factor": list(grupos), "efecto": p_des - p_alt})
    contrib = contrib[contrib.efecto.abs() > 0.005].sort_values("efecto")
    if contrib.empty:
        st.info("Este estudiante se parece mucho al perfil típico: ningún factor mueve el riesgo de forma apreciable.")
    else:
        fig = go.Figure(go.Bar(x=contrib.efecto, y=contrib.factor, orientation="h",
                               marker_color=np.where(contrib.efecto > 0, "#C0392B", "#3F8F3A"),
                               text=[("+" if v > 0 else "−") + porc(abs(v), 0) for v in contrib.efecto], textposition="outside"))
        lim = max(0.05, contrib.efecto.abs().max() * 1.3)
        fig.update_layout(height=60 + 34 * len(contrib), margin=dict(l=10, r=40, t=10, b=10),
                          xaxis=dict(tickformat="+.0%", range=[-lim, lim], title="cambio en la probabilidad de deserción"))
        grafico(fig)
        st.caption("Cada barra compara la probabilidad de este estudiante con la que tendría si ese factor tuviera el "
                   "valor típico del entrenamiento (rojo = sube el riesgo, verde = lo baja). Es una explicación por "
                   "oclusión: describe lo que aprendió el modelo, no una relación causal.")

    acciones = []
    if not fila["matricula_al_dia"] or fila["deudor"]:
        acciones.append("💳 **Apoyo financiero:** revisar acuerdos de pago, becas o créditos. Tener la matrícula al día "
                        "es uno de los factores que más separa a quien deserta de quien sigue.")
    for s in range(1, etapa + 1):
        ins, apr = fila[f"s{s}_inscritas"], fila[f"s{s}_aprobadas"]
        if ins > 0 and apr / ins < 0.5:
            acciones.append(f"📚 **Acompañamiento académico:** en el {'1.er' if s == 1 else '2.º'} semestre aprobó "
                            f"{apr:.0f} de {ins:.0f} unidades. Tutorías y plan de nivelación antes del siguiente corte.")
    if etapa == 0:
        acciones.append("🕑 **Seguimiento:** con solo datos de matrícula la alerta es orientativa; repetir la evaluación "
                        "al cerrar el primer semestre, cuando el modelo gana bastante precisión.")
    if nivel == "Alto" and not acciones:
        acciones.append("🤝 **Contacto prioritario** de bienestar universitario para conocer la situación del estudiante.")
    if acciones:
        st.markdown("#### Acciones sugeridas")
        for a in acciones:
            st.markdown(f"- {a}")

    if st.button("💾 Guardar esta evaluación en la bitácora (tabla `predicciones`)"):
        ok = registrar({"origen": "individual", "etapa": ETAPAS[etapa], "prob_desercion": float(prob[0]),
                        "prob_matriculado": float(prob[1]), "prob_graduado": float(prob[2]), "clase": clase,
                        "nivel_riesgo": nivel, "entrada_json": json.dumps(fila, default=float)})
        st.success("Guardada en la base SQLite.") if ok else st.error("No fue posible escribir en la base.")

# --------------------------------------------------------------------------- lotes
with tab_lote:
    st.markdown("Sube un CSV con un estudiante por fila y las mismas columnas de la base (en español). "
                "Puedes descargar la plantilla de ejemplo con 30 estudiantes reales de prueba.")
    a, b = st.columns([1, 1])
    a.download_button("⬇️ Descargar plantilla de ejemplo", (MOD / "ejemplo_lote.csv").read_bytes(),
                      "plantilla_estudiantes.csv", "text/csv")
    usar_prueba = b.toggle("Usar los 885 estudiantes del conjunto de prueba (con su resultado real)")
    archivo = st.file_uploader("CSV de estudiantes", type="csv", disabled=usar_prueba)
    etapa_l = st.radio("Etapa de la alerta", list(ETAPAS.values()), index=2, horizontal=True, key="etapa_lote")
    e_l = list(ETAPAS.values()).index(etapa_l)
    datos = None
    if usar_prueba:
        datos = consultar("SELECT * FROM v_estudiantes_plano WHERE particion = 'prueba' ORDER BY estudiante_id")
    elif archivo is not None:
        datos = pd.read_csv(archivo, sep=None, engine="python")
    if datos is not None:
        faltan = [c for c in cdf.NUMERICAS + cdf.BINARIAS + ["programa", "modo_ingreso", "estado_civil",
                  "formacion_previa", "educacion_madre", "educacion_padre", "ocupacion_madre", "ocupacion_padre"]
                  if c not in datos.columns]
        if faltan:
            st.error(f"Faltan columnas: {', '.join(faltan)}")
        else:
            p = predecir(datos, e_l, modelo_sel)
            res = pd.DataFrame({"estudiante_id": datos.get("estudiante_id", pd.Series(range(1, len(datos) + 1))),
                                "programa": datos["programa"].map(pp.PROGRAMAS),
                                "prob_desercion": p[:, 0], "prob_matriculado": p[:, 1], "prob_graduado": p[:, 2],
                                "prediccion": [CLASES[i] for i in p.argmax(1)],
                                "riesgo": [nivel_riesgo(v)[0] for v in p[:, 0]]})
            if "resultado" in datos.columns:
                res["resultado_real"] = datos["resultado"].values
            c1, c2, c3 = st.columns(3)
            for col, n in zip((c1, c2, c3), ("Alto", "Medio", "Bajo")):
                col.metric(f"Riesgo {n.lower()}", f"{(res.riesgo == n).sum()} estudiantes",
                           f"{porc((res.riesgo == n).mean(), 0)} del grupo", delta_color="off")
            st.dataframe(res.sort_values("prob_desercion", ascending=False), use_container_width=True, hide_index=True,
                         column_config={k: st.column_config.ProgressColumn(k.replace("_", " "), format="%.2f",
                                                                           min_value=0, max_value=1)
                                        for k in ("prob_desercion", "prob_matriculado", "prob_graduado")})
            if "resultado_real" in res:
                acierto = (res.prediccion == res.resultado_real).mean()
                st.success(f"Exactitud sobre este grupo: **{porc(acierto)}** ({modelo_sel}, {etapa_l.lower()}).")
                cm = pd.crosstab(res.resultado_real, res.prediccion).reindex(index=CLASES, columns=CLASES, fill_value=0)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="Predicho", y="Real", color="Estudiantes"))
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Matriz de confusión")
                grafico(fig)
            st.download_button("⬇️ Descargar resultados", res.to_csv(index=False).encode("utf-8"),
                               "predicciones_desercion.csv", "text/csv")
            if st.button("💾 Guardar el lote en la bitácora"):
                ok = all(registrar({"origen": "lote", "etapa": etapa_l, "prob_desercion": float(r.prob_desercion),
                                    "prob_matriculado": float(r.prob_matriculado), "prob_graduado": float(r.prob_graduado),
                                    "clase": r.prediccion, "nivel_riesgo": r.riesgo,
                                    "entrada_json": json.dumps({"estudiante_id": int(r.estudiante_id)})})
                         for r in res.itertuples())
                st.success(f"{len(res)} evaluaciones guardadas.") if ok else st.error("No fue posible escribir en la base.")

# --------------------------------------------------------------------------- desempeño
with tab_desemp:
    st.markdown("Resultados sobre **885 estudiantes de prueba** que ninguna red vio durante el entrenamiento. Son los "
                "mismos del ACA 1, así que la regresión logística es una comparación directa.")
    pr = pd.DataFrame(META["prueba"])
    fig = px.bar(pr, x="etapa", y="F1 macro", color="modelo", barmode="group", text_auto=".3f",
                 category_orders={"etapa": list(ETAPAS.values())},
                 color_discrete_map={"GRU híbrida": "#E08A1E", "MLP profundo": "#3F8F3A", "FT-Transformer": "#6C3FA0",
                                     "Regresión logística (ACA 1)": "#5B6770"})
    fig.update_layout(height=380, yaxis=dict(range=[0.4, 0.8]), margin=dict(l=10, r=10, t=30, b=10),
                      title="F1 macro por etapa de alerta", legend=dict(orientation="h", y=-0.2))
    grafico(fig)
    tabla = pr.pivot_table(index="modelo", columns="etapa", values="F1 macro")[list(ETAPAS.values())]
    tabla.columns = [f"F1 · {c.lower()}" for c in tabla.columns]
    fin = pr[pr.etapa == ETAPAS[2]].set_index("modelo")
    tabla["Exactitud (año)"] = fin["Exactitud"]
    tabla["ROC-AUC (año)"] = fin["ROC-AUC (OvR macro)"]
    tabla["Sensibilidad deserción (año)"] = fin["Sensibilidad Deserción"]
    st.dataframe(tabla_es(tabla), use_container_width=True)

    a, b = st.columns(2)
    with a:
        cm = np.array(META["matrices_confusion"][modelo_sel][ETAPAS[2]])
        fig = px.imshow(cm, x=CLASES, y=CLASES, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicho", y="Real", color="Estudiantes"))
        fig.update_layout(height=360, title=f"Matriz de confusión · {modelo_sel} · tras el 1.er año",
                          margin=dict(l=10, r=10, t=40, b=10))
        grafico(fig)
        pc = pd.DataFrame(META["por_clase"]).T.rename(columns={"precision": "precisión", "recall": "sensibilidad",
                                                              "f1-score": "F1", "support": "estudiantes"})
        pc["estudiantes"] = pc["estudiantes"].astype(int)
        st.caption(f"Métricas por clase de la {PRINCIPAL} (tras el 1.er año)")
        st.dataframe(tabla_es(pc), use_container_width=True)
    with b:
        curvas = META["curvas"][modelo_sel]
        fig = go.Figure()
        for i, h in enumerate(curvas):
            fig.add_trace(go.Scatter(y=h["perdida_entrenamiento"], mode="lines", line=dict(color="#1A4F8B", width=1.3),
                                     name="Entrenamiento", legendgroup="tr", showlegend=i == 0, opacity=.6))
            fig.add_trace(go.Scatter(y=h["perdida_validacion"], mode="lines", line=dict(color="#E08A1E", width=1.3),
                                     name="Validación", legendgroup="va", showlegend=i == 0, opacity=.85))
        fig.update_layout(height=360, title=f"Curvas de pérdida · {modelo_sel} (ensamble de 5 redes)",
                          xaxis_title="Época", yaxis_title="Entropía cruzada", margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.2))
        grafico(fig)
        imp = pd.DataFrame(META["importancia"]).head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.caida_f1, y=imp.grupo, orientation="h", marker_color="#E08A1E",
                               error_x=dict(type="data", array=imp.desv)))
        fig.update_layout(height=330, title="¿Qué mira la GRU? (importancia por permutación)",
                          xaxis_title="caída del F1 macro", margin=dict(l=10, r=10, t=40, b=10))
        grafico(fig)

    st.markdown("**Validación cruzada (5 pliegues, conjunto de entrenamiento)**")
    st.dataframe(tabla_es(pd.DataFrame(META["validacion_cruzada"]).set_index("modelo")), use_container_width=True)
    d = META["diferencia_vs_lr"][ETAPAS[2]]
    st.caption(f"Diferencia de F1 macro GRU − regresión logística tras el año: {num(d['diferencia'])} "
               f"(IC 95 % por bootstrap: {num(d['ic95'][0])} a {num(d['ic95'][1])}). El intervalo incluye el cero: "
               "la GRU iguala o supera ligeramente a la línea base, con un solo modelo para las tres etapas.")

# --------------------------------------------------------------------------- cómo funciona
with tab_como:
    st.markdown("#### Flujo del proyecto")
    st.image(str(RAIZ / "assets" / "flujo_metodologico.png"), use_container_width=True)
    st.markdown("#### La red: una GRU híbrida")
    st.image(str(RAIZ / "assets" / "arquitectura_red.png"), use_container_width=True)
    st.markdown("""
- **Memoria que avanza con el estudiante.** La red empieza con un resumen de cómo llegó el estudiante (programa, edad,
  situación financiera, familia…). Ese resumen es su "memoria inicial". Luego lee el primer semestre y actualiza la
  memoria; después lee el segundo y vuelve a actualizarla.
- **Un solo modelo para alertar en tres momentos.** Durante el entrenamiento se le esconden al azar los semestres que
  "todavía no han pasado". Así aprende a dar una alerta el día de la matrícula, al cerrar el primer semestre o al cerrar
  el año.
- **Cinco redes votan.** La predicción es el promedio de cinco redes entrenadas con particiones distintas de los
  datos, lo que la hace más estable.
- **Liviana para la nube.** Las redes se entrenaron en PyTorch y se exportaron a ONNX; esta app las ejecuta con
  onnxruntime sin necesidad de PyTorch.
""")
    st.markdown(f"Parámetros entrenables por red: " + ", ".join(f"**{k}** {v:,}".replace(",", ".")
                                                               for k, v in META["parametros"].items()))
    st.info("Limitaciones: los datos provienen de una institución portuguesa (Realinho et al., 2021). Antes de usar el "
            "modelo en una institución colombiana habría que reentrenarlo con datos propios y revisar el uso de variables "
            "sensibles como el género y la edad.")

# --------------------------------------------------------------------------- SQL
with tab_sql:
    st.markdown("La app lee una base **SQLite** con el dataset normalizado. Estas son algunas de las consultas del análisis.")
    resumen = consultar("""SELECT 'estudiantes' AS tabla, COUNT(*) AS filas FROM estudiantes
                           UNION ALL SELECT 'semestres', COUNT(*) FROM semestres
                           UNION ALL SELECT 'programas', COUNT(*) FROM programas
                           UNION ALL SELECT 'cohortes', COUNT(*) FROM cohortes""")
    st.dataframe(resumen, hide_index=True)
    CONSULTAS = {
        "Deserción por programa": """SELECT p.nombre AS programa, COUNT(*) AS estudiantes,
       ROUND(100.0 * AVG(e.resultado = 'Deserción'), 1) AS desercion_pct
FROM estudiantes e JOIN programas p ON p.codigo = e.programa
GROUP BY p.nombre ORDER BY desercion_pct DESC""",
        "Trayectoria media por resultado": """SELECT e.resultado, s.semestre,
       ROUND(AVG(s.aprobadas), 2) AS aprobadas, ROUND(AVG(s.nota), 2) AS promedio
FROM semestres s JOIN estudiantes e USING (estudiante_id)
WHERE e.sin_registro_curricular = 0
GROUP BY e.resultado, s.semestre ORDER BY e.resultado, s.semestre""",
        "Factores financieros": """SELECT matricula_al_dia, deudor, COUNT(*) AS estudiantes,
       ROUND(100.0 * AVG(resultado = 'Deserción'), 1) AS desercion_pct
FROM estudiantes GROUP BY matricula_al_dia, deudor""",
        "Deserción por cohorte": """SELECT c.cohorte_id, c.tasa_desempleo, c.tasa_inflacion, c.pib, COUNT(*) AS estudiantes,
       ROUND(100.0 * AVG(e.resultado = 'Deserción'), 1) AS desercion_pct
FROM estudiantes e JOIN cohortes c USING (cohorte_id) GROUP BY c.cohorte_id ORDER BY desercion_pct DESC""",
        "Bitácora de predicciones de la app": "SELECT * FROM predicciones ORDER BY id DESC LIMIT 50",
    }
    elegida = st.selectbox("Consulta", list(CONSULTAS))
    texto = st.text_area("SQL (solo lectura: se permiten consultas SELECT)", CONSULTAS[elegida], height=150)
    if st.button("Ejecutar consulta"):
        if not texto.strip().lower().startswith(("select", "with")):
            st.error("Solo se permiten consultas de lectura (SELECT).")
        else:
            try:
                st.dataframe(consulta_libre(texto), use_container_width=True, hide_index=True)
            except Exception as ex:  # noqa: BLE001
                st.error(f"Error en la consulta: {ex}")

st.divider()
st.caption("Dataset: Realinho, V., Vieira Martins, M., Machado, J. y Baptista, L. (2021). Predict students' dropout and "
           "academic success [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89 · "
           "Herramienta de apoyo a la decisión: no reemplaza el criterio de consejeros y docentes.")
