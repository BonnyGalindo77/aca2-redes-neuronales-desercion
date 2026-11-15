"""Genera el póster infografía (A3 vertical, PDF con hipervínculos y QR).

Lee modelo/metadatos.json, figuras/*.png, assets/*.png y enlaces.json.
Uso:  python scripts/generar_poster.py   → entregables/Poster_ACA2_Redes_Neuronales_Desercion.pdf
Requiere: playwright + Chrome/Chromium y qrcode.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import qrcode
import qrcode.image.svg

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "entregables"
SALIDA.mkdir(exist_ok=True)


def leer(ruta):
    p = RAIZ / ruta
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def img(ruta):
    p = RAIZ / ruta
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def qr(url):
    if not url:
        return "<div class='qrph'>QR pendiente</div>"
    im = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=1)
    b = io.BytesIO()
    im.save(b)
    s = b.getvalue().decode()
    return s[s.index("<svg"):].replace("<svg ", '<svg class="qr" ', 1)


def f(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


def figura_poster(M):
    """Figura compacta de resultados: F1 por etapa, matriz de confusión y curvas de pérdida."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    et = list(M["etapas"].values())
    mods = ["GRU híbrida", "MLP profundo", "FT-Transformer", "Regresión logística (ACA 1)"]
    col = {"GRU híbrida": "#E08A1E", "MLP profundo": "#3F8F3A", "FT-Transformer": "#6C3FA0",
           "Regresión logística (ACA 1)": "#5B6770"}
    P = {(r["modelo"], r["etapa"]): r["F1 macro"] for r in M["prueba"]}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.2, 2.75), gridspec_kw={"width_ratios": [1.45, 0.95, 1.2]})
    x = np.arange(3); w = 0.2
    for i, m in enumerate(mods):
        v = [P[(m, e)] for e in et]
        bars = a.bar(x + (i - 1.5) * w, v, w, color=col[m], label=m.replace(" (ACA 1)", " ACA 1"))
        for bb, vv in zip(bars, v):
            a.text(bb.get_x() + bb.get_width() / 2, vv + 0.004, f"{vv:.3f}".replace(".", ","), ha="center",
                   fontsize=5.6, rotation=90, va="bottom")
    a.set_xticks(x, ["Al ingreso", "Tras 1.er semestre", "Tras 1.er año"]); a.set_ylim(0.45, 0.78)
    a.set_title("F1 macro en prueba por etapa", fontweight="bold"); a.legend(fontsize=6, ncol=2, frameon=False, loc="upper left")
    a.spines[["top", "right"]].set_visible(False)
    cm = np.array(M["matrices_confusion"][M["modelo_principal"]][et[2]])
    norm = cm / cm.sum(1, keepdims=True)
    b.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            b.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}".replace("%", " %"), ha="center", va="center", fontsize=7,
                   color="white" if norm[i, j] > .55 else "#1F2A33")
    lab = ["Deserción", "Matric.", "Graduado"]
    b.set_xticks(range(3), lab); b.set_yticks(range(3), lab); b.set_xlabel("Predicho"); b.set_ylabel("Real")
    b.set_title("Matriz de confusión (GRU)", fontweight="bold")
    for i, h in enumerate(M["curvas"][M["modelo_principal"]]):
        c.plot(h["perdida_entrenamiento"], color="#1A4F8B", alpha=.6, lw=1, label="Entrenamiento" if i == 0 else None)
        c.plot(h["perdida_validacion"], color="#E08A1E", alpha=.9, lw=1, label="Validación" if i == 0 else None)
    c.set_title("Pérdida por época (5 redes GRU)", fontweight="bold"); c.set_xlabel("Época")
    c.legend(frameon=False, fontsize=7); c.grid(alpha=.3); c.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(w_pad=1.2)
    fig.savefig(RAIZ / "figuras" / "06_resultados_poster.png", dpi=260, facecolor="white")
    plt.close(fig)

    import sqlite3
    import pandas as pd
    con = sqlite3.connect(RAIZ / "data" / "desercion.db")
    t = pd.read_sql_query("""SELECT e.resultado, s.semestre, AVG(s.aprobadas) AS aprobadas,
        AVG(CASE WHEN s.inscritas > 0 THEN 1.0 * s.aprobadas / s.inscritas END) AS tasa, AVG(s.nota) AS promedio
        FROM semestres s JOIN estudiantes e USING (estudiante_id) WHERE e.sin_registro_curricular = 0
        GROUP BY e.resultado, s.semestre""", con)
    con.close()
    colc = {"Deserción": "#C0392B", "Matriculado": "#E08A1E", "Graduado": "#1A4F8B"}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.05))
    for ax, (v, tt) in zip(axs, [("aprobadas", "Unidades aprobadas"), ("tasa", "Tasa de aprobación"),
                                 ("promedio", "Promedio (0-20)")]):
        for cl in ["Deserción", "Matriculado", "Graduado"]:
            d = t[t.resultado == cl].sort_values("semestre")
            ax.plot(["S1", "S2"], d[v], "o-", color=colc[cl], lw=2, ms=5, label=cl)
            ax.annotate(f"{d[v].iloc[1]:.2f}".replace(".", ","), (1, d[v].iloc[1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5, va="center", color=colc[cl])
        ax.set_title(tt, fontweight="bold", fontsize=8); ax.set_xlim(-0.25, 1.45)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.3)
    axs[0].legend(fontsize=6.5, frameon=False, loc="center left")
    plt.tight_layout(w_pad=0.8)
    fig.savefig(RAIZ / "figuras" / "07_trayectorias_poster.png", dpi=260, facecolor="white")
    plt.close(fig)


def construir():
    M = leer("modelo/metadatos.json")
    E = leer("enlaces.json")
    figura_poster(M)
    P = {(r["modelo"], r["etapa"]): r for r in M["prueba"]}
    et = list(M["etapas"].values())
    modelos = ["GRU híbrida", "MLP profundo", "FT-Transformer", "Regresión logística (ACA 1)"]
    g = M["metricas_principal"]
    ic = M["ic95_principal"]
    hp = M["hiperparametros"]
    par = M["parametros"]

    filas = ""
    for m in modelos:
        cls = " class='principal'" if m == M["modelo_principal"] else ""
        celdas = "".join(f"<td>{f(P[(m, e)]['F1 macro'])}</td>" for e in et)
        filas += (f"<tr{cls}><td>{m}</td>{celdas}<td>{f(P[(m, et[2])]['ROC-AUC (OvR macro)'])}</td>"
                  f"<td>{pct(P[(m, et[2])]['Exactitud'])}</td></tr>")

    pc = M["por_clase"]
    clases = "".join(f"<tr><td>{c}</td><td>{f(pc[c]['precision'])}</td><td>{f(pc[c]['recall'])}</td>"
                     f"<td>{f(pc[c]['f1-score'])}</td></tr>" for c in M["clases"])
    url_app, url_video, url_repo = E.get("app", ""), E.get("video", ""), E.get("repo", "")

    def enlace(t, u):
        cuerpo = f"<a href='{u}'>{u.replace('https://', '')}</a>" if u else "<i>se agrega al publicar</i>"
        return f"<div class='enl'>{qr(u)}<div><b>{t}</b><br>{cuerpo}</div></div>"

    T = Path(RAIZ / "scripts" / "poster_textos.json")
    X = json.loads(T.read_text(encoding="utf-8"))

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><style>
@page {{ size: 297mm 420mm; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: 'DejaVu Sans', 'Segoe UI', Arial, sans-serif; color: #1F2A33; font-size: 7.5pt;
       line-height: 1.3; background: #F3F6F9; }}
.pag {{ width: 297mm; height: 420mm; padding: 6mm 8mm 4mm; display: flex; flex-direction: column; gap: 2.3mm; overflow: hidden; }}
header {{ display: grid; grid-template-columns: 46mm 1fr 58mm; gap: 5mm; align-items: center; background: white;
          border-radius: 4mm; padding: 2.4mm 5mm; border-bottom: 1.8mm solid #1A4F8B; }}
header img {{ width: 46mm; }}
.sup {{ font-size: 7.4pt; letter-spacing: .12em; text-transform: uppercase; color: #3F8F3A; font-weight: 700; }}
h1 {{ margin: .4mm 0 .6mm; font-size: 16.5pt; line-height: 1.08; color: #1A4F8B; }}
h1 span {{ color: #E08A1E; }}
.sub {{ font-size: 8.4pt; color: #33414C; }}
.equipo {{ margin-top: 1mm; font-size: 7.6pt; }}
.grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 2.3mm; }}
.card {{ background: white; border-radius: 3mm; padding: 2.4mm 3.4mm; border-top: 1.4mm solid #1A4F8B; }}
.card h2 {{ margin: 0 0 1.2mm; font-size: 9.6pt; color: #1A4F8B; text-transform: uppercase; letter-spacing: .04em; }}
.c-ve {{ border-top-color: #3F8F3A; }} .c-ve h2 {{ color: #3F8F3A; }}
.c-na {{ border-top-color: #E08A1E; }} .c-na h2 {{ color: #B86A0C; }}
.c-ro {{ border-top-color: #C0392B; }} .c-ro h2 {{ color: #C0392B; }}
.c-mo {{ border-top-color: #6C3FA0; }} .c-mo h2 {{ color: #6C3FA0; }}
.s3 {{ grid-column: span 3; }} .s4 {{ grid-column: span 4; }} .s5 {{ grid-column: span 5; }}
.s6 {{ grid-column: span 6; }} .s7 {{ grid-column: span 7; }} .s8 {{ grid-column: span 8; }} .s9 {{ grid-column: span 9; }} .s12 {{ grid-column: span 12; }}
p {{ margin: 0 0 1.1mm; }}
.preg {{ background: #FFF5E8; border-left: 1.2mm solid #E08A1E; padding: 2mm 3mm; font-style: italic; margin-top: 1.6mm; }}
img.fig {{ width: 100%; display: block; }}
.kpis {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2mm; }}
.kpi {{ background: #F3F6F9; border-radius: 2.4mm; padding: 1.4mm 2.2mm; }}
.kpi b {{ display: block; font-size: 13pt; color: #1A4F8B; line-height: 1.1; }}
.kpi.r b {{ color: #C0392B; }} .kpi.n b {{ color: #E08A1E; }}
.kpi span {{ font-size: 7pt; color: #5B6770; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.8pt; }}
th, td {{ padding: 1mm 1.4mm; border-bottom: .3mm solid #E3E8EC; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #EEF3F8; color: #1A4F8B; }}
tr.principal td {{ background: #FFF5E8; font-weight: 700; }}
ul {{ margin: 0; padding-left: 4mm; }} li {{ margin-bottom: 1.1mm; }}
.shots {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3mm; }}
.res {{ display: grid; grid-template-columns: 1.75fr 1fr; gap: 3mm; align-items: start; }}
.shots img {{ width: 100%; height: 39mm; object-fit: cover; object-position: top; border: .3mm solid #D5DCE2; border-radius: 2mm; }}
.cap {{ font-size: 7pt; color: #5B6770; margin-top: .8mm; }}
.enl {{ display: flex; gap: 2.4mm; align-items: center; margin-bottom: 1.4mm; font-size: 7pt; word-break: break-all; }}
svg.qr {{ width: 14mm; height: 14mm; flex: none; }}
.qrph {{ width: 14mm; height: 14mm; border: .4mm dashed #9AA6B0; display: flex; align-items: center; justify-content: center;
         font-size: 6.5pt; color: #9AA6B0; flex: none; text-align: center; }}
a {{ color: #1A4F8B; text-decoration: none; }}
.refs p {{ font-size: 6.1pt; line-height: 1.25; padding-left: 3mm; text-indent: -3mm; margin-bottom: .9mm; }}
footer {{ display: flex; justify-content: space-between; font-size: 7pt; color: #5B6770; padding: 0 2mm; margin-top: auto; }}
.tag {{ display: inline-block; background: #EEF3F8; color: #1A4F8B; border-radius: 1.4mm; padding: .2mm 1.6mm; margin: 0 .6mm .8mm 0;
        font-size: 7pt; font-weight: 700; }}
</style></head><body><div class="pag">

<header>
 <img src="{img('assets/logo_cun.png')}">
 <div>
  <div class="sup">Especialización en Inteligencia Artificial · Aprendizaje Automático · ACA 2</div>
  <h1>Alerta temprana de deserción estudiantil con <span>redes neuronales recurrentes</span></h1>
  <div class="sub">Una GRU que lee la trayectoria del estudiante semestre a semestre y avisa desde el día de la matrícula</div>
  <div class="equipo"><b>Bonny Darhyll Brayan Galindo Santos · Anternol de los Reyes Bedoya Naranjo · Andrés Alberto López Herrera · Camilo José Rodríguez Ayubi</b><br>
  Grupo 7 · Docente: Dr. Hamilton Rivera Flor · Corporación Unificada Nacional de Educación Superior (CUN) · Noviembre de 2026</div>
 </div>
 <div class="kpis">
  <div class="kpi"><b>4.424</b><span>estudiantes · 8.848 semestres</span></div>
  <div class="kpi r"><b>32,1 %</b><span>desertó</span></div>
  <div class="kpi n"><b>{pct(g['Exactitud'])}</b><span>exactitud en prueba (GRU)</span></div>
  <div class="kpi"><b>{pct(g['Sensibilidad Deserción'])}</b><span>de los desertores detectados</span></div>
 </div>
</header>

<div class="grid">
 <div class="card s6"><h2>Objetivo general</h2><p>{X['objetivo']}</p></div>
 <div class="card s6 c-na"><h2>Situación problema</h2><p>{X['problema']}</p>
  <div class="preg">¿Cómo aprovechar las arquitecturas de redes neuronales para extraer patrones complejos en datos de educación, optimizando el rendimiento predictivo y permitiendo su uso práctico en Streamlit?</div></div>
 <div class="card s3 c-ve" style="grid-row: span 2"><h2>Flujo metodológico</h2><img class="fig" src="{img('assets/flujo_vertical.png')}"></div>
 <div class="card s4"><h2>Datos y gestión SQL</h2><p>{X['datos']}</p>
  <table><tr><th>Tabla SQLite</th><th>Filas</th></tr>
  <tr><td>estudiantes (ingreso, finanzas, resultado)</td><td>4.424</td></tr>
  <tr><td>semestres (trayectoria S1, S2)</td><td>8.848</td></tr>
  <tr><td>cohortes · programas</td><td>10 · 17</td></tr>
  <tr><td>predicciones (bitácora de la app)</td><td>—</td></tr></table>
  <p style="margin-top:1.2mm">{X['datos2']}</p></div>
 <div class="card s5"><h2>La trayectoria importa</h2>
  <img class="fig" src="{img('figuras/07_trayectorias_poster.png')}"><p style="margin-top:1mm">{X['tray']}</p></div>

 <div class="card s5 c-na"><h2>Arquitectura principal: GRU híbrida</h2>
  <img class="fig" src="{img('assets/arquitectura_red.png')}"><p style="margin-top:1.2mm">{X['arquitectura']}</p></div>
 <div class="card s4 c-na"><h2>Modelos comparados</h2>
  <p><span class="tag">GRU híbrida</span><br>oculto {hp['GRU híbrida']['oculto']}, dropout {f(hp['GRU híbrida']['dropout'], 1)} · {miles(par['GRU híbrida'])} parámetros</p>
  <p><span class="tag">MLP profundo</span><br>capas {'-'.join(map(str, hp['MLP profundo']['capas']))}, dropout {f(hp['MLP profundo']['dropout'], 1)} · {miles(par['MLP profundo'])} parámetros</p>
  <p><span class="tag">FT-Transformer</span><br>{hp['FT-Transformer']['capas']} bloques de autoatención, {hp['FT-Transformer']['cabezas']} cabezas · {miles(par['FT-Transformer'])} parámetros</p>
  <p>{X['entrenamiento']}</p></div>

 <div class="card s12 c-ro"><h2>Modelos y resultados en 885 estudiantes de prueba (nunca vistos)</h2>
  <div class="res"><img class="fig" src="{img('figuras/06_resultados_poster.png')}">
  <div><table><tr><th>Modelo</th><th>F1 ingreso</th><th>F1 S1</th><th>F1 año</th><th>ROC-AUC</th><th>Exactitud</th></tr>{filas}</table>
  <p style="margin-top:1.4mm">{X['resultados']}</p></div></div></div>

 <div class="card s9 c-mo"><h2>Vista previa de la aplicación en Streamlit</h2><div class="shots">
  <div><img src="{img('assets/captura_app_1.png')}"><div class="cap">{X['cap1']}</div></div>
  <div><img src="{img('assets/captura_app_2.png')}"><div class="cap">{X['cap2']}</div></div></div></div>
 <div class="card s3 c-mo"><h2>Enlaces</h2>{enlace('Aplicación desplegada', url_app)}{enlace('Video explicativo', url_video)}{enlace('Repositorio y notebook', url_repo)}</div>

 <div class="card s6 c-ve"><h2>Conclusiones</h2><ul>{''.join(f'<li>{c}</li>' for c in X['conclusiones'])}</ul></div>
 <div class="card s6 refs"><h2>Referencias (APA 7)</h2>{''.join(f'<p>{r}</p>' for r in X['referencias'])}</div>
</div>
<footer><span>ACA 2 · Aprendizaje Automático · CUN · 2026</span><span>Dataset: UCI Machine Learning Repository (id 697) · CC BY 4.0</span></footer>
</div></body></html>"""
    return html


async def a_pdf(html, salida):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        nav = await p.chromium.launch(executable_path="/usr/bin/google-chrome" if Path("/usr/bin/google-chrome").exists() else None)
        pag = await nav.new_page()
        await pag.set_content(html, wait_until="load")
        alto = await pag.evaluate("[document.querySelector('.pag').scrollHeight, document.querySelector('.pag').clientHeight]")
        print("alto contenido / página (px):", alto, "→ desborda" if alto[0] > alto[1] + 1 else "→ cabe")
        await pag.pdf(path=str(salida), width="297mm", height="420mm", print_background=True)
        await pag.screenshot(path=str(salida.with_suffix(".png")), full_page=True)
        await nav.close()


if __name__ == "__main__":
    h = construir()
    (SALIDA / "poster.html").write_text(h, encoding="utf-8")
    destino = SALIDA / "Poster_ACA2_Redes_Neuronales_Desercion.pdf"
    asyncio.run(a_pdf(h, destino))
    print("Póster generado:", destino)
