"""Genera los diagramas del flujo metodológico y de la arquitectura de la red.

Uso:  python scripts/diagramas.py   → assets/flujo_metodologico.png, assets/arquitectura_red.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "assets"
SALIDA.mkdir(exist_ok=True)

AZUL, VERDE, NARANJA, ROJO, GRIS, MORADO = "#1A4F8B", "#3F8F3A", "#E08A1E", "#C0392B", "#5B6770", "#6C3FA0"
TINTA = "#1F2A33"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def caja(ax, x, y, w, h, titulo, cuerpo="", color=AZUL, fs_t=11, fs_c=8.8, hb=0.46, r=0.12):
    """Caja con encabezado de color. Coordenadas en pulgadas (aspecto 1:1)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc="white", ec=color, lw=1.8, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - hb), w, hb, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=color, ec=color, lw=1.8, zorder=3))
    ax.add_patch(plt.Rectangle((x, y + h - hb), w, r, fc=color, ec=color, lw=1.8, zorder=3))
    ax.text(x + w / 2, y + h - hb / 2, titulo, ha="center", va="center", color="white",
            fontsize=fs_t, fontweight="bold", zorder=4)
    if cuerpo:
        ax.text(x + w / 2, y + (h - hb) / 2, cuerpo, ha="center", va="center", color=TINTA,
                fontsize=fs_c, linespacing=1.45, zorder=4)


def flecha(ax, x0, y0, x1, y1, color=GRIS, lw=2.0, estilo="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=estilo, mutation_scale=15,
                                 color=color, lw=lw, zorder=5, shrinkA=0, shrinkB=0))


def lienzo(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def flujo():
    fig, ax = lienzo(14, 1.95)
    pasos = [
        ("1 · Datos y SQL", "CSV UCI → SQLite\n4.424 estudiantes · 8.848 semestres\nlimpieza y consultas en SQL", AZUL),
        ("2 · Preparación", "embeddings (8 categóricas)\nsecuencia S1 → S2 + máscara\naumento de datos", VERDE),
        ("3 · Modelado", "GRU híbrida · MLP · FT-Transformer\n14 configuraciones\nvalidación cruzada de 5 pliegues", NARANJA),
        ("4 · Evaluación", "885 estudiantes nunca vistos\nF1 macro · ROC-AUC · matriz\nvs. regresión del ACA 1", ROJO),
        ("5 · Despliegue", "ensamble de 5 redes → ONNX\napp en Streamlit Cloud\nindividual y por lotes", MORADO),
    ]
    w, gap, x0 = 2.46, 0.4, 0.08
    for i, (t, c, col) in enumerate(pasos):
        x = x0 + i * (w + gap)
        caja(ax, x, 0.06, w, 1.82, t, c, col, fs_t=12, fs_c=9.4, hb=0.5)
        if i < len(pasos) - 1:
            flecha(ax, x + w + 0.05, 0.97, x + w + gap - 0.05, 0.97, lw=2.4)
    fig.savefig(SALIDA / "flujo_metodologico.png", dpi=220, facecolor="white")
    plt.close(fig)


def flujo_vertical():
    """Versión vertical y compacta del flujo, para el póster."""
    W, H = 3.4, 5.15
    fig, ax = lienzo(W, H)
    pasos = [
        ("Datos y SQL", "CSV UCI → SQLite (5 tablas)\nlimpieza y EDA con consultas", AZUL),
        ("Preparación", "embeddings · secuencia S1→S2\nmáscara de etapa · aumento", VERDE),
        ("Modelado", "GRU · MLP · FT-Transformer\n14 configuraciones · CV 5×", NARANJA),
        ("Evaluación", "885 estudiantes nunca vistos\nF1 · ROC-AUC · vs. ACA 1", ROJO),
        ("Despliegue", "ensamble de 5 redes → ONNX\nStreamlit Cloud", MORADO),
    ]
    h, gap = 0.84, 0.21
    for i, (t, c, col) in enumerate(pasos):
        y = H - 0.05 - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.05, y), W - 0.1, h, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="white", ec=col, lw=1.6, zorder=2))
        ax.add_patch(FancyBboxPatch((0.05, y), 0.5, h, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=col, ec=col, lw=1.6, zorder=3))
        ax.add_patch(plt.Rectangle((0.45, y), 0.1, h, fc=col, ec=col, zorder=3))
        ax.text(0.3, y + h / 2, str(i + 1), color="white", fontsize=15, fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(0.68, y + h - 0.2, t, color=col, fontsize=10, fontweight="bold", va="center", zorder=4)
        ax.text(0.68, y + h / 2 - 0.13, c, color=TINTA, fontsize=7.8, va="center", linespacing=1.35, zorder=4)
        if i < len(pasos) - 1:
            flecha(ax, W / 2, y - 0.02, W / 2, y - gap + 0.02, lw=2)
    fig.savefig(SALIDA / "flujo_vertical.png", dpi=260, facecolor="white")
    plt.close(fig)


def arquitectura():
    fig, ax = lienzo(14, 4.72)
    caja(ax, 0.08, 2.65, 2.4, 1.75, "Datos de ingreso", "8 categóricas → embeddings\n(programa, modalidad,\nnivel educativo de los padres…)",
         AZUL, fs_c=8.6)
    caja(ax, 0.08, 0.6, 2.4, 1.75, "Contexto", "7 numéricas estandarizadas\n+ 8 binarias (matrícula al día,\ndeudor, beca, género…)",
         AZUL, fs_c=8.6)
    caja(ax, 2.95, 1.75, 1.75, 1.9, "Codificador\nde contexto", "densa 128 → 64\nLayerNorm · GELU\ndropout 0,4", VERDE,
         fs_t=10, fs_c=8.8, hb=0.7)
    flecha(ax, 2.5, 3.5, 2.93, 3.0)
    flecha(ax, 2.5, 1.5, 2.93, 2.3)

    fila_y, fila_h = 2.15, 1.25
    caja(ax, 5.1, fila_y, 1.3, fila_h, "h₀", "estado inicial\n(al ingreso)", NARANJA, fs_c=8.4)
    caja(ax, 6.8, fila_y, 1.3, fila_h, "GRU · S1", "h₁", NARANJA, fs_c=11)
    caja(ax, 8.5, fila_y, 1.3, fila_h, "GRU · S2", "h₂", NARANJA, fs_c=11)
    caja(ax, 10.2, fila_y - 0.15, 1.6, fila_h + 0.3, "Máscara", "usa el estado del\núltimo semestre\nconocido: h₀, h₁ o h₂",
         ROJO, fs_c=8.2)
    caja(ax, 12.2, fila_y - 0.35, 1.72, fila_h + 0.7, "Cabeza densa", "64 → 3 · softmax\nDeserción\nMatriculado\nGraduado",
         MORADO, fs_c=8.8)
    ym = fila_y + (fila_h - 0.46) / 2
    flecha(ax, 4.72, ym, 5.08, ym)
    for xa, xb in ((6.42, 6.78), (8.12, 8.48), (9.82, 10.18)):
        flecha(ax, xa, ym, xb, ym, color=NARANJA, lw=2.4)
    flecha(ax, 11.82, ym, 12.18, ym, color=ROJO, lw=2.2)

    caja(ax, 6.4, 0.1, 2.0, 1.45, "1.er semestre", "inscritas · evaluadas\naprobadas · promedio\ntasa de aprobación…",
         GRIS, fs_t=9.5, fs_c=8)
    caja(ax, 8.55, 0.1, 2.0, 1.45, "2.º semestre", "los mismos 9 rasgos\n(solo si ya ocurrió)", GRIS, fs_t=9.5, fs_c=8)
    flecha(ax, 7.45, 1.57, 7.45, 2.13)
    flecha(ax, 9.15, 1.57, 9.15, 2.13)

    # el contexto también llega directo a la cabeza (por encima de la cadena)
    ax.plot([3.82, 3.82, 13.06], [3.67, 4.35, 4.35], color=VERDE, lw=2, zorder=1)
    flecha(ax, 13.06, 4.35, 13.06, 3.62, color=VERDE)
    ax.text(8.4, 4.5, "el contexto de ingreso también llega directo a la cabeza", fontsize=9, color=VERDE,
            style="italic", ha="center")
    fig.savefig(SALIDA / "arquitectura_red.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flujo()
    flujo_vertical()
    arquitectura()
    print("Diagramas generados en", SALIDA)
