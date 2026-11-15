"""
entrenamiento.py
----------------
Bucle de entrenamiento, aumento de datos, validación cruzada y métricas
(Pasos 2, 3 y 4 del flujo).

Decisiones principales
* Pérdida: entropía cruzada con pesos de clase balanceados (como el
  class_weight="balanced" del ACA 1) y suavizado de etiquetas de 0,05.
* Aumento de datos en cada lote (solo en entrenamiento):
    - enmascaramiento de trayectoria: a cada estudiante se le asigna al azar
      la etapa 0, 1 o 2 y se ocultan los semestres posteriores. Multiplica
      los ejemplos y enseña al modelo a alertar con información incompleta;
    - ruido gaussiano sobre variables continuas (sigma = 0,05 en unidades
      estandarizadas);
    - "dropout" de categorías: con probabilidad 0,05 una categoría se
      reemplaza por el índice de desconocido.
* Optimizador AdamW, reducción de la tasa de aprendizaje en meseta y parada
  temprana sobre la pérdida de validación (promedio de las tres etapas),
  restaurando los mejores pesos.

ACA 2 - Aprendizaje Automático | Especialización en IA - CUN
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import redes

PROB_ETAPAS = (0.2, 0.3, 0.5)
COLS_CONTINUAS_EST = 7          # las 7 primeras columnas de `est` son numéricas continuas
N_CONTINUOS_SEM = 8             # los 8 primeros rasgos de semestre son continuos (el 9.º es bandera)


def fijar_semilla(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)


def a_tensores(t: dict) -> dict:
    return {"cat": torch.as_tensor(t["cat"]), "est": torch.as_tensor(t["est"]),
            "seq": torch.as_tensor(t["seq"]), "mask": torch.as_tensor(t["mask"])}


def pesos_clase(y: np.ndarray) -> torch.Tensor:
    cnt = np.bincount(y, minlength=redes.N_CLASES)
    return torch.tensor(len(y) / (redes.N_CLASES * cnt), dtype=torch.float32)


def codificar_por_etapa(cod, df, etapa):
    return a_tensores(cod.transformar(df, etapa))


def _aumentar(lote: dict, gen: torch.Generator, ruido=0.05, p_cat=0.05, prob_etapas=PROB_ETAPAS) -> dict:
    n = lote["cat"].shape[0]
    etapa = torch.multinomial(torch.tensor(prob_etapas, dtype=torch.float), n, replacement=True, generator=gen)
    m = torch.stack([(etapa >= 1), (etapa >= 2)], dim=1).float()
    seq = lote["seq"].clone()
    est = lote["est"].clone()
    if ruido > 0:
        est[:, :COLS_CONTINUAS_EST] += ruido * torch.randn(n, COLS_CONTINUAS_EST, generator=gen)
        seq[:, :, :N_CONTINUOS_SEM] += ruido * torch.randn(n, 2, N_CONTINUOS_SEM, generator=gen)
    seq = seq * m.unsqueeze(-1)
    cat = lote["cat"].clone()
    if p_cat > 0:
        cat[torch.rand(cat.shape, generator=gen) < p_cat] = 0
    return {"cat": cat, "est": est, "seq": seq, "mask": m}


@torch.no_grad()
def predecir(modelo: nn.Module, t: dict, lote: int = 2048) -> np.ndarray:
    modelo.eval()
    n = t["cat"].shape[0]
    salida = []
    for i in range(0, n, lote):
        out = modelo(*(t[k][i:i + lote] for k in ("cat", "est", "seq", "mask")))
        if not isinstance(modelo, redes.Ensamble):
            out = torch.softmax(out, dim=1)
        salida.append(out)
    return torch.cat(salida).numpy()


def entrenar(nombre, hp, cod, df_tr, y_tr, df_va, y_va, semilla=42, epocas=250, paciencia=30,
             lr=2e-3, peso_decay=1e-4, lote=128, ruido=0.05, p_cat=0.05, aumentar=True,
             prob_etapas=PROB_ETAPAS, verbose=False):
    """Entrena una red y devuelve (modelo con los mejores pesos, historial).

    aumentar=False desactiva el ruido gaussiano y el dropout de categorías.
    prob_etapas controla el enmascaramiento de trayectoria; (0, 0, 1) entrena
    solo con el año completo (sin enmascaramiento)."""
    fijar_semilla(semilla)
    gen = torch.Generator().manual_seed(semilla)
    base = a_tensores(cod.transformar(df_tr, 2))          # se enmascara por lote
    va = {e: codificar_por_etapa(cod, df_va, e) for e in (0, 1, 2)}
    ytr, yva = torch.as_tensor(y_tr), torch.as_tensor(y_va)
    d_est, d_sem = base["est"].shape[1], base["seq"].shape[2]
    modelo = redes.construir(nombre, cod.cardinalidades(), d_est, d_sem, **hp)
    w = pesos_clase(np.asarray(y_tr))
    perdida = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    perdida_eval = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=peso_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    hist = {k: [] for k in ("perdida_entrenamiento", "perdida_validacion", "f1_validacion",
                            "exactitud_validacion", "f1_validacion_por_etapa", "lr")}
    mejor, mejor_estado, sin_mejora = np.inf, None, 0
    n = len(ytr)
    t0 = time.time()
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(n, generator=gen)
        acum = 0.0
        for i in range(0, n, lote):
            idx = perm[i:i + lote]
            b = {k: v[idx] for k, v in base.items()}
            if aumentar:
                b = _aumentar(b, gen, ruido, p_cat, prob_etapas)
            else:
                et = torch.multinomial(torch.tensor(prob_etapas, dtype=torch.float), len(idx), replacement=True,
                                       generator=gen)
                b["mask"] = torch.stack([(et >= 1), (et >= 2)], dim=1).float()
                b["seq"] = b["seq"] * b["mask"].unsqueeze(-1)
            opt.zero_grad()
            l = perdida(modelo(b["cat"], b["est"], b["seq"], b["mask"]), ytr[idx])
            l.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            acum += l.item() * len(idx)
        # validación en las tres etapas
        modelo.eval()
        with torch.no_grad():
            pv, f1s = 0.0, []
            for e in (0, 1, 2):
                logits = modelo(va[e]["cat"], va[e]["est"], va[e]["seq"], va[e]["mask"])
                pv += perdida_eval(logits, yva).item() / 3
                pred = logits.argmax(1).numpy()
                f1s.append(f1_score(y_va, pred, average="macro"))
                if e == 2:
                    acc = accuracy_score(y_va, pred)
        hist["perdida_entrenamiento"].append(acum / n)
        hist["perdida_validacion"].append(pv)
        hist["f1_validacion"].append(f1s[2])
        hist["f1_validacion_por_etapa"].append(f1s)
        hist["exactitud_validacion"].append(acc)
        hist["lr"].append(opt.param_groups[0]["lr"])
        sched.step(pv)
        if pv < mejor - 1e-4:
            mejor, mejor_estado, sin_mejora = pv, copy.deepcopy(modelo.state_dict()), 0
            hist["mejor_epoca"] = ep + 1
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
        if verbose and ep % 20 == 0:
            print(f"  época {ep + 1:>3} | pérdida tr {acum / n:.4f} | val {pv:.4f} | F1 val {f1s[2]:.3f}")
    modelo.load_state_dict(mejor_estado)
    hist["segundos"] = time.time() - t0
    hist["epocas"] = len(hist["perdida_validacion"])
    return modelo, hist


def validacion_cruzada(nombre, hp, cod_fabrica, df, y, k=5, semilla=42, **kw):
    """Validación cruzada estratificada con parada temprana interna.

    En cada pliegue: el codificador se ajusta solo con los datos de
    entrenamiento del pliegue, se separa un 15 % interno para la parada
    temprana y se predice el pliegue retenido en las tres etapas. Devuelve las
    probabilidades fuera de pliegue (OOF), los modelos y sus historiales.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
    oof = {e: np.zeros((len(y), redes.N_CLASES)) for e in (0, 1, 2)}
    modelos, historiales, cods = [], [], []
    for f, (itr, iva) in enumerate(skf.split(df, y)):
        i_fit, i_es = train_test_split(itr, test_size=0.15, stratify=y[itr], random_state=semilla + f)
        cod = cod_fabrica().ajustar(df.iloc[i_fit])
        m, h = entrenar(nombre, hp, cod, df.iloc[i_fit], y[i_fit], df.iloc[i_es], y[i_es],
                        semilla=semilla + f, **kw)
        for e in (0, 1, 2):
            oof[e][iva] = predecir(m, codificar_por_etapa(cod, df.iloc[iva], e))
        modelos.append(m)
        historiales.append(h)
        cods.append(cod)
    f1 = {e: [f1_score(y[iva], oof[e][iva].argmax(1), average="macro")
              for _, iva in skf.split(df, y)] for e in (0, 1, 2)}
    return {"oof": oof, "modelos": modelos, "historiales": historiales, "codificadores": cods,
            "f1_pliegues": f1}


def metricas(y, proba) -> dict:
    pred = proba.argmax(1)
    return {
        "Exactitud": accuracy_score(y, pred),
        "Exactitud balanceada": balanced_accuracy_score(y, pred),
        "F1 macro": f1_score(y, pred, average="macro"),
        "F1 ponderado": f1_score(y, pred, average="weighted"),
        "Precisión macro": precision_score(y, pred, average="macro", zero_division=0),
        "Sensibilidad macro": recall_score(y, pred, average="macro"),
        "Sensibilidad Deserción": recall_score(y, pred, labels=[0], average="macro"),
        "ROC-AUC (OvR macro)": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "Kappa de Cohen": cohen_kappa_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "Log-loss": float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1)))),
    }


def bootstrap_ic(y, proba, n=1000, semilla=42) -> dict:
    rng = np.random.default_rng(semilla)
    pred = proba.argmax(1)
    res = {"Exactitud": [], "F1 macro": [], "ROC-AUC (OvR macro)": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        res["Exactitud"].append(accuracy_score(y[i], pred[i]))
        res["F1 macro"].append(f1_score(y[i], pred[i], average="macro"))
        res["ROC-AUC (OvR macro)"].append(roc_auc_score(y[i], proba[i], multi_class="ovr"))
    return {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in res.items()}


def bootstrap_diferencia(y, proba_a, proba_b, n=2000, semilla=42) -> dict:
    """IC 95 % de la diferencia de F1 macro (a - b) en el mismo conjunto."""
    rng = np.random.default_rng(semilla)
    pa, pb = proba_a.argmax(1), proba_b.argmax(1)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
    d = np.asarray(d)
    return {"diferencia": float(f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")),
            "ic95": np.percentile(d, [2.5, 97.5]).tolist(), "p_a_mejor": float((d > 0).mean())}


def entrenar_ensamble(nombre, hp, cod, df, y, k=5, semilla=42, **kw):
    """Modelo final: k redes entrenadas sobre particiones distintas del
    conjunto de entrenamiento (bagging por pliegues), todas con el mismo
    codificador global. Cada una usa como parada temprana su pliegue retenido.
    Devuelve el ensamble (promedio de probabilidades) y los historiales."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla + 100)
    modelos, historiales = [], []
    for f, (itr, ies) in enumerate(skf.split(df, y)):
        m, h = entrenar(nombre, hp, cod, df.iloc[itr], y[itr], df.iloc[ies], y[ies],
                        semilla=semilla + 100 + f, **kw)
        modelos.append(m)
        historiales.append(h)
    return redes.Ensamble(modelos).eval(), historiales
