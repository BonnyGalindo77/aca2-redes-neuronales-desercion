"""
codificador.py
--------------
Convierte la tabla de estudiantes en los tensores que reciben las redes
neuronales (Paso 2 del flujo). Solo usa numpy y pandas, para que la app de
Streamlit lo pueda importar sin instalar PyTorch.

Cada estudiante se representa con cuatro bloques:

    cat   [n, 8]      índices enteros de 8 variables categóricas -> capas de embedding
    est   [n, 15]     7 numéricas estandarizadas + 8 binarias (contexto de ingreso)
    seq   [n, 2, 9]   trayectoria: un vector de 9 rasgos por semestre (S1, S2)
    mask  [n, 2]      1 si el semestre ya se conoce, 0 si todavía no

La máscara permite usar el mismo modelo en las tres etapas de alerta:
    etapa 0 = al ingreso (sin semestres), 1 = tras el 1.er semestre, 2 = tras el 1.er año.

Todas las estadísticas (medianas, medias, desviaciones, vocabularios) se
calculan solo con el conjunto de entrenamiento y se guardan en JSON.

ACA 2 - Aprendizaje Automático | Especialización en IA - CUN
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocesamiento as pp

CATEGORICAS = ["estado_civil", "modo_ingreso", "programa", "nivel_formacion_previa",
               "nivel_educ_madre", "nivel_educ_padre", "grupo_ocup_madre", "grupo_ocup_padre"]
NUMERICAS = ["orden_opcion", "nota_formacion_previa", "nota_admision", "edad_ingreso",
             "tasa_desempleo", "tasa_inflacion", "pib"]
BINARIAS = ["jornada_diurna", "desplazado", "necesidades_especiales", "deudor",
            "matricula_al_dia", "genero_masculino", "becado", "internacional"]
VARS_SEM = ["convalidadas", "inscritas", "evaluaciones", "aprobadas", "nota", "sin_evaluacion"]
CONTEOS = ["convalidadas", "inscritas", "evaluaciones", "aprobadas", "sin_evaluacion"]
RASGOS_SEM = VARS_SEM + ["tasa_aprobacion", "tasa_exito_evaluaciones", "sin_registro"]
ETAPAS = {0: "Al ingreso (sin notas)", 1: "Tras el 1.er semestre", 2: "Tras el 1.er año"}
MIN_FRECUENCIA = 20  # igual que en el ACA 1: categorías con < 20 casos se agrupan


def _clave(v) -> str:
    """Clave de vocabulario estable: 171, 171.0 y "171" se tratan igual."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    return str(v)


def _derivar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los niveles/grupos agrupados del ACA 1 si no están."""
    X = df.copy()
    if "nivel_educ_madre" not in X.columns:
        X["nivel_educ_madre"] = X["educacion_madre"].map(pp.COD_A_NIVEL)
        X["nivel_educ_padre"] = X["educacion_padre"].map(pp.COD_A_NIVEL)
        X["nivel_formacion_previa"] = X["formacion_previa"].map(pp.COD_A_NIVEL)
        X["grupo_ocup_madre"] = X["ocupacion_madre"].map(pp.grupo_ocupacion)
        X["grupo_ocup_padre"] = X["ocupacion_padre"].map(pp.grupo_ocupacion)
    return X


def _rasgos_semestre(X: pd.DataFrame, s: int) -> pd.DataFrame:
    """Rasgos de un semestre con nombres genéricos (sin prefijo s1_/s2_)."""
    r = pd.DataFrame({v: pd.to_numeric(X.get(f"s{s}_{v}"), errors="coerce") for v in VARS_SEM},
                     index=X.index)
    faltante = r[["inscritas", "aprobadas"]].isna().any(axis=1)
    if "sin_registro_curricular" in X.columns:
        faltante |= X["sin_registro_curricular"].fillna(0).astype(int).eq(1)
    r["tasa_aprobacion"] = np.where(r["inscritas"] > 0, r["aprobadas"] / r["inscritas"].clip(lower=1), 0.0)
    r["tasa_exito_evaluaciones"] = np.where(r["evaluaciones"] > 0,
                                            r["aprobadas"] / r["evaluaciones"].clip(lower=1), 0.0)
    r.loc[faltante, ["tasa_aprobacion", "tasa_exito_evaluaciones"]] = np.nan
    r["sin_registro"] = faltante.astype(float)
    return r


class CodificadorTabular:
    """Ajusta y aplica la codificación de entrada de las redes neuronales."""

    def __init__(self):
        self.vocab: dict[str, dict[str, int]] = {}
        self.med_num: dict[str, float] = {}
        self.mu_num: dict[str, float] = {}
        self.sd_num: dict[str, float] = {}
        self.moda_bin: dict[str, float] = {}
        self.med_sem: dict[str, float] = {}
        self.mu_sem: dict[str, float] = {}
        self.sd_sem: dict[str, float] = {}

    # ------------------------------------------------------------------ ajuste
    def ajustar(self, df: pd.DataFrame) -> "CodificadorTabular":
        X = _derivar(df)
        for c in CATEGORICAS:
            frec = X[c].dropna().map(_clave).value_counts()
            frecuentes = sorted(frec[frec >= MIN_FRECUENCIA].index)
            # índice 0 = infrecuente / desconocido / faltante
            self.vocab[c] = {v: i + 1 for i, v in enumerate(frecuentes)}
        for c in NUMERICAS:
            v = pd.to_numeric(X[c], errors="coerce")
            self.med_num[c] = float(v.median())
            v = v.fillna(self.med_num[c])
            self.mu_num[c], self.sd_num[c] = float(v.mean()), float(v.std() or 1.0)
        for c in BINARIAS:
            self.moda_bin[c] = float(pd.to_numeric(X[c], errors="coerce").mode().iloc[0])
        # Estadísticas de semestre agrupando S1 y S2 (el mismo rasgo significa lo mismo)
        sem = pd.concat([_rasgos_semestre(X, 1), _rasgos_semestre(X, 2)])
        for c in RASGOS_SEM[:-1]:
            v = sem[c]
            self.med_sem[c] = float(v.median())
            v = v.fillna(self.med_sem[c])
            if c in CONTEOS:
                v = np.log1p(v)
            self.mu_sem[c], self.sd_sem[c] = float(v.mean()), float(v.std() or 1.0)
        return self

    # --------------------------------------------------------------- transformar
    def transformar(self, df: pd.DataFrame, etapa: int | np.ndarray = 2) -> dict:
        X = _derivar(df)
        n = len(X)
        cat = np.zeros((n, len(CATEGORICAS)), dtype=np.int64)
        for j, c in enumerate(CATEGORICAS):
            cat[:, j] = X[c].map(lambda v: self.vocab[c].get(_clave(v), 0) if pd.notna(v) else 0).to_numpy()
        num = np.column_stack([
            ((pd.to_numeric(X[c], errors="coerce").fillna(self.med_num[c]) - self.mu_num[c]) / self.sd_num[c])
            .to_numpy() for c in NUMERICAS])
        binv = np.column_stack([pd.to_numeric(X[c], errors="coerce").fillna(self.moda_bin[c]).to_numpy()
                                for c in BINARIAS])
        est = np.hstack([num, binv]).astype(np.float32)

        seq = np.zeros((n, 2, len(RASGOS_SEM)), dtype=np.float32)
        for k, s in enumerate((1, 2)):
            r = _rasgos_semestre(X, s)
            cols = []
            for c in RASGOS_SEM[:-1]:
                v = r[c].fillna(self.med_sem[c])
                if c in CONTEOS:
                    v = np.log1p(v.clip(lower=0))
                cols.append(((v - self.mu_sem[c]) / self.sd_sem[c]).to_numpy())
            cols.append(r["sin_registro"].to_numpy())
            seq[:, k, :] = np.column_stack(cols)

        etapa = np.broadcast_to(np.asarray(etapa), (n,)).astype(int)
        mask = np.column_stack([(etapa >= 1), (etapa >= 2)]).astype(np.float32)
        seq = seq * mask[:, :, None]  # lo que aún no ha ocurrido no se ve
        return {"cat": cat, "est": est, "seq": seq.astype(np.float32), "mask": mask}

    # ----------------------------------------------------------- persistencia
    def a_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("vocab", "med_num", "mu_num", "sd_num", "moda_bin", "med_sem", "mu_sem", "sd_sem")}

    def guardar(self, ruta: str | Path) -> None:
        Path(ruta).write_text(json.dumps(self.a_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: str | Path) -> "CodificadorTabular":
        obj = cls()
        for k, v in json.loads(Path(ruta).read_text(encoding="utf-8")).items():
            setattr(obj, k, v)
        return obj

    def cardinalidades(self) -> list[int]:
        """Tamaño de cada tabla de embedding (vocabulario + índice 0)."""
        return [len(self.vocab[c]) + 1 for c in CATEGORICAS]


def aplanar(t: dict) -> np.ndarray:
    """Versión plana (sin categóricas) para el MLP: est + seq aplanada + máscara."""
    n = t["est"].shape[0]
    return np.hstack([t["est"], t["seq"].reshape(n, -1), t["mask"]]).astype(np.float32)
