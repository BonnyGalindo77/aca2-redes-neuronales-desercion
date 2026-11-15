"""
linea_base.py
-------------
Regresión logística del ACA 1 (C = 0,1, L2, pesos balanceados), reconstruida
con el mismo preprocesamiento para servir de línea base a las redes del ACA 2.
Se usa solo en el notebook (requiere scikit-learn).

ACA 2 - Aprendizaje Automático | Especialización en IA - CUN
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import preprocesamiento as pp

ESCENARIO_POR_ETAPA = {0: "ingreso", 1: "semestre1", 2: "anual"}


def pipeline_aca1(etapa: int = 2, semilla: int = 42) -> Pipeline:
    cols = pp.columnas(ESCENARIO_POR_ETAPA[etapa])
    prep = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("escalar", StandardScaler())]), cols["numericas"]),
        ("bin", SimpleImputer(strategy="most_frequent"), cols["binarias"]),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=20, sparse_output=False))]),
         cols["categoricas"]),
    ])
    return Pipeline([("seleccion", FunctionTransformer(lambda X: X[cols["entradas"]])),
                     ("ingenieria", FunctionTransformer(pp.ingenieria_caracteristicas)),
                     ("preprocesamiento", prep),
                     ("modelo", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000,
                                                   random_state=semilla))])
