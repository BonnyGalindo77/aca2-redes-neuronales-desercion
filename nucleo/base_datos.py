"""
base_datos.py
-------------
Gestión del dataset en una base de datos SQLite (Paso 1 del flujo).

El CSV plano de la UCI (una fila por estudiante, 37 columnas) se lleva a un
esquema relacional:

    programas   (codigo, nombre)
    cohortes    (cohorte_id, tasa_desempleo, tasa_inflacion, pib)
    estudiantes (estudiante_id, datos de ingreso y socioeconómicos, cohorte_id,
                 sin_registro_curricular, resultado, particion)
    semestres   (estudiante_id, semestre, convalidadas, inscritas,
                 evaluaciones, aprobadas, nota, sin_evaluacion)   <- formato largo
    predicciones(bitácora de las predicciones hechas desde la app)

La tabla `semestres` guarda la trayectoria en formato largo (dos filas por
estudiante). De ahí se arman con SQL las secuencias que recibe la red
recurrente. Las reglas de limpieza del ACA 1 se aplican aquí con sentencias
UPDATE para que queden documentadas dentro de la propia base.

ACA 2 - Aprendizaje Automático | Especialización en IA - CUN
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocesamiento as pp

ESTATICAS = ["estado_civil", "modo_ingreso", "orden_opcion", "programa", "jornada_diurna",
             "formacion_previa", "nota_formacion_previa", "educacion_madre", "educacion_padre",
             "ocupacion_madre", "ocupacion_padre", "nota_admision", "desplazado",
             "necesidades_especiales", "deudor", "matricula_al_dia", "genero_masculino", "becado",
             "edad_ingreso", "internacional"]
VARS_SEMESTRE = ["convalidadas", "inscritas", "evaluaciones", "aprobadas", "nota", "sin_evaluacion"]

ESQUEMA = """
DROP TABLE IF EXISTS semestres;
DROP TABLE IF EXISTS estudiantes;
DROP TABLE IF EXISTS cohortes;
DROP TABLE IF EXISTS programas;

CREATE TABLE programas (
    codigo  INTEGER PRIMARY KEY,
    nombre  TEXT NOT NULL
);

CREATE TABLE cohortes (
    cohorte_id      INTEGER PRIMARY KEY,
    tasa_desempleo  REAL NOT NULL,
    tasa_inflacion  REAL NOT NULL,
    pib             REAL NOT NULL,
    UNIQUE (tasa_desempleo, tasa_inflacion, pib)
);

CREATE TABLE estudiantes (
    estudiante_id           INTEGER PRIMARY KEY,
    estado_civil            INTEGER,
    modo_ingreso            INTEGER,
    orden_opcion            INTEGER,
    programa                INTEGER REFERENCES programas(codigo),
    jornada_diurna          INTEGER CHECK (jornada_diurna IN (0, 1)),
    formacion_previa        INTEGER,
    nota_formacion_previa   REAL,
    educacion_madre         INTEGER,
    educacion_padre         INTEGER,
    ocupacion_madre         INTEGER,
    ocupacion_padre         INTEGER,
    nota_admision           REAL,
    desplazado              INTEGER CHECK (desplazado IN (0, 1)),
    necesidades_especiales  INTEGER CHECK (necesidades_especiales IN (0, 1)),
    deudor                  INTEGER CHECK (deudor IN (0, 1)),
    matricula_al_dia        INTEGER CHECK (matricula_al_dia IN (0, 1)),
    genero_masculino        INTEGER CHECK (genero_masculino IN (0, 1)),
    becado                  INTEGER CHECK (becado IN (0, 1)),
    edad_ingreso            INTEGER,
    internacional           INTEGER CHECK (internacional IN (0, 1)),
    cohorte_id              INTEGER REFERENCES cohortes(cohorte_id),
    sin_registro_curricular INTEGER DEFAULT 0,
    resultado               TEXT CHECK (resultado IN ('Deserción', 'Matriculado', 'Graduado')),
    particion               TEXT CHECK (particion IN ('entrenamiento', 'prueba'))
);

CREATE TABLE semestres (
    estudiante_id   INTEGER NOT NULL REFERENCES estudiantes(estudiante_id),
    semestre        INTEGER NOT NULL CHECK (semestre IN (1, 2)),
    convalidadas    REAL,
    inscritas       REAL,
    evaluaciones    REAL,
    aprobadas       REAL,
    nota            REAL,
    sin_evaluacion  REAL,
    PRIMARY KEY (estudiante_id, semestre)
);

CREATE INDEX idx_est_programa  ON estudiantes(programa);
CREATE INDEX idx_est_resultado ON estudiantes(resultado);
CREATE INDEX idx_est_particion ON estudiantes(particion);

"""

TABLA_PREDICCIONES = """
CREATE TABLE IF NOT EXISTS predicciones (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha            TEXT DEFAULT (datetime('now', 'localtime')),
    origen           TEXT,
    etapa            TEXT,
    prob_desercion   REAL,
    prob_matriculado REAL,
    prob_graduado    REAL,
    clase            TEXT,
    nivel_riesgo     TEXT,
    entrada_json     TEXT
)"""

# Reglas de limpieza del ACA 1 expresadas en SQL
LIMPIEZA_SQL = {
    "Marcar estudiantes sin registro curricular (0 inscritas en S1 y S2)": """
        UPDATE estudiantes SET sin_registro_curricular = 1
        WHERE estudiante_id IN (
            SELECT estudiante_id FROM semestres
            GROUP BY estudiante_id HAVING SUM(inscritas) = 0)""",
    "Convertir sus ceros estructurales en NULL": """
        UPDATE semestres
        SET convalidadas = NULL, inscritas = NULL, evaluaciones = NULL,
            aprobadas = NULL, nota = NULL, sin_evaluacion = NULL
        WHERE estudiante_id IN (SELECT estudiante_id FROM estudiantes
                                WHERE sin_registro_curricular = 1)""",
    "Anular promedios < 10 con unidades aprobadas (se aprueba con 10/20)": """
        UPDATE semestres SET nota = NULL WHERE aprobadas > 0 AND nota < 10""",
}

# Vista plana (una fila por estudiante) reconstruida con agregación condicional
VISTA_PLANA = """
DROP VIEW IF EXISTS v_estudiantes_plano;
CREATE VIEW v_estudiantes_plano AS
SELECT e.*, c.tasa_desempleo, c.tasa_inflacion, c.pib,
{pivot}
FROM estudiantes e
JOIN cohortes c USING (cohorte_id)
JOIN semestres s USING (estudiante_id)
GROUP BY e.estudiante_id;
"""


def _pivot_sql() -> str:
    partes = [f"    MAX(CASE WHEN s.semestre = {k} THEN s.{v} END) AS s{k}_{v}"
              for k in (1, 2) for v in VARS_SEMESTRE]
    return ",\n".join(partes)


def conectar(ruta_db: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(ruta_db))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def crear_base(ruta_csv: str | Path, ruta_db: str | Path, semilla: int = 42) -> dict:
    """Crea la base SQLite desde el CSV original de la UCI y devuelve un
    resumen de las filas cargadas y de las reglas de limpieza aplicadas."""
    from sklearn.model_selection import train_test_split  # solo se usa al construir la base

    crudo = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="utf-8-sig")
    crudo.columns = [c.strip().replace("\t", "") for c in crudo.columns]
    df = crudo.rename(columns=pp.RENOMBRAR)
    df["resultado"] = df["resultado"].map(pp.CLASES)
    df.insert(0, "estudiante_id", np.arange(1, len(df) + 1))

    # Misma partición 80/20 estratificada (semilla 42) del ACA 1: así la prueba
    # es exactamente el mismo grupo de 885 estudiantes y la comparación es justa.
    idx_tr, idx_te = train_test_split(df.index, test_size=0.20, stratify=df["resultado"],
                                      random_state=semilla)
    df["particion"] = "entrenamiento"
    df.loc[idx_te, "particion"] = "prueba"

    Path(ruta_db).parent.mkdir(parents=True, exist_ok=True)
    con = conectar(ruta_db)
    con.executescript(ESQUEMA)
    con.execute("DROP TABLE IF EXISTS predicciones")
    con.execute(TABLA_PREDICCIONES)

    # Tabla de preparación con el CSV tal cual llega
    crudo.to_sql("uci_crudo", con, if_exists="replace", index=False)

    con.executemany("INSERT INTO programas VALUES (?, ?)", list(pp.PROGRAMAS.items()))

    coh = (df[["tasa_desempleo", "tasa_inflacion", "pib"]].drop_duplicates()
           .sort_values(["tasa_desempleo", "tasa_inflacion", "pib"]).reset_index(drop=True))
    coh.insert(0, "cohorte_id", np.arange(1, len(coh) + 1))
    coh.to_sql("cohortes", con, if_exists="append", index=False)
    df = df.merge(coh, on=["tasa_desempleo", "tasa_inflacion", "pib"], how="left")

    df[["estudiante_id"] + ESTATICAS + ["cohorte_id", "resultado", "particion"]].to_sql(
        "estudiantes", con, if_exists="append", index=False)

    largo = pd.concat([
        df[["estudiante_id"] + [f"s{k}_{v}" for v in VARS_SEMESTRE]]
        .rename(columns={f"s{k}_{v}": v for v in VARS_SEMESTRE}).assign(semestre=k)
        for k in (1, 2)])
    largo[["estudiante_id", "semestre"] + VARS_SEMESTRE].to_sql(
        "semestres", con, if_exists="append", index=False)

    aplicadas = {}
    for desc, sql in LIMPIEZA_SQL.items():
        cur = con.execute(sql)
        aplicadas[desc] = cur.rowcount
    con.executescript(VISTA_PLANA.format(pivot=_pivot_sql()))
    con.commit()

    resumen = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
               for t in ("uci_crudo", "programas", "cohortes", "estudiantes", "semestres")}
    con.close()
    return {"filas": resumen, "limpieza": aplicadas}


def consultar(ruta_db: str | Path, sql: str, params: tuple | dict = ()) -> pd.DataFrame:
    con = conectar(ruta_db)
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def cargar_plano(ruta_db: str | Path, particion: str | None = None) -> pd.DataFrame:
    """Devuelve la vista plana (una fila por estudiante), opcionalmente filtrada."""
    sql = "SELECT * FROM v_estudiantes_plano"
    if particion:
        sql += " WHERE particion = :p"
    sql += " ORDER BY estudiante_id"
    return consultar(ruta_db, sql, {"p": particion} if particion else ())


def cargar_trayectorias(ruta_db: str | Path) -> pd.DataFrame:
    """Trayectoria en formato largo, ordenada por estudiante y semestre: es la
    materia prima de las secuencias de la red recurrente."""
    return consultar(ruta_db, """
        SELECT s.*, e.sin_registro_curricular
        FROM semestres s JOIN estudiantes e USING (estudiante_id)
        ORDER BY s.estudiante_id, s.semestre""")


def registrar_prediccion(ruta_db: str | Path, fila: dict) -> None:
    con = conectar(ruta_db)
    try:
        con.execute(TABLA_PREDICCIONES)
        cols = ", ".join(fila)
        marcas = ", ".join(f":{k}" for k in fila)
        con.execute(f"INSERT INTO predicciones ({cols}) VALUES ({marcas})", fila)
        con.commit()
    finally:
        con.close()
