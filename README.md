# Alerta temprana de deserción estudiantil con redes neuronales

**ACA 2 · Aprendizaje Automático** · Especialización en Inteligencia Artificial · Corporación Unificada Nacional de Educación Superior (CUN)
Docente: Dr. Hamilton Rivera Flor · Grupo 7 · Noviembre de 2026

**Integrantes:** Bonny Darhyll Brayan Galindo Santos · Anternol de los Reyes Bedoya Naranjo · Andrés Alberto López Herrera · Camilo José Rodríguez Ayubi

Continuación del ACA 1 (regresión logística). Una **GRU híbrida con embeddings** lee la trayectoria de cada estudiante semestre a semestre y estima la probabilidad de *Deserción*, *Matriculado* o *Graduado*. Gracias al enmascaramiento de etapa, un solo modelo sirve para alertar al ingreso, tras el 1.er semestre y tras el 1.er año. Se compara con un MLP profundo, un FT-Transformer y la regresión logística del ACA 1 sobre los mismos 885 estudiantes de prueba.

## Estructura

```
ACA2_Redes_Neuronales_Desercion.ipynb   Notebook reproducible (pasos 1 a 5)
app.py                                  App de Streamlit (onnxruntime, sin PyTorch)
nucleo/
  base_datos.py        Base SQLite: esquema, carga, limpieza con SQL, consultas
  preprocesamiento.py  Diccionarios y reglas del ACA 1 (sin scikit-learn)
  codificador.py       Tensores de entrada: embeddings, secuencia S1→S2, máscara de etapa
  redes.py             GRU/LSTM híbrida, MLP profundo, FT-Transformer, ensamble (PyTorch)
  entrenamiento.py     Aumento de datos, entrenamiento, validación cruzada, métricas
  linea_base.py        Regresión logística del ACA 1 (línea base)
data/
  students_dropout.csv Dataset original (UCI, id 697)
  desercion.db         Base SQLite generada por el notebook
modelo/                Ensambles en ONNX, codificador.json, metadatos.json, plantilla de lote
figuras/               Gráficas generadas por el notebook
assets/                Logo CUN y diagramas (scripts/diagramas.py)
scripts/               Diagramas y póster
entregables/           Póster en PDF
```

## Cómo ejecutar

**App (local):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Notebook:** instalar `requirements-notebook.txt` y ejecutar todas las celdas desde la raíz del repositorio (en Colab, la primera celda clona el repositorio si no encuentra la carpeta `nucleo/`). La búsqueda de hiperparámetros toma unos 20 minutos en CPU. Al terminar regenera `data/desercion.db`, `modelo/` y `figuras/`.

## Dataset

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89 (licencia CC BY 4.0).
