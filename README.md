# XAI Project – Explainable Machine Learning Pipelines

Este proyecto implementa una arquitectura modular para el entrenamiento, evaluación y **explicación de modelos de Machine Learning**, integrando técnicas clásicas de _Explainable AI_ (SHAP y LIME) junto con una **capa interpretativa basada en LLMs**.
El objetivo principal es estudiar y comparar explicaciones locales y globales en distintos datasets, facilitando la reproducibilidad y extensibilidad del pipeline.

Actualmente se trabaja con los datasets **BCF** y **TIRESIA**, utilizando modelos basados en **Random Forest** y explicaciones generadas automáticamente.

---

## Objetivos del proyecto

### Objetivo general

Explorar el uso de **modelos de lenguaje (LLMs) como una capa interpretativa sobre métodos de Explainable AI (XAI)**, integrándolos en pipelines de Machine Learning para generar **explicaciones en lenguaje natural** a partir de salidas técnicas como SHAP y LIME.

---

### Objetivos específicos

- Implementar **pipelines reproducibles de ML + XAI** sobre datasets tabulares (BCF y TIRESIA).
- Entrenar modelos y generar explicaciones **locales y globales** con SHAP y LIME.
- Integrar un **LLM** que interprete las salidas XAI y produzca explicaciones en lenguaje natural.
- Analizar de forma cualitativa la **coherencia y utilidad** de las explicaciones generadas por el LLM.


---

## Descripción de los módulos principales

### `src/pipelines/`

Orquestación completa del flujo de trabajo.

- `bcf_pipeline.py`: pipeline end-to-end para el dataset BCF.
- `tiresia_pipeline.py`: pipeline end-to-end para el dataset TIRESIA.

Cada pipeline integra:

1. Carga de datos
2. Preprocesamiento
3. Entrenamiento del modelo
4. Generación de explicaciones
5. Interpretación mediante LLM

---

## Notebooks de experimentación

- `BCF_experiments.ipynb`: ejecución interactiva de experimentos sobre el dataset BCF.
- `TIRESIA_experiments.ipynb`: análisis y evaluación del pipeline aplicado a TIRESIA.

Estos notebooks funcionan como soporte experimental y validación de los pipelines implementados en `src/`.

---

## Requisitos e instalación

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

Configurar las variables de entorno en un archivo `.env` (por ejemplo, claves de acceso a APIs de LLM).

### Configuración de Gemini API Key

1. Copiar `.env.example` a `.env` en la raíz del proyecto.
2. Definir la clave:

```env
GEMINI_API_KEY=tu_api_key_aqui
```

`src/llm/gemini_explainer.py` carga automáticamente esta variable para inicializar el cliente de Gemini.
