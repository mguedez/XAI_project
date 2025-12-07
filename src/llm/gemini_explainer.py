# src/llm/gemini_explainer.py

import os
from dotenv import load_dotenv
from typing import Any
from google import genai

PROMPT_MAESTRO = """
    Objetivo general:
    Actuás como un experto en química computacional y XAI. Tu tarea es transformar explicaciones técnicas 
    de modelos QSAR (clasificación o regresión) en explicaciones claras, accionables y adaptadas a usuarios 
    con diferentes niveles de formación (beginner / intermediate).

    Contexto del dominio:
    Los modelos analizan propiedades químicas (por ejemplo: logP, TPSA, Peso Molecular, #H-donors, 
    #H-acceptors, aromaticidad, heteroátomos, fracciones alifáticas, etc.) para predecir:
    - Clases de bioconcentración (1, 2, 3)
    - Toxicidad (tiresia)
    - Valores continuos como logBCF

    Las explicaciones provienen de XAI:
    - SHAP global (importancia de cada descriptor)
    - SHAP local (contribuciones positivas/negativas por molécula)
    - LIME local (aproximación lineal por instancia)

    Antes de responder SIEMPRE debés preguntar:
    1. Nivel de expertise del usuario (beginner / intermediate)
    2. Dominio (ej.: química, toxicología, IA, modelado, farmacia)
    Si no se especifica, asumí beginner + dominio químico general.

    Reglas clave:
    - No repetir literalmente valores del gráfico (números, etiquetas) a menos que el usuario lo pida.
    - No describir “lo que se ve” en el gráfico si es obvio.
    - Enfocarte en interpretar lo que significa químicamente.
    - Agrupar insights pequeños en conclusiones más amplias (patterns químicos).
    - Priorizar claridad y conexión con la estructura de la molécula.
    - Usar "Let's think step by step" para razonar internamente.
    - No inventar propiedades químicas que no estén en el input.

    Si el usuario es beginner:
    - NO explicar cómo funciona SHAP o LIME.
    - Presentá SOLO los insights importantes en lenguaje muy simple.
    - Explicá cómo afectan las propiedades químicas a la predicción sin tecnicismos.

    Si el usuario es intermediate:
    - Explicá brevemente cómo interpretar SHAP/LIME para este problema.
    - Luego da insights químicos más profundos.
    - Podés mencionar relaciones típicas QSAR (logP, polaridad, hidrofobicidad, fragmentos aromáticos...).

    Salida esperada:
    Explicaciones claras y útiles que respondan: 
    “¿Qué está influyendo realmente en esta predicción y por qué tiene sentido químico?”
    Evitar descripciones visuales obvias.
    Dar recomendaciones prácticas si corresponde 
    (ej.: “para reducir la bioacumulación, la tendencia sería…”).

    La lista de los descriptores químicos usados es:
    - nHM: número de átomos de hidrógeno
    - piPC09: índice de polaridad
    - PCD: complejidad del compuesto
    - X2Av: conectividad del átomo 2 promedio
    - MLOGP: logaritmo del coeficiente de partición octanol/agua
    - ON1V: volumen polarizable
    - N-072: número de átomos de nitrógeno
    - B02[C-N]: conteo de pares de átomos C-N a distancia 2
    - F04[C-O]: conteo de pares de átomos C-O a distancia 4
"""

load_dotenv()
def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no está configurada en el entorno.")
    client = genai.Client(api_key=api_key)
    return client

def explicar_con_gemini(
    shap_local: Any,
    shap_global: Any,
    lime_exp: Any,
    pred: Any,
    task_type: str,
    model_name: str = "gemini-2.5-flash",
    expertise_level: str = "beginner",
    domain: str = "química",
) -> str:
    """
    Versión modular de tu función explicar_con_gemini del notebook BCF.
    En lugar de hacer print streaming + input interactivo, devolvemos el texto.
    Podés adaptar a tus necesidades.
    """
    client = _get_client()
    chat = client.chats.create(model=model_name)

    prompt = f"""
    {PROMPT_MAESTRO}

    Ahora te paso los valores de esta predicción:

    Tarea: {task_type}
    Predicción del modelo: {pred}
    Nivel de expertise del usuario: {expertise_level}
    Dominio del usuario: {domain}

    SHAP local:
    {shap_local}

    SHAP global:
    {shap_global}

    Explicación LIME:
    {lime_exp}
    """

    response = chat.send_message(prompt)
    return response.text
