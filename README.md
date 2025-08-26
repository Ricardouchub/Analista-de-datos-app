<!-- README.md -->
<p align="center">
  <img width="1600" height="400" alt="image" src="img/banner.png" />
</p>


<p align="center">
  <!-- Badges principales -->
  <img src="https://img.shields.io/badge/Estado-Completado-2ECC71?style=flat-square" alt="Estado"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Gradio-5.43.1-F58025?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HF Spaces"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Frames-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Plotly-Visualizaci%C3%B3n-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"/>
  <img src="https://img.shields.io/badge/Sentence--Transformers-Embeddings-0E83CD?style=flat-square" alt="Sentence Transformers"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-1E90FF?style=flat-square" alt="FAISS"/>
  <img src="https://img.shields.io/badge/DeepSeek-API-8A2BE2?style=flat-square" alt="DeepSeek"/>
</p>

Este proyecto es un potente Analista de Datos IA que permite conversar con tus archivos CSV o Excel. Sube tus datos y la aplicación se encarga del resto: limpieza automática, perfilado, uniones entre tablas y un Análisis Exploratorio (EDA) generado por IA. Haz preguntas complejas en lenguaje natural y recibe respuestas claras, tablas y gráficos dinámicos al instante.


---

## Cómo Funciona 

El flujo de trabajo de la aplicación sigue estos pasos:

* **Carga y Limpieza**: El usuario sube los archivos. pandas los lee y aplica una limpieza automática.
* **Perfilado y EDA**: Se identifican los tipos de columnas `numéricas`, `categóricas` y `fechas`. Se genera un perfil técnico. Inmediatamente, se hace una llamada a la IA para generar el EDA Inteligente.
* **Chat y Planificación**: El usuario hace una pregunta. La pregunta, el historial del chat y el esquema de los datos se envían a la IA, que devuelve un plan de acción en formato JSON.
* **Ejecución**: El plan JSON se traduce a una cadena de operaciones de pandas (.query(), .groupby(), .agg(), etc.).
* **Respuesta y Visualización**: El resultado se presenta al usuario como un resumen en texto, una tabla en Markdown o un gráfico generado con Plotly.

---

## Features Principales

* **Chat-with-Data Universal**: Sube uno o más archivos CSV/Excel y empieza a preguntar.
* **Joins Automáticos**: Si subes múltiples archivos, la app detecta claves comunes y sugiere uniones.
* **Inteligencia Artificial Avanzada**: Usa la API de DeepSeek para:
* **EDA Inteligente**: Genera un resumen ejecutivo de tus datos apenas los subes.
* **Text-to-Plan**: Convierte tus preguntas en un plan de ejecución para pandas.
* **Memoria Conversacional**: Recuerda el contexto de tus últimas preguntas.
* **Búsqueda Semántica de Columnas**: Entiende a qué te refieres aunque no uses el nombre exacto de la columna (ej. "ganancias" vs "beneficio_neto") gracias a Sentence-Transformers y FAISS.
* **Visualización Dinámica**: Genera gráficos (barras, líneas, etc.) cuando la pregunta lo amerita.
* **Seguridad y Eficiencia**: Incluye caché de consultas, límites de uso por sesión y un sistema de cola para gestionar múltiples usuarios.

---

## Stack

**Interfaz**: Gradio

**Backend**: Python

**Manipulación de Datos**: pandas

**Visualización**: Plotly

**IA (LLM)**: DeepSeek

**Embeddings de Texto**: sentence-transformers

**Búsqueda Vectorial**: FAISS (Facebook AI Similarity Search)

---

## Screenshots

<img width="1002" height="1193" alt="image" src="img/main.png" />

<img width="1002" height="1193" alt="image" src="img/test.png" />


---

## Autor

**Ricardo Urdaneta**

**[Linkedin](https://www.linkedin.com/in/ricardourdanetacastro)**
