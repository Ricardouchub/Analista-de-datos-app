<!-- README.md -->
<p align="center">
  <img width="1600" height="400" alt="image" src="https://github.com/user-attachments/assets/1d0e6dce-4520-4d0a-803c-0f9f14692f80" />
</p>

<h1 align="center">🤖 Analista de Datos IA</h1>
<p align="center">
  Universal Chat-with-Data • Joins automáticos (exact/fuzzy) • EDA inteligente
</p>

<p align="center">
  <!-- Badges principales -->
  <img src="https://img.shields.io/badge/Estado-Activo-2ECC71?style=flat-square" alt="Estado"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Gradio-5.43.1-F58025?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HF Spaces"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Frames-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Plotly-Visualizaci%C3%B3n-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"/>
  <img src="https://img.shields.io/badge/Sentence--Transformers-Embeddings-0E83CD?style=flat-square" alt="Sentence Transformers"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-1E90FF?style=flat-square" alt="FAISS"/>
  <img src="https://img.shields.io/badge/DeepSeek-API-8A2BE2?style=flat-square" alt="DeepSeek"/>
  <img src="https://img.shields.io/badge/Licencia-MIT-000000?style=flat-square" alt="MIT"/>
</p>

<p align="center">
  <!-- Reemplaza la URL del Space cuando lo publiques -->
  <a href="https://huggingface.co/spaces/Ricardouchub/analista-de-datos-ia">
    <img src="https://img.shields.io/badge/Abrir_en-HF_Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black" alt="Open in Spaces">
  </a>
</p>

---

## ✨ ¿Qué hace?
- **Chat-with-Data universal**: sube **CSV/Excel**, pregunta en lenguaje natural y obtén respuestas con **tablas y gráficos**.
- **Joins automáticos**: sugiere y aplica uniones **exactas y fuzzy** entre múltiples archivos (detección de claves por similitud y cardinalidad).
- **EDA inteligente**: panel automático con **perfil de columnas**, **missingness**, **correlaciones** y un **resumen generado con IA**.
- **Caché ligera** y **límite de consultas** por sesión.

---

## 🧱 Stack
- **UI**: Gradio 5.43.1  
- **Datos**: pandas, pyarrow  
- **Visualización**: Plotly  
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)  
- **Búsqueda**: FAISS (CPU)  
- **IA**: DeepSeek Chat API (para parsear intención y resúmenes)
