# analista_de_datos.py
# Autor Original: Ricardo Urdaneta (https://github.com/Ricardouchub)

import os
import re
import json
import tempfile
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Compatibilidad v4/v5 para alturas de DataFrame ---
IS_V5 = int(gr.__version__.split(".", 1)[0]) >= 5
DF_H = "max_height" if IS_V5 else "height"

# ==============================================================================
# 0. CONFIGURACIÓN Y CLASES DE ESTADO
# ==============================================================================

@dataclass
class SessionState:
    data_manager: Optional['DataManager'] = None
    query_count: int = 0
    query_cache: Dict[str, Any] = field(default_factory=dict)
    
    def increment_query_count(self):
        self.query_count += 1

class ResourceManager:
    _embedder = None
    _deepseek_client = None

    @classmethod
    def get_embedder(cls) -> SentenceTransformer:
        if cls._embedder is None:
            print("Cargando modelo de embeddings...")
            cls._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return cls._embedder
    
    @classmethod
    def get_deepseek_client(cls) -> OpenAI:
        if cls._deepseek_client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("La variable de entorno DEEPSEEK_API_KEY no está configurada.")
            print("Configurando cliente de API para DeepSeek...")
            cls._deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        return cls._deepseek_client

# ==============================================================================
# 1. CLASES DE LÓGICA DE NEGOCIO
# ==============================================================================
@dataclass
class ColumnProfile:
    numeric: List[str]
    datetime: List[str]
    categorical: List[str]
    all_cols: List[str]

class DataManager:
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        self.final_df: Optional[pd.DataFrame] = None
        self.profile: Optional[ColumnProfile] = None
        self.col_index: Optional[faiss.IndexFlatIP] = None
        self.system_log: List[str] = []

    def _log(self, message: str):
        print(message)
        self.system_log.append(message)
    
    def _read_file(self, path: str) -> pd.DataFrame:
        name = os.path.basename(path)
        self._log(f"Leyendo archivo: {name}...")
        try:
            file_ext = name.lower().split('.')[-1]
            if file_ext == "csv": return pd.read_csv(path, encoding='utf-8', on_bad_lines='warn')
            elif file_ext in ["xlsx", "xls"]: return pd.read_excel(path, engine='openpyxl')
            else:
                self._log(f"ADVERTENCIA: Extensión no reconocida. Intentando leer como CSV.")
                return pd.read_csv(path, encoding='utf-8', on_bad_lines='warn')
        except Exception as e:
            raise ValueError(f"Error al leer '{name}': {e}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Iniciando limpieza de datos...")
        cleaned_df = df.copy()
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].str.strip()
        self._log("-> Espacios en blanco eliminados.")
        return cleaned_df

    def load_and_process_files(self):
        if not self.file_paths: raise ValueError("No se proporcionaron archivos.")
        all_dfs = [self._read_file(p) for p in self.file_paths]
        self._log(f"✅ Se cargaron {len(all_dfs)} archivos.")
        if len(all_dfs) == 1: self.final_df = all_dfs[0]
        else:
            merged_df = all_dfs[0]
            for i in range(1, len(all_dfs)):
                common_cols = list(set(merged_df.columns) & set(all_dfs[i].columns))
                merged_df = pd.merge(merged_df, all_dfs[i], how='outer', on=common_cols if common_cols else None)
            self.final_df = merged_df

        self.final_df = self._clean_data(self.final_df)
        self.profile = self._detect_column_roles(self.final_df)
        self.col_index = self._build_col_index(self.final_df.columns.tolist())
        self._log("✅ Dataset combinado, limpiado y perfilado.")

    def _detect_column_roles(self, df: pd.DataFrame) -> ColumnProfile:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        dt_cols = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]): dt_cols.append(c); continue
            if df[c].dtype == 'object':
                try:
                    sample = df[c].dropna().sample(min(50, len(df[c].dropna())), random_state=42)
                    if not sample.empty:
                        pd.to_datetime(sample, errors='raise')
                        df[c] = pd.to_datetime(df[c], errors='coerce')
                        if df[c].notna().sum() / len(df) > 0.5:
                            dt_cols.append(c)
                            self._log(f"INFO: Columna '{c}' convertida a fecha.")
                except (ValueError, TypeError): pass
        cat_cols = [c for c in df.columns if c not in numeric_cols and c not in dt_cols]
        return ColumnProfile(numeric=numeric_cols, datetime=dt_cols, categorical=cat_cols, all_cols=df.columns.tolist())
    
    def _build_col_index(self, columns: List[str]) -> faiss.IndexFlatIP:
        embedder = ResourceManager.get_embedder()
        embs = embedder.encode(columns, normalize_embeddings=True).astype("float32")
        idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
        return idx

    def run_eda(self) -> Dict[str, Any]:
        if self.final_df is None: return {}
        self._log("Generando reporte EDA técnico...")
        miss = self.final_df.isna().mean().mul(100).round(2).rename("missing_pct")
        profile_df = pd.concat([self.final_df.dtypes.astype(str).rename("dtype"), miss, self.final_df.nunique().rename("unique_cnt")], axis=1).reset_index()
        corr_fig = None
        if len(self.profile.numeric) > 1:
            corr = self.final_df[self.profile.numeric].corr(numeric_only=True)
            corr_fig = px.imshow(corr, title="Matriz de Correlación", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        self._log("✅ Reporte EDA técnico generado.")
        return {"profile_df": profile_df, "corr_fig": corr_fig}

class QueryProcessor:
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.client = ResourceManager.get_deepseek_client()

    def _get_api_completion(self, messages: List[Dict], temperature: float = 0.0) -> str:
        try:
            completion = self.client.chat.completions.create(model="deepseek-chat", messages=messages, temperature=temperature)
            return completion.choices[0].message.content
        except Exception as e:
            return f'{{"intent": "error", "message": "Error en la API: {e}"}}'

    def generate_intelligent_eda(self) -> str:
        self.dm._log("Generando EDA inteligente con IA...")
        profile_info = f"Columnas numéricas: {self.dm.profile.numeric}\nColumnas categóricas: {self.dm.profile.categorical}\nColumnas de fecha: {self.dm.profile.datetime}"
        preview_table = self.dm.final_df.head(3).to_markdown()

        system_message = {"role": "system", "content": "Eres un analista de datos senior. Tu tarea es realizar un análisis exploratorio inicial (EDA) conciso y útil basado en el esquema y una vista previa de los datos. Responde en formato Markdown."}
        user_message_content = f"""
Aquí está el perfil de un nuevo dataset:
{profile_info}

Y una pequeña vista previa:
{preview_table}

Por favor, proporciona un resumen ejecutivo en 3-4 puntos clave. Enfócate en:
1.  **Descripción General**: ¿De qué parecen tratar los datos?
2.  **Columnas Clave**: ¿Cuáles parecen ser las columnas más importantes para el análisis?
3.  **Sugerencias de Análisis**: Basado en las columnas, ¿qué 2 o 3 preguntas interesantes podrías hacerle a estos datos?
4.  **Calidad de Datos**: ¿Observas algo evidente (ej. valores nulos en la vista previa) que merezca ser mencionado?
"""
        user_message = {"role": "user", "content": user_message_content}
        eda_analysis = self._get_api_completion([system_message, user_message], temperature=0.5)
        self.dm._log("✅ EDA inteligente generado.")
        return eda_analysis

    def _generate_dynamic_examples(self) -> str:
        num = self.dm.profile.numeric[0] if self.dm.profile.numeric else "valor"
        cat = self.dm.profile.categorical[0] if self.dm.profile.categorical else "categoria"
        date = self.dm.profile.datetime[0] if self.dm.profile.datetime else "fecha"
        ex1 = f'Pregunta: "top 5 {cat} por suma de {num}" -> JSON: {{"intent": "top_k", "dimension": ["{cat}"], "metrics": ["{num}"], "agg_func": "sum", "sort_by": "{num}", "sort_order": "descending", "limit": 5, "filters": []}}'
        ex2 = f'Pregunta: "evolución del promedio de {num} por {date}" -> JSON: {{"intent": "trend", "dimension": ["{date}"], "metrics": ["{num}"], "agg_func": "mean", "sort_by": "{date}", "sort_order": "ascending", "limit": null, "filters": []}}'
        return f"{ex1}\n{ex2}"

    def parse_question_to_plan(self, question: str, chat_history: List[Dict]) -> Dict[str, Any]:
        history_context = ""
        if chat_history:
            for turn in chat_history[-2:]:
                history_context += f"Pregunta anterior: {turn['content']}\n"
        
        system_message = {"role": "system", "content": "Tu única función es traducir la pregunta a un JSON válido. No escribas nada más. Tu respuesta DEBE empezar con `{` y terminar con `}`."}
        user_message_content = f"Contexto: {history_context}\n---\nBasado en el contexto y la nueva pregunta, crea un plan JSON.\nEsquema: num_cols={self.dm.profile.numeric}, cat_cols={self.dm.profile.categorical}, date_cols={self.dm.profile.datetime}\nEjemplos:\n{self._generate_dynamic_examples()}\nMi Nueva Pregunta: \"{question}\""
        user_message = {"role": "user", "content": user_message_content}
        
        json_str = self._get_api_completion([system_message, user_message])
        try:
            return json.loads(re.search(r'\{.*\}', json_str, re.DOTALL).group(0))
        except Exception: return {"intent": "error", "message": "No pude interpretar la pregunta."}

    def _match_column(self, term: str) -> Optional[str]:
        if not term or not isinstance(term, str): return None
        if term in self.dm.profile.all_cols: return term
        best_match, score, _ = process.extractOne(term, self.dm.profile.all_cols)
        return best_match if score > 80 else None
        
    def execute_plan(self, plan: Dict[str, Any]) -> pd.DataFrame:
        temp_df = self.dm.final_df.copy()
        if plan.get("filters"):
            for f in plan["filters"]:
                col, op, val = self._match_column(f.get("column")), f.get("operator"), f.get("value")
                if not all([col, op, val is not None]): continue
                try: temp_df = temp_df.query(f"`{col}` {op} '{val}'" if isinstance(val, str) else f"`{col}` {op} {val}", engine='python')
                except Exception as e: print(f"Error al filtrar: {e}")
        
        dims = [self._match_column(d) for d in plan.get("dimension", []) if self._match_column(d)]
        metrics = [self._match_column(m) for m in plan.get("metrics", []) if self._match_column(m)]
        agg_func = plan.get("agg_func", "mean")
        
        if dims and metrics:
            temp_df = temp_df.groupby(dims).agg({m: agg_func for m in metrics}).reset_index()
        
        sort_col = self._match_column(plan.get("sort_by")) or (metrics[0] if metrics else None)
        if sort_col and sort_col in temp_df.columns:
            temp_df = temp_df.sort_values(by=sort_col, ascending=(plan.get("sort_order", "a") == "a"))
        
        if plan.get("limit"): temp_df = temp_df.head(int(plan["limit"]))
        return temp_df

    def generate_summary(self, result_df: pd.DataFrame, question: str) -> str:
        messages = [{"role": "system", "content": "Explica resultados de forma clara y concisa en español."},
                    {"role": "user", "content": f'Basado en mi pregunta y estos datos, redacta una respuesta breve.\nPregunta: "{question}"\nDatos:\n{result_df.head(5).to_markdown(index=False)}'}]
        return self._get_api_completion(messages)

# ==============================================================================
# 2. LÓGICA DE LA INTERFAZ (FLUJOS Y FUNCIONES)
# ==============================================================================
def build_dataset_flow(files, progress=gr.Progress(track_tqdm=True)):
    """Flujo completo para cargar, validar y analizar los archivos con feedback de progreso."""
    MAX_FILE_SIZE_MB, MAX_TOTAL_ROWS = 25, 200000
    
    try:
        progress(0, desc="Iniciando validación de archivos...")
        if not files:
            raise ValueError("Por favor, sube al menos un archivo.")

        for f in files:
            if os.path.getsize(f.name) / (1024*1024) > MAX_FILE_SIZE_MB:
                raise ValueError(f"'{os.path.basename(f.name)}' excede el límite de {MAX_FILE_SIZE_MB} MB.")
        
        dm = DataManager([f.name for f in files])
        
        progress(0.2, desc="Cargando y procesando archivos...")
        dm.load_and_process_files()
        
        if len(dm.final_df) > MAX_TOTAL_ROWS:
            raise ValueError(f"El dataset excede el límite de {MAX_TOTAL_ROWS} filas.")
        
        progress(0.6, desc="Generando EDA técnico (perfil y correlaciones)...")
        eda_results = dm.run_eda()
        
        progress(0.8, desc="Generando análisis con IA (esto puede tardar unos segundos)...")
        processor = QueryProcessor(dm)
        intelligent_eda = processor.generate_intelligent_eda()
        
        rows, cols = dm.final_df.shape
        summary_text = f"✅ **Dataset listo:** Tabla de **{rows} filas** y **{cols} columnas**."
        
        session_state = SessionState(data_manager=dm)
        
        progress(1, desc="¡Análisis completado!")
        
        return (session_state, "Análisis completado.", summary_text, "\n".join(dm.system_log),
                dm.final_df.head(10), eda_results.get("profile_df", pd.DataFrame()), eda_results.get("corr_fig"),
                intelligent_eda, gr.update(visible=True), gr.update(selected=1))

    except Exception as e:
        error_message = f"❌ Error: {e}"
        return None, error_message, "", str(e), None, None, None, None, gr.update(visible=False), gr.update(selected=0)

def chat_response_flow(message: str, history: List[Dict], session_state: SessionState):
    MAX_QUERIES = 20
    if session_state is None or session_state.data_manager is None:
        yield "Por favor, carga un dataset primero en la pestaña '1. Cargar Datos'."; return
    
    if session_state.query_count >= MAX_QUERIES:
        yield f"Alcanzaste el límite de {MAX_QUERIES} consultas. Carga un nuevo dataset."; return

    if message in session_state.query_cache:
        yield session_state.query_cache[message]; return

    processor = QueryProcessor(session_state.data_manager)
    
    plan = processor.parse_question_to_plan(message, history)
    if plan.get("intent") == "error":
        yield plan.get("message"); return
        
    result_df = processor.execute_plan(plan)
    summary = processor.generate_summary(result_df, message)
    
    session_state.increment_query_count()
    
    response = summary
    if not result_df.empty:
        table_markdown = result_df.head(10).to_markdown(index=False)
        response += f"\n\n**Vista Previa de los Resultados:**\n{table_markdown}"
    
    session_state.query_cache[message] = response
    yield response

# ==============================================================================
# 3. CONSTRUCCIÓN DE LA INTERFAZ (CENTRADA + RESIZE TAB 2)
# ==============================================================================
custom_css = """
/* Fuente e higiene visual */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --radius-lg: 14px; }

/* Quitar gris en bloques/labels/headers */
.gradio-container, :root {
  --block-background-fill: transparent !important;
  --block-border-color: transparent !important;
  --block-label-background-fill: transparent !important;
  --block-title-background-fill: transparent !important;
}
.gradio-container .gr-markdown,
.gradio-container .gr-markdown *, 
.gradio-container .label, 
.gradio-container .label-wrap,
.gradio-container .panel-header,
.gradio-container .block .header,
.gradio-container .form { 
  background: transparent !important; 
  box-shadow: none !important;
  border: none !important;
}

/* Centro de la app */
#app-center { 
  max-width: 1100px; 
  margin: 0 auto !important; 
  padding: 8px 16px 24px;
}

/* Header */
.app-header { 
  background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 60%, #9333ea 100%);
  color: white; border-radius: 16px; padding: 20px 24px; box-shadow: 0 8px 28px rgba(99,102,241,0.25);
}
.app-header h1 { margin: 0 0 6px 0; font-weight: 700; letter-spacing: -0.015em; }
.app-header p { margin: 0; opacity: 0.95; }

/* Tarjetas */
.card { 
  background: #ffffff; border-radius: var(--radius-lg);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
  padding: 18px; border: 1px solid #eef2f7;
}
.dark .card { background: #0b1220; border-color: #1f2937; }

/* Botones */
.gr-button { border-radius: 10px !important; }

/* Layout redimensionable (solo Tab 2) */
#layout {
  display: grid;
  grid-template-columns: var(--left, 1fr) 8px var(--right, 1fr);
  gap: 12px;
  align-items: stretch;
}
#panel-left, #panel-right { min-width: 260px; overflow: auto; }
#dragbar {
  width: 8px; cursor: col-resize; border-radius: 6px;
  background: linear-gradient(180deg, #e2e8f0, #cbd5e1);
}
.dark #dragbar { background: linear-gradient(180deg, #1f2937, #111827); }

/* Landing centrado aún más estrecho */
#center-landing { max-width: 900px; margin: 0 auto; }
"""

# Script para hacer draggable el separador (Tab 2)
resize_script = """
<script>
(function(){
  const root = document.getElementById('layout');
  const bar  = document.getElementById('dragbar');
  if(!root || !bar) return;
  root.style.setProperty('--left', '1fr');
  root.style.setProperty('--right','1fr');
  let dragging = false;
  bar.addEventListener('mousedown', ()=> dragging = true);
  window.addEventListener('mouseup', ()=> dragging = false);
  window.addEventListener('mousemove', (e)=>{
    if(!dragging) return;
    const rect = root.getBoundingClientRect();
    let x = e.clientX - rect.left;
    const min = 240;
    x = Math.max(min, Math.min(rect.width - min, x));
    const left = x / rect.width;
    const right = 1 - left;
    root.style.gridTemplateColumns = (left*100) + '% 8px ' + (right*100) + '%';
  });
})();
</script>
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Analista de Datos IA", css=custom_css) as demo:
    with gr.Column(elem_id="app-center"):
        # Header moderno (centrado por el wrapper)
        gr.HTML(
            """
            <div class="app-header">
              <h1>🤖 Analista de Datos IA</h1>
              <p>Sube tus datos • Haz preguntas en lenguaje natural • Obtén análisis claros y visuales</p>
            </div>
            """
        )

        app_state = gr.State()

        with gr.Tabs() as tabs:
            # --- Tab 1: Cargar Datos (centrado) ---
            with gr.TabItem("1. Cargar Datos", id=0):
                with gr.Column(elem_id="center-landing"):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 📊 Estado del pipeline")
                        status_markdown = gr.Markdown("**Estado:** Esperando archivos…")
                        summary_markdown = gr.Markdown()
                        gr.Markdown("---")
                        gr.Markdown("### 📥 Subir archivos")
                        files = gr.File(
                            show_label=False,
                            file_count="multiple",
                            file_types=[".csv", ".xlsx", ".xls"]
                        )
                        build_btn = gr.Button("🚀 Construir y Analizar", variant="primary")

            # --- Tab 2: Panel de Datos (IZQ) + Chat (DER) con resizer ---
            with gr.TabItem("2. Chat Interactivo", id=1):
                with gr.Row(elem_id="layout"):
                    # IZQUIERDA: Panel de análisis de datos
                    with gr.Column(scale=5, elem_id="panel-left"):
                        with gr.Group(elem_classes=["card"]):
                            with gr.Accordion("📈 Panel de Datos y Análisis", open=True):
                                gr.Markdown("#### Análisis Inteligente por IA")
                                intelligent_eda_output = gr.Markdown("Sube un archivo para generar el análisis…")
                                gr.Markdown("---")
                                gr.Markdown("#### Vista Previa de los Datos")
                                preview_df_output = gr.Dataframe(interactive=False, wrap=True, **{DF_H: 260}, show_label=False)
                                with gr.Tabs():
                                    with gr.TabItem("EDA Técnico"):
                                        gr.Markdown("**Perfil de Columnas**")
                                        eda_profile_df = gr.Dataframe(interactive=False, **{DF_H: 220}, show_label=False)
                                        gr.Markdown("**Matriz de Correlación**")
                                        corr_plot = gr.Plot(label=None, show_label=False)
                                    with gr.TabItem("Log del Sistema"):
                                        gr.Markdown("**Registro de operaciones**")
                                        system_log_output = gr.Code(show_label=False, interactive=False)
                    # Separador draggable
                    gr.HTML("<div id='dragbar'></div>", elem_id="dragbar")
                    # DERECHA: Chat
                    with gr.Column(scale=7, elem_id="panel-right"):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### 💬 Conversa con tus datos")
                            examples = [
                                "Top 5 productos por suma de ventas",
                                "Evolución del promedio de ingresos por mes",
                                "Filtrar país = Chile y monto entre 1000 y 2000",
                                "Comparar precio promedio por región en el tiempo"
                            ]
                            gr.ChatInterface(
                                fn=chat_response_flow,
                                additional_inputs=[app_state],
                                title=None,
                                description="Ejemplos: " + " · ".join([f"*{e}*" for e in examples]),
                                type="messages",
                            )
        # Script de resize (Tab 2)
        gr.HTML(resize_script)

    # Acción del botón (mapeo de outputs conservado)
    build_btn.click(
        fn=build_dataset_flow,
        inputs=[files],
        outputs=[
            app_state,               # state
            status_markdown,         # estado
            summary_markdown,        # resumen dataset
            system_log_output,       # log
            preview_df_output,       # preview tabla
            eda_profile_df,          # perfil columnas
            corr_plot,               # correlaciones
            intelligent_eda_output,  # EDA IA
            tabs,                    # mostrar pestañas
            tabs                     # enfocar Tab 2
        ]
    )

if __name__ == "__main__":
    try:
        ResourceManager.get_deepseek_client()
        demo.queue().launch()
    except ValueError as e:
        print(f"\n\nERROR DE INICIO: {e}")