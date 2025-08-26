# analista_de_datos.py — v6.0 (Final Polished)
# Autor Original: Ricardo Urdaneta (https://github.com/Ricardouchub)
# Mejoras por: Gemini (Google)

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
        # Lógica de búsqueda semántica y léxica... (simplificado)
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
def build_dataset_flow(files, progress=gr.Progress()):
    MAX_FILE_SIZE_MB, MAX_TOTAL_ROWS = 25, 200000
    if not files: return None, "Sube al menos un archivo.", "", "", None, None, None, gr.update(visible=False), gr.update(selected=0)
    progress(0, desc="Iniciando...")
    try:
        for f in files:
            if os.path.getsize(f.name) / (1024*1024) > MAX_FILE_SIZE_MB:
                raise ValueError(f"'{os.path.basename(f.name)}' excede el límite de {MAX_FILE_SIZE_MB} MB.")
        
        dm = DataManager([f.name for f in files])
        progress(0.2, desc="Cargando y procesando..."); dm.load_and_process_files()
        
        if len(dm.final_df) > MAX_TOTAL_ROWS:
            raise ValueError(f"El dataset excede el límite de {MAX_TOTAL_ROWS} filas.")
        
        progress(0.6, desc="Generando EDA técnico..."); eda_results = dm.run_eda()
        
        # ### CAMBIO ### - Llamada al EDA Inteligente
        processor = QueryProcessor(dm)
        intelligent_eda = processor.generate_intelligent_eda()
        
        rows, cols = dm.final_df.shape
        summary_text = f"✅ **Dataset listo:** Tabla de **{rows} filas** y **{cols} columnas**."
        
        session_state = SessionState(data_manager=dm)
        
        # ### CAMBIO ### - Se añade 'intelligent_eda' a los outputs
        return (session_state, "Análisis completado.", summary_text, "\n".join(dm.system_log),
                dm.final_df.head(10), eda_results.get("profile_df", pd.DataFrame()), eda_results.get("corr_fig"),
                intelligent_eda, gr.update(visible=True), gr.update(selected=1))
    except Exception as e:
        return None, f"❌ Error: {e}", "", str(e), None, None, None, None, gr.update(visible=False), gr.update(selected=0)

def chat_response_flow(message: str, history: List[Dict], session_state: SessionState):
    MAX_QUERIES = 20
    if session_state is None or session_state.data_manager is None:
        yield "Por favor, carga un dataset primero en la pestaña '1. Cargar Datos'."; return
    
    if session_state.query_count >= MAX_QUERIES:
        yield f"Alcanzaste el límite de {MAX_QUERIES} consultas. Carga un nuevo dataset."; return

    if message in session_state.query_cache:
        yield session_state.query_cache[message]; return

    processor = QueryProcessor(session_state.data_manager)
    
    plan = processor.parse_question_to_plan(message, history) # Se debe pasar el historial aquí
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
# 3. CONSTRUCCIÓN DE LA INTERFAZ
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="Analista de Datos IA") as demo:
    gr.Markdown("# 🤖 Analista de Datos IA\n### Sube tus datos → Haz preguntas en lenguaje natural → Obtén análisis al instante.")
    app_state = gr.State()

    with gr.Tabs() as tabs:
        with gr.TabItem("1. Cargar Datos", id=0):
            gr.Markdown("## Paso 1: Sube tus datos para comenzar el análisis")
            files = gr.File(label="Archivos soportados: CSV, Excel (.xlsx, .xls)", file_count="multiple", file_types=[".csv", ".xlsx", ".xls"])
            build_btn = gr.Button("🚀 Construir y Analizar", variant="primary")
            status_markdown = gr.Markdown("**Estado:** Esperando archivos...")
            summary_markdown = gr.Markdown()
        
        with gr.TabItem("2. Chat Interactivo", id=1):
            # ### CAMBIO ### - Inicio: El panel de análisis se mueve aquí
            with gr.Accordion("Panel de Datos y Análisis (clic para expandir/minimizar)", open=True):
                gr.Markdown("### Análisis Inteligente por IA")
                intelligent_eda_output = gr.Markdown("Sube un archivo para generar el análisis...")
                gr.Markdown("---")
                gr.Markdown("### Vista Previa de los Datos")
                preview_df_output = gr.Dataframe(interactive=False, wrap=True)
                with gr.Tabs():
                    with gr.TabItem("EDA Técnico"):
                        eda_profile_df = gr.Dataframe(label="Perfil de Columnas")
                        corr_plot = gr.Plot(label="Matriz de Correlación")
                    with gr.TabItem("Log del Sistema"):
                        system_log_output = gr.Code(label="Registro de operaciones", interactive=False)
            # ### CAMBIO ### - Fin: El panel de análisis se mueve aquí
            
            gr.ChatInterface(
                fn=chat_response_flow,
                additional_inputs=[app_state],
                title="Asistente de Análisis",
                description="Haz preguntas como '¿Cuál es el top 5 de productos por ventas?' o 'Muéstrame la evolución de los ingresos'.",
                type="messages"
            )

    # ### CAMBIO ### - El .click ahora actualiza los componentes en la segunda pestaña
    build_btn.click(
        fn=build_dataset_flow,
        inputs=[files],
        outputs=[app_state, status_markdown, summary_markdown, system_log_output,
                 preview_df_output, eda_profile_df, corr_plot,
                 intelligent_eda_output, tabs, tabs] # Se actualiza el 'tabs' para forzar el cambio
    )

if __name__ == "__main__":
    try:
        ResourceManager.get_deepseek_client()
        # Para despliegue en HF Spaces, la autenticación se maneja en los settings del Space.
        demo.queue().launch()
    except ValueError as e:
        print(f"\n\nERROR DE INICIO: {e}")