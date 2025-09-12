# app.py
# Autor Original: Ricardo Urdaneta (https://github.com/Ricardouchub)

import os
import re
import json
import tempfile
from typing import List, Tuple, Dict, Any, Optional, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import process
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Compatibilidad v4/v5 para alturas de DataFrame ---
IS_V5 = int(gr.__version__.split(".", 1)[0]) >= 5
DF_H = "max_height" if IS_V5 else "height"

# --- Soporte opcional para Pydantic y PydanticAI ---
try:
    from pydantic import BaseModel, Field, ConfigDict
    HAS_PYDANTIC = True
except Exception:
    BaseModel = object  # stub
    def Field(*a, **k): return None
    class ConfigDict: pass
    HAS_PYDANTIC = False

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
# LÓGICA DE NEGOCIO (Carga, Limpieza, Perfilado, EDA, Chat)
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
        profile_df = pd.concat([
            self.final_df.dtypes.astype(str).rename("dtype"),
            miss,
            self.final_df.nunique().rename("unique_cnt")
        ], axis=1).reset_index()
        corr_fig = None
        if len(self.profile.numeric) > 1:
            corr = self.final_df[self.profile.numeric].corr(numeric_only=True)
            corr_fig = px.imshow(corr, title="Matriz de Correlación", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        self._log("✅ Reporte EDA técnico generado.")
        return {"profile_df": profile_df, "corr_fig": corr_fig}

# ----------------------------- Pydantic Models -------------------------------
if HAS_PYDANTIC:
    class Filter(BaseModel):
        model_config = ConfigDict(extra='forbid')
        column: str
        operator: Literal['==','!=','>','>=','<','<=','in','not in','contains','startswith','endswith'] = '=='
        value: Any

    class Plan(BaseModel):
        model_config = ConfigDict(extra='forbid')
        intent: Literal['top_k','trend','groupby','filter','describe','error'] = 'top_k'
        dimension: List[str] = Field(default_factory=list)
        metrics: List[str] = Field(default_factory=list)
        agg_func: Literal['sum','mean','count','min','max','median'] = 'mean'
        sort_by: Optional[str] = None
        sort_order: Literal['ascending','descending'] = 'ascending'
        limit: Optional[int] = Field(default=None, ge=1, le=100)
        filters: List[Filter] = Field(default_factory=list)
        chart_type: Optional[Literal['bar','line','scatter','pie']] = None
else:
    Filter = None
    Plan = None
# ---------------------------------------------------------------------------

class QueryProcessor:
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.client = ResourceManager.get_deepseek_client()
        self.planner_agent = self._build_planner_agent()

    # --------------------- PydanticAI Agent builder -------------------------
    def _build_planner_agent(self):
        """Crea el agente tipado si hay dependencias; si no, devuelve None."""
        if not HAS_PYDANTIC or Plan is None:
            print("[planner] Pydantic no disponible: usaré fallback a parser JSON.")
            return None
        try:
            # Puente automático DeepSeek -> OpenAI-compatible
            if os.getenv("OPENAI_API_KEY", "") == "" and os.getenv("DEEPSEEK_API_KEY", ""):
                os.environ.setdefault("OPENAI_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
                os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

            from pydantic_ai import Agent, tool  # import local para no romper si no está instalado

            @tool
            def list_schema() -> dict:
                "Esquema real del dataset actual."
                return {
                    "numeric": self.dm.profile.numeric,
                    "categorical": self.dm.profile.categorical,
                    "datetime": self.dm.profile.datetime,
                    "all": self.dm.profile.all_cols,
                }

            @tool
            def column_info(name: str) -> dict:
                "Info rápida de una columna concreta."
                if name not in self.dm.profile.all_cols:
                    return {"exists": False}
                s = self.dm.final_df[name]
                return {
                    "exists": True,
                    "dtype": str(s.dtype),
                    "unique": int(s.nunique(dropna=True)),
                    "missing_pct": float(s.isna().mean()*100),
                    "is_numeric": name in self.dm.profile.numeric,
                    "is_datetime": name in self.dm.profile.datetime,
                    "sample_values": s.dropna().astype(str).head(5).tolist(),
                }

            model_name = os.getenv("PLANNER_MODEL", "openai:deepseek-chat")  # o "openai:gpt-4o-mini"
            schema = json.dumps(Plan.model_json_schema(), ensure_ascii=False)
            system = (
                "Eres un planificador de consultas de datos. "
                "Devuelve EXCLUSIVAMENTE un objeto JSON que cumpla este esquema:\n"
                f"{schema}\n"
                "No expliques nada; solo rellena los campos del Plan."
            )
            agent = Agent(model=model_name, system_prompt=system, result_type=Plan, tools=[list_schema, column_info])
            print("[planner] PydanticAI activado con modelo:", model_name)
            return agent
        except Exception as e:
            print(f"[planner] deshabilitado -> {e}")
            return None
    # -----------------------------------------------------------------------

    def _get_api_completion(self, messages: List[Dict], temperature: float = 0.0) -> str:
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-chat", messages=messages, temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f'{{"intent": "error", "message": "Error en la API: {e}"}}'

    def generate_intelligent_eda(self) -> str:
        self.dm._log("Generando EDA inteligente con IA...")
        profile_info = (
            f"Columnas numéricas: {self.dm.profile.numeric}\n"
            f"Columnas categóricas: {self.dm.profile.categorical}\n"
            f"Columnas de fecha: {self.dm.profile.datetime}"
        )
        preview_table = self.dm.final_df.head(3).to_markdown()

        system_message = {"role": "system", "content":
            "Eres un analista de datos senior. Tu tarea es realizar un análisis exploratorio "
            "inicial (EDA) conciso y útil basado en el esquema y una vista previa de los datos. "
            "Responde en formato Markdown."
        }
        user_message_content = f"""
Aquí está el perfil de un nuevo dataset:
{profile_info}

Y una pequeña vista previa:
{preview_table}

Por favor, proporciona un resumen ejecutivo en 3-4 puntos clave. Enfócate en:
1. **Descripción General**: ¿De qué parecen tratar los datos?
2. **Columnas Clave**: ¿Cuáles parecen ser las columnas más importantes para el análisis?
3. **Sugerencias de Análisis**: Basado en las columnas, ¿qué 2 o 3 preguntas interesantes podrías hacerle a estos datos?
4. **Calidad de Datos**: ¿Observas algo evidente (ej. valores nulos en la vista previa) que merezca ser mencionado?
"""
        user_message = {"role": "user", "content": user_message_content}
        eda_analysis = self._get_api_completion([system_message, user_message], temperature=0.5)
        self.dm._log("✅ EDA inteligente generado.")
        return eda_analysis
    
    def _generate_dynamic_examples(self) -> str:
        num = self.dm.profile.numeric[0] if self.dm.profile.numeric else "valor"
        cat = self.dm.profile.categorical[0] if self.dm.profile.categorical else "categoria"
        date = self.dm.profile.datetime[0] if self.dm.profile.datetime else "fecha"
        ex1 = (
            f'Pregunta: "top 5 {cat} por suma de {num}" -> JSON: '
            f'{{"intent":"top_k","dimension":["{cat}"],"metrics":["{num}"],'
            f'"agg_func":"sum","sort_by":"{num}","sort_order":"descending","limit":5,'
            f'"filters":[],"chart_type":"bar"}}'
        )
        ex2 = (
            f'Pregunta: "evolución del promedio de {num} por {date}" -> JSON: '
            f'{{"intent":"trend","dimension":["{date}"],"metrics":["{num}"],'
            f'"agg_func":"mean","sort_by":"{date}","sort_order":"ascending","limit":null,'
            f'"filters":[],"chart_type":"line"}}'
        )
        return f"{ex1}\n{ex2}"
        
    def parse_question_to_plan(self, question: str, chat_history: List[Dict]) -> Dict[str, Any]:
        # Contexto con roles claros y sin vistas previas largas
        history_context = ""
        if chat_history:
            for turn in chat_history[-2:]:
                role = "Usuario" if turn.get("role") == "user" else "Asistente"
                content = turn.get("content", "")
                summary = content.split("\n\n**Vista Previa")[0]
                history_context += f"{role}: {summary}\n"

        examples = self._generate_dynamic_examples()
        user_message_content = (
            f"Contexto de la conversación anterior:\n{history_context}\n---\n"
            f"Esquema de columnas: num={self.dm.profile.numeric}, "
            f"cat={self.dm.profile.categorical}, date={self.dm.profile.datetime}\n"
            f"Ejemplos de salida:\n{examples}\n"
            f"Pregunta nueva: \"{question}\"\n"
            "Construye un Plan válido (ver esquema del sistema)."
        )

        # 1) Intento tipado con PydanticAI
        if self.planner_agent is not None:
            try:
                result = self.planner_agent.run_sync(user_message_content)
                plan_obj = result.data  # Plan ya validado
                return plan_obj.model_dump()
            except Exception as e:
                print(f"[planner:typed] fallo -> {e}")

        # 2) Fallback a parser JSON tradicional
        system_message = {"role": "system", "content":
            "Tu única función es traducir la pregunta a un JSON válido. "
            "No escribas nada más. Tu respuesta DEBE empezar con `{` y terminar con `}`."
        }
        user_message = {"role": "user", "content": user_message_content}
        json_str = self._get_api_completion([system_message, user_message])
        try:
            return json.loads(re.search(r'\{.*\}', json_str, re.DOTALL).group(0))
        except Exception:
            return {"intent": "error", "message": "No pude interpretar la pregunta."}
    
    # ------------ Matching híbrido de columnas (léxico + semántico) ----------
    def _match_column(self, term: str) -> Optional[str]:
        if not term or not isinstance(term, str):
            return None
        if term in self.dm.profile.all_cols:
            return term

        # Léxico
        lexical_match, lexical_score, _ = process.extractOne(term, self.dm.profile.all_cols)

        # Semántico con FAISS
        try:
            embedder = ResourceManager.get_embedder()
            q_emb = embedder.encode([term], normalize_embeddings=True).astype("float32")
            distances, indices = self.dm.col_index.search(q_emb, 1)
            semantic_match = self.dm.profile.all_cols[int(indices[0][0])]
            sim = float(distances[0][0])             # coseno en [-1, 1]
            semantic_score = max(0.0, min(1.0, (sim + 1) / 2.0)) * 100  # 0..100
        except Exception:
            semantic_match, semantic_score = lexical_match, 0.0

        if lexical_score >= 85 or semantic_score >= 85:
            return lexical_match if lexical_score >= semantic_score else semantic_match
        return lexical_match
    # -------------------------------------------------------------------------

    def canonicalize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve columnas a nombres reales y filtra inválidas."""
        plan = dict(plan)
        plan["dimension"] = [self._match_column(d) for d in plan.get("dimension", []) if self._match_column(d)]
        plan["metrics"]  = [self._match_column(m) for m in plan.get("metrics", []) if self._match_column(m)]
        if plan.get("sort_by"):
            plan["sort_by"] = self._match_column(plan["sort_by"])
        if plan.get("filters"):
            fixed = []
            for f in plan["filters"]:
                col = self._match_column(f.get("column"))
                if col:
                    f = dict(f); f["column"] = col; fixed.append(f)
            plan["filters"] = fixed
        return plan

    def validate_and_fix_plan(self, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Autocorrecciones y advertencias para robustez."""
        warnings: List[str] = []
        p = dict(plan)

        # Normalizar listas (y quitar duplicados conservando orden)
        p["dimension"] = list(dict.fromkeys(p.get("dimension", [])))
        p["metrics"] = list(dict.fromkeys(p.get("metrics", [])))

        # Quitar columnas inexistentes (por si vino del fallback)
        p = self.canonicalize_plan(p)

        # Limitar límite
        if p.get("limit") is not None:
            try:
                p["limit"] = int(max(1, min(100, int(p["limit"]))))
            except Exception:
                p["limit"] = 20
                warnings.append("`limit` inválido; se fijó en 20.")

        # Métricas numéricas cuando la agregación no es count
        if p.get("agg_func", "mean") != "count":
            numeric = set(self.dm.profile.numeric)
            fixed_metrics = [m for m in p.get("metrics", []) if m in numeric]
            if not fixed_metrics and p.get("metrics"):
                warnings.append("Las métricas solicitadas no son numéricas; se cambió a 'count'.")
                p["agg_func"] = "count"
            else:
                p["metrics"] = fixed_metrics

        # chart por defecto si falta
        if not p.get("chart_type"):
            if any(d in self.dm.profile.datetime for d in p.get("dimension", [])):
                p["chart_type"] = "line"
            elif p.get("intent") in ("top_k","groupby"):
                p["chart_type"] = "bar"

        # Evitar charts con cardinalidad desmesurada
        if p.get("chart_type") in ("bar","pie") and p.get("limit") is None:
            p["limit"] = 20

        return p, warnings

    # ------------------ Filtrado seguro (sin .query dinámico) -----------------
    def _apply_filter(self, df: pd.DataFrame, col: str, op: str, val: Any) -> pd.DataFrame:
        if col not in df.columns:
            return df
        s = df[col]
        try:
            if op == '==':   mask = s == val
            elif op == '!=': mask = s != val
            elif op == '>':  mask = s > val
            elif op == '>=': mask = s >= val
            elif op == '<':  mask = s < val
            elif op == '<=': mask = s <= val
            elif op == 'in':
                arr = val if isinstance(val, (list, tuple, set)) else [val]
                mask = s.isin(list(arr))
            elif op == 'not in':
                arr = val if isinstance(val, (list, tuple, set)) else [val]
                mask = ~s.isin(list(arr))
            elif op == 'contains':
                mask = s.astype(str).str.contains(str(val), case=False, na=False)
            elif op == 'startswith':
                mask = s.astype(str).str.startswith(str(val), na=False)
            elif op == 'endswith':
                mask = s.astype(str).str.endswith(str(val), na=False)
            else:
                return df
            return df[mask]
        except Exception as e:
            print(f"Error aplicando filtro {col} {op} {val}: {e}")
            return df
    # -------------------------------------------------------------------------

    def execute_plan(self, plan: Dict[str, Any]) -> pd.DataFrame:
        temp_df = self.dm.final_df.copy()

        # Filtros seguros
        if plan.get("filters"):
            for f in plan["filters"]:
                col, op, val = self._match_column(f.get("column")), f.get("operator"), f.get("value")
                if not all([col, op]) or (val is None and op not in ('isnull','notnull')):
                    continue
                temp_df = self._apply_filter(temp_df, col, op, val)
        
        dims = [self._match_column(d) for d in plan.get("dimension", []) if self._match_column(d)]
        metrics = [self._match_column(m) for m in plan.get("metrics", []) if self._match_column(m)]
        agg_func = plan.get("agg_func", "mean")
        
        if dims and metrics:
            temp_df = temp_df.groupby(dims).agg({m: agg_func for m in metrics}).reset_index()
        
        sort_col = self._match_column(plan.get("sort_by")) or (metrics[0] if metrics else None)
        if sort_col and sort_col in temp_df.columns:
            temp_df = temp_df.sort_values(by=sort_col, ascending=(plan.get("sort_order", "ascending") == "ascending"))
        
        if plan.get("limit"): 
            temp_df = temp_df.head(int(plan["limit"]))
        return temp_df

    def generate_summary(self, result_df: pd.DataFrame, question: str) -> str:
        messages = [
            {"role": "system", "content": "Explica resultados de forma clara y concisa en español."},
            {"role": "user", "content":
                f'Basado en mi pregunta y estos datos, redacta una respuesta breve.\n'
                f'Pregunta: "{question}"\nDatos:\n{result_df.head(5).to_markdown(index=False)}'
            }
        ]
        return self._get_api_completion(messages)

# ==============================================================================
# LÓGICA DE LA INTERFAZ (Flujos y funciones de UI)
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
        yield "Por favor, carga un dataset primero en la pestaña '1. Cargar Datos'.", gr.update(visible=False)
        return

    if session_state.query_count >= MAX_QUERIES:
        yield f"Alcanzaste el límite de {MAX_QUERIES} consultas. Carga un nuevo dataset.", gr.update(visible=False)
        return

    if message in session_state.query_cache:
        yield session_state.query_cache[message], gr.update(visible=False)
        return

    processor = QueryProcessor(session_state.data_manager)
    plan = processor.parse_question_to_plan(message, history)
    if plan.get("intent") == "error":
        yield plan.get("message"), gr.update(visible=False)
        return

    # Canonizar + validar/autocorregir
    plan = processor.canonicalize_plan(plan)
    plan, plan_warnings = processor.validate_and_fix_plan(plan)

    # Ejecutar plan y resumir
    result_df = processor.execute_plan(plan)
    summary = processor.generate_summary(result_df, message)

    # Agregar notas del plan si hubo correcciones
    if plan_warnings:
        summary += "\n\n> **Notas del plan:**\n> " + "\n> ".join(f"- {w}" for w in plan_warnings)

    # Orden natural si dimensión es datetime
    dims = plan.get("dimension", [])
    metrics = plan.get("metrics", [])
    if dims and dims[0] in processor.dm.profile.datetime:
        try:
            result_df = result_df.sort_values(by=dims[0])
        except Exception:
            pass

    # Intentar gráfico dinámico
    fig = None
    try:
        chart_type = (plan.get("chart_type") or "").lower().strip()
        if chart_type and result_df is not None and not result_df.empty and dims and metrics:
            x = dims[0]; y = metrics[0]
            if chart_type == "bar":
                fig = px.bar(result_df, x=x, y=y, title=f"{y} por {x}")
            elif chart_type == "line":
                fig = px.line(result_df, x=x, y=y, markers=True, title=f"Evolución de {y}")
            elif chart_type == "scatter":
                x_scatter = metrics[1] if len(metrics) > 1 else x
                fig = px.scatter(result_df, x=x_scatter, y=y, title=f"Relación {y} vs {x_scatter}")
            elif chart_type == "pie":
                fig = px.pie(result_df, names=x, values=y, title=f"Distribución de {y} por {x}")

        # Si no hubo figura y hay múltiples métricas, graficar multiserie
        if fig is None and chart_type in ("bar","line") and len(metrics) > 1 and dims:
            melted = result_df.melt(id_vars=[dims[0]], value_vars=metrics, var_name="serie", value_name="valor")
            if chart_type == "bar":
                fig = px.bar(melted, x=dims[0], y="valor", color="serie", barmode="group",
                             title=f"Series: {', '.join(metrics)} por {dims[0]}")
            else:
                fig = px.line(melted, x=dims[0], y="valor", color="serie", markers=True,
                              title=f"Series: {', '.join(metrics)} por {dims[0]}")
    except Exception as e:
        print(f"No se pudo generar el gráfico: {e}")
        fig = None

    session_state.increment_query_count()
    session_state.query_cache[message] = summary

    # Mostrar gráfico si existe, si no ocultarlo (sin dejar hueco)
    if fig is not None:
        yield summary, gr.update(value=fig, visible=True)
    else:
        yield summary, gr.update(visible=False)


# ==============================================================================
# CONSTRUCCIÓN DE LA INTERFAZ (Tab 1 centrado • Tab 2 50/50 full-bleed)
# ==============================================================================
custom_css = """
/* Fuente e higiene visual */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --radius-lg: 14px; }

/* Quitar fondos grises y bordes de bloques por defecto */
.gradio-container, :root {
  --block-background-fill: transparent !important;
  --block-border-color: transparent !important;
  --block-label-background-fill: transparent !important;
  --block-title-background-fill: transparent !important;
}
.gradio-container .gr-markdown,
.gradio-container .label,
.gradio-container .form {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

/* Wrapper centrado para la cabecera y Tab 1 */
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

/* Tab 2: contenedor full-bleed que ocupa todo el ancho de la ventana */
#tab2-bleed { 
  width: 100vw; 
  margin-left: calc(50% - 50vw); 
  padding: 0 20px 24px; 
}

/* Tab 2: dos columnas 50/50 */
#two-col { 
  display: grid !important; 
  grid-template-columns: 1fr 1fr; 
  gap: 16px; 
}

/* Alturas cómodas */
#analysis-card, #chat-card { min-height: 72vh; }

/* Chat: input abajo, historial crece */
#chat-card { display: flex; flex-direction: column; }
#chat-card > .gr-block { display: flex; flex-direction: column; height: 100%; }
#chat-card .gr-chat-interface { display: flex; flex-direction: column; height: 100%; }
#chat-card .gr-chatbot { flex: 1 1 auto; min-height: 0; }
#chat-card form { margin-top: auto; padding-top: 8px; }

/* Título de la sección de gráficos */
#viz-title { margin-bottom: 6px; opacity: .9; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Analista de Datos IA", css=custom_css) as demo:
    with gr.Column(elem_id="app-center"):
        # Header (centrado)
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
                with gr.Column():
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
                        build_btn = gr.Button("Construir y Analizar", variant="primary")

            # --- Tab 2: Panel de Datos (IZQ) + Chat (DER) 50/50 full-bleed ---
            with gr.TabItem("2. Chat Interactivo", id=1):
                with gr.Column(elem_id="tab2-bleed"):
                    with gr.Row(elem_id="two-col"):
                        # IZQUIERDA: Panel de análisis de datos
                        with gr.Column():
                            with gr.Group(elem_classes=["card"], elem_id="analysis-card"):
                                with gr.Accordion("📈 Panel de Datos y Análisis", open=True):
                                    gr.Markdown("#### Análisis Inteligente por IA")
                                    intelligent_eda_output = gr.Markdown("Sube un archivo para generar el análisis…")
                                    gr.Markdown("---")
                                    gr.Markdown("#### Vista Previa de los Datos")
                                    preview_df_output = gr.Dataframe(interactive=False, wrap=True, **{DF_H: 320}, show_label=False)
                                    with gr.Tabs():
                                        with gr.TabItem("EDA Técnico"):
                                            gr.Markdown("**Perfil de Columnas**")
                                            eda_profile_df = gr.Dataframe(interactive=False, **{DF_H: 260}, show_label=False)
                                            gr.Markdown("**Matriz de Correlación**")
                                            corr_plot = gr.Plot(label=None, show_label=False)
                                        with gr.TabItem("Log del Sistema"):
                                            gr.Markdown("**Registro de operaciones**")
                                            system_log_output = gr.Code(show_label=False, interactive=False)

                        # DERECHA: Chat
                        with gr.Column():
                            with gr.Group(elem_classes=["card"], elem_id="chat-card"):
                                # Título de la zona de gráficos
                                gr.Markdown("### 📊 Visualizaciones", elem_id="viz-title")

                                # El gráfico inicia oculto; se mostrará cuando haya figura
                                chat_plot = gr.Plot(label=None, show_label=False, visible=False)

                                # Título del chat (debajo de la visualización)
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
                                    additional_outputs=[chat_plot],  # conectamos el Plot
                                    title=None,
                                    description="Ejemplos: " + " · ".join([f"*{e}*" for e in examples]),
                                    type="messages",
                                )

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