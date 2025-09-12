# tests_plans.py
import os
import pandas as pd
import numpy as np
import pytest

# Importa desde tu app
import app as appmod

# ---------- Utilidades para armar un entorno mínimo de prueba ----------
def make_dm():
    """Crea un DataManager con un DataFrame pequeño y construye perfil + índice."""
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03","2024-01-04"]),
        "producto": ["A","B","A","B"],
        "region": ["Chile","Peru","Chile","Argentina"],
        "ventas": [100, 200, 150, 50],
        "comentario": ["ok", "ok", "atraso", "ok"]
    })
    dm = appmod.DataManager(file_paths=[])
    dm.final_df = df
    dm.profile = dm._detect_column_roles(df)
    dm.col_index = dm._build_col_index(dm.final_df.columns.tolist())
    return dm

class FakePlan:
    """Simula un objeto Plan de Pydantic (suficiente para .model_dump())."""
    def __init__(self, d): self._d = d
    def model_dump(self): return dict(self._d)

class FakePlannerAgent:
    """Stub del planner para no llamar a LLM."""
    def __init__(self, plan_dict):
        self._plan = plan_dict
    def run_sync(self, prompt: str):
        class Result: pass
        r = Result()
        r.data = FakePlan(self._plan)
        return r

# ----------------------------------------------------------------------

def test_match_column_hibrido():
    dm = make_dm()
    qp = appmod.QueryProcessor(dm)
    qp.planner_agent = None  # no usamos el agente real

    # 'venta' debería matchear con 'ventas' (rapidfuzz/FAISS)
    col = qp._match_column("venta")
    assert col in dm.profile.all_cols

def test_canonicalize_and_validate_fix():
    dm = make_dm()
    qp = appmod.QueryProcessor(dm); qp.planner_agent = None

    # Plan pide métrica no-numérica -> se corrige a count
    plan = {
        "intent": "groupby",
        "dimension": ["region"],
        "metrics": ["comentario"],   # no numérica
        "agg_func": "mean",          # inválido para texto
        "chart_type": None,
        "limit": 999                 # fuera de rango razonable
    }
    plan = qp.canonicalize_plan(plan)
    fixed, warns = qp.validate_and_fix_plan(plan)

    assert fixed["agg_func"] == "count"
    assert fixed["limit"] <= 100
    assert any("no son numéricas" in w for w in warns)

def test_filters_seguro_contains():
    dm = make_dm()
    qp = appmod.QueryProcessor(dm); qp.planner_agent = None

    df = dm.final_df.copy()
    # debería quedarse con filas cuya region contenga "chi" -> Chile
    df2 = qp._apply_filter(df, "region", "contains", "chi")
    assert set(df2["region"].unique()) == {"Chile"}

def test_execute_plan_groupby_sum_sorted():
    dm = make_dm()
    qp = appmod.QueryProcessor(dm); qp.planner_agent = None

    plan = {
        "intent": "groupby",
        "dimension": ["producto"],
        "metrics": ["ventas"],
        "agg_func": "sum",
        "sort_by": "ventas",
        "sort_order": "descending",
        "limit": 10,
        "filters": []
    }
    plan = qp.canonicalize_plan(plan)
    df = qp.execute_plan(plan)
    # A: 100+150=250, B: 200+50=250 -> empate; orden estable, columnas presentes
    assert set(df.columns) >= {"producto", "ventas"}
    assert len(df) == 2

def test_planner_stub_end_to_end():
    dm = make_dm()
    qp = appmod.QueryProcessor(dm)

    # Simulamos un plan válido que devuelve el agente
    stub_plan = {
        "intent": "trend",
        "dimension": ["fecha"],
        "metrics": ["ventas"],
        "agg_func": "mean",
        "sort_by": "fecha",
        "sort_order": "ascending",
        "limit": None,
        "filters": [],
        "chart_type": "line"
    }
    qp.planner_agent = FakePlannerAgent(stub_plan)

    # No debería caer en fallback ni llamar al LLM real
    plan = qp.parse_question_to_plan("evolución de ventas por fecha", chat_history=[])
    plan = qp.canonicalize_plan(plan)
    fixed, warns = qp.validate_and_fix_plan(plan)

    df = qp.execute_plan(fixed)
    assert not df.empty
    assert fixed["chart_type"] == "line"

# --------- Si quieres correr sin pytest: python tests_plans.py ----------
if __name__ == "__main__":
    # Mini "runner" sin pytest (por si no quieres instalarlo)
    try:
        test_match_column_hibrido(); print("OK test_match_column_hibrido")
        test_canonicalize_and_validate_fix(); print("OK test_canonicalize_and_validate_fix")
        test_filters_seguro_contains(); print("OK test_filters_seguro_contains")
        test_execute_plan_groupby_sum_sorted(); print("OK test_execute_plan_groupby_sum_sorted")
        test_planner_stub_end_to_end(); print("OK test_planner_stub_end_to_end")
        print("Todos los tests pasaron ✅")
    except AssertionError as e:
        print("Algún test falló ❌:", e)