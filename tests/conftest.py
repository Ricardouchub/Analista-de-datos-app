# tests/conftest.py
import os
import pytest

@pytest.fixture(autouse=True)
def _env_keys(monkeypatch):
    # Evita fallo en QueryProcessor.__init__ por falta de DEEPSEEK_API_KEY
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")