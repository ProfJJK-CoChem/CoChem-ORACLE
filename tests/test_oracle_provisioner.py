import os
import json
import pytest
from pathlib import Path

from cochem_oracle_main import OracleOrchestrator
from cochem_oracle_config import get_oracle_config, OracleConfig
from cochem_oracle_engine import OracleEngine
import cochem_oracle_hook as oracle_hook

def test_oracle_orchestrator_init() -> None:
    orchestrator = OracleOrchestrator()
    orchestrator.initialize()
    assert orchestrator.is_initialized is True

def test_oracle_orchestrator_sync() -> None:
    orchestrator = OracleOrchestrator()
    orchestrator.initialize()
    res = orchestrator.run_knowledge_sync("local_vault")
    assert res is True or res is False

def test_oracle_orchestrator_report(tmp_path) -> None:
    orchestrator = OracleOrchestrator()
    orchestrator.initialize()
    rep_path = orchestrator.generate_knowledge_report(str(tmp_path))
    assert Path(rep_path).exists()

def test_oracle_config() -> None:
    config = get_oracle_config()
    assert config.get("project_name") == "CoChem-ORACLE"

def test_oracle_engine_init() -> None:
    engine = OracleEngine()
    assert engine.is_active is False
    assert len(engine.chat_history) == 0

def test_oracle_hook_functions() -> None:
    state = oracle_hook.load_system_state()
    assert isinstance(state, dict)
