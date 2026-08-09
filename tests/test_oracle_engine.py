import pytest
from cochem_oracle_engine import OracleEngine

def test_oracle_engine_instantiation():
    engine = OracleEngine()
    assert engine.is_active == False
    assert engine.chat_history == []

def test_oracle_engine_activate_deactivate():
    engine = OracleEngine()
    activated = engine.activate()
    assert engine.is_active == True
    engine.deactivate()
    assert engine.is_active == False
