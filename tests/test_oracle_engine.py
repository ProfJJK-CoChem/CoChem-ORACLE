import pytest
from cochem_oracle_engine import (
    OracleEngine,
    compute_split_conformal_interval,
    validate_section_12_5
)

def test_oracle_engine_instantiation() -> None:
    engine = OracleEngine()
    assert engine.is_active == False
    assert engine.chat_history == []

def test_oracle_engine_activate_deactivate() -> None:
    engine = OracleEngine()
    activated = engine.activate()
    assert engine.is_active == True
    engine.deactivate()
    assert engine.is_active == False

def test_oracle_engine_system_prompt_v4() -> None:
    engine = OracleEngine()
    assert "[M]" in engine.system_prompt
    assert "[D]" in engine.system_prompt
    assert "[E]" in engine.system_prompt
    assert "Section 12.5" in engine.system_prompt
    assert "Product Class" in engine.system_prompt
    assert "Track" in engine.system_prompt

def test_split_conformal_interval() -> None:
    res = compute_split_conformal_interval(0.2, confidence_level=0.95)
    assert res["confidence_level"] == 0.95
    assert "lower_bound" in res
    assert "upper_bound" in res
    assert "point_estimate" in res
    assert res["lower_bound"] <= res["point_estimate"] <= res["upper_bound"]

def test_validate_section_12_5_compliant() -> None:
    text = "The accuracy is 0.1 kcal/mol [M] measured via W4 theory."
    res = validate_section_12_5(text)
    assert res["compliant"] is True
    assert len(res["violations"]) == 0
    assert res["warning"] == ""

def test_validate_section_12_5_non_compliant() -> None:
    text = "GPU excluded due to accuracy being 2.0 kcal/mol [D]."
    res = validate_section_12_5(text)
    assert res["compliant"] is False
    assert len(res["violations"]) > 0
    assert "Section 12.5 Rule 7 Violation" in res["warning"]
