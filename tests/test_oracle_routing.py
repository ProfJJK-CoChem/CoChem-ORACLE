import os
import json
import pytest
from pathlib import Path
from cochem_oracle_rag import route_method_query, SPEND_PRIORITY_SEQUENCE

BASE_DIR = Path(__file__).resolve().parent.parent

def test_route_method_query_product_a() -> None:
    res = route_method_query({
        "query": "De novo calculation of unknown conformer",
        "has_experimental_reference": False,
        "num_atoms": 8,
        "require_anharmonic_vpt2": True,
        "method_type": "coupled_cluster"
    })
    assert res["product_class"] == "PRODUCT_A"
    assert res["track"] == "CFOUR"
    assert res["recommended_tier"] == "T3-12h"
    assert res["provenance_tag"] == "[E]"
    assert res["section_12_5_compliant"] is True
    assert "conformal_interval" in res
    assert res["spend_priority_sequence"] == SPEND_PRIORITY_SEQUENCE

def test_route_method_query_product_b() -> None:
    res = route_method_query({
        "query": "Semi-experimental structure calculation",
        "has_experimental_reference": True,
        "num_atoms": 25,
        "require_anharmonic_vpt2": False,
        "method_type": "dft"
    })
    assert res["product_class"] == "PRODUCT_B"
    assert res["track"] == "ORCA"
    assert res["recommended_tier"] == "T1-1min"
    assert res["provenance_tag"] == "[M]"
    assert res["section_12_5_compliant"] is True

def test_route_method_query_product_c() -> None:
    res = route_method_query({
        "query": "Isomer energy difference",
        "is_difference_calc": True,
        "num_atoms": 12,
        "gpu_available": True,
        "engine": "pyscf"
    })
    assert res["product_class"] == "PRODUCT_C"
    assert res["track"] == "PYSCF"
    assert res["recommended_tier"] == "T1-1min"
    assert res["provenance_tag"] == "[D]"
    assert res["section_12_5_compliant"] is True

def test_route_method_query_vdw() -> None:
    res = route_method_query({
        "query": "Study of a Van der Waals complex",
        "is_difference_calc": False,
        "num_atoms": 20,
        "gpu_available": False,
        "engine": "orca"
    })
    assert res["track"] == "MPQC"
    assert res["recommended_tier"] == "T1-30min"
    assert "cc-pVTZ-F12" in res["answer"]

def test_schema_validity() -> None:
    schema_path = BASE_DIR / "cochem_oracle_response.json"
    assert schema_path.exists()
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.loads(f.read())
    assert schema["title"] == "ORACLEResponseV4"
    assert "required" in schema
    for key in ["query", "recommended_tier", "product_class", "track", "provenance_tag", "section_12_5_compliant", "answer"]:
        assert key in schema["required"]
