#!/usr/bin/env python3
"""
Adversarial Stress Testing & Fuzzing Suite for CoChem-ORACLE.
Executed by Challenger 1 for Gate M8 Iteration 1 empirical verification.
"""

import math
import pytest
import tempfile
import os
import json
import asyncio
from pathlib import Path

from cochem_oracle_engine import (
    OracleEngine,
    compute_split_conformal_interval,
    validate_section_12_5
)
from cochem_oracle_rag import (
    extract_error_signature,
    query_vault_for_error,
    build_error_diagnostic_prompt,
    render_rag_error_summary,
    route_method_query
)
from cochem_log_scrubber import TelemetrySanitizer, export_telemetry
from cochem_knowledge_sync import semantic_chunker, sync_knowledge_base


# =====================================================================
# 1. Stress Testing validate_section_12_5
# =====================================================================

def test_fuzz_validate_section_12_5_non_string_types() -> None:
    """Fuzz validate_section_12_5 with non-string input types."""
    res_none = validate_section_12_5(None)
    assert res_none["compliant"] is True

    res_int = validate_section_12_5(12345)
    assert res_int["compliant"] is True

    res_list = validate_section_12_5(["hardware exclusion", "[D]"])
    assert res_list["compliant"] is False
    assert len(res_list["violations"]) == 1


def test_fuzz_validate_section_12_5_edge_claims() -> None:
    """Test boundary combinations of claims and provenance tags."""
    # Only [D] tag with exclusion -> violation
    res1 = validate_section_12_5("We exclude GPU path because of high memory [D].")
    assert res1["compliant"] is False
    assert len(res1["violations"]) == 1

    # [D] and [M] tag together -> compliant (has [M])
    res2 = validate_section_12_5("We exclude GPU path based on estimated [D] and measured [M] data.")
    assert res2["compliant"] is True

    # Accuracy claim with [E] tag solely -> violation
    res3 = validate_section_12_5("The accuracy is within 0.1 kcal/mol [E].")
    assert res3["compliant"] is False

    # Neutral text -> compliant
    res4 = validate_section_12_5("Calculation completed successfully at T2-3h tier.")
    assert res4["compliant"] is True


# =====================================================================
# 2. Stress Testing compute_split_conformal_interval
# =====================================================================

def test_fuzz_compute_split_conformal_interval_edge_values() -> None:
    """Fuzz split conformal prediction interval with extreme values."""
    # Negative distance -> handled via max(0.0, ...)
    res_neg = compute_split_conformal_interval(-5.0)
    assert res_neg["lower_bound"] >= 0.0
    assert res_neg["upper_bound"] <= 100.0

    # Zero distance
    res_zero = compute_split_conformal_interval(0.0)
    assert res_zero["point_estimate"] == 100.0
    assert res_zero["lower_bound"] == 95.0
    assert res_zero["upper_bound"] == 100.0

    # Huge finite distance
    res_huge = compute_split_conformal_interval(1e9)
    assert res_huge["lower_bound"] == 0.0

    # Infinity input handled gracefully
    res_inf = compute_split_conformal_interval(float('inf'))
    assert res_inf["lower_bound"] == 95.0
    assert res_inf["upper_bound"] == 100.0

    # NaN input handled gracefully
    res_nan = compute_split_conformal_interval(float('nan'))
    assert isinstance(res_nan, dict)

    # String input handled gracefully
    res_str = compute_split_conformal_interval("invalid_dist")
    assert res_str["lower_bound"] == 95.0


# =====================================================================
# 3. Stress Testing route_method_query
# =====================================================================

def test_fuzz_route_method_query_invalid_inputs() -> None:
    """Fuzz route_method_query with invalid, missing, and non-standard query dicts."""
    # Non-dict input
    res_none = route_method_query(None)
    assert isinstance(res_none, dict)
    assert "recommended_tier" in res_none

    res_int = route_method_query(99999)
    assert isinstance(res_int, dict)

    # Missing all optional keys
    res_empty = route_method_query({})
    assert res_empty["product_class"] == "PRODUCT_A"
    assert res_empty["recommended_tier"] == "T3-12h"

    # Extreme atom counts: negative, zero, millions
    res_neg_atoms = route_method_query({"num_atoms": -10})
    assert isinstance(res_neg_atoms, dict)

    res_huge_atoms = route_method_query({"num_atoms": 1_000_000, "product_class": "PRODUCT_A"})
    assert res_huge_atoms["recommended_tier"] == "T1-30min"

    # Non-integer num_atoms handled gracefully
    res_str_atoms = route_method_query({"num_atoms": "50", "product_class": "PRODUCT_A"})
    assert res_str_atoms["recommended_tier"] == "T2-3h"

    res_none_atoms = route_method_query({"num_atoms": None})
    assert isinstance(res_none_atoms, dict)

    # Invalid non-numeric chroma_distance handled gracefully
    res_bad_dist = route_method_query({"chroma_distance": "NOT_A_FLOAT"})
    assert isinstance(res_bad_dist, dict)


# =====================================================================
# 4. Stress Testing Error RAG Diagnostics
# =====================================================================

def test_fuzz_extract_error_signature() -> None:
    """Fuzz error signature extraction."""
    # None or empty string
    assert extract_error_signature(None) == ""
    assert extract_error_signature("") == ""
    assert extract_error_signature("   \n  ") == ""

    # Non-string input handled gracefully
    assert extract_error_signature(12345) == ""

    # Realistic traceback
    tb = """Traceback (most recent call last):
  File "cochem_oracle_engine.py", line 42, in ask_oracle
    raise ValueError("Invalid configuration for ORCA track")
ValueError: Invalid configuration for ORCA track"""
    sig = extract_error_signature(tb)
    assert "cochem_oracle_engine" in sig
    assert "ValueError: Invalid configuration for ORCA track" in sig


def test_fuzz_query_vault_for_error_unpopulated() -> None:
    """Test RAG query against unpopulated / missing ChromaDB vault."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["COCHEM_ARTIFACT_DIR"] = tmpdir
        tb = "ValueError: Test error in unpopulated vault"
        
        # Querying vault when no collection exists should return empty list gracefully
        chunks = query_vault_for_error(tb, top_k=3)
        assert isinstance(chunks, list)
        assert len(chunks) == 0

        # Render summary in RAG fallback mode
        summary = render_rag_error_summary(tb)
        assert "No Matching Documentation Found" in summary or "ORACLE Error Diagnostic" in summary

        # Build error diagnostic prompt
        prompt = build_error_diagnostic_prompt(tb, user_question="How to fix?")
        assert "No Documentation Context Available" in prompt or "ValueError" in prompt


# =====================================================================
# 5. Stress Testing OracleEngine Async & Activation
# =====================================================================

def test_fuzz_oracle_engine_lifecycle() -> None:
    """Stress test OracleEngine lifecycle, queries, and fallback rendering."""
    async def _run() -> None:
        engine = OracleEngine()
        assert not engine.is_active

        # Asking dormant engine should return dormant message
        responses = []
        async for token in engine.ask_oracle("How to run ORCA?"):
            responses.append(token)
        assert "dormant" in "".join(responses).lower()

        # Activate engine (RAG fallback mode if no GGUF model)
        activated = engine.activate()
        assert activated
        assert engine.is_active

        # Ask query in fallback mode
        fallback_responses = []
        async for token in engine.ask_oracle("What is Product Class A?"):
            fallback_responses.append(token)
        full_resp = "".join(fallback_responses)
        assert "RAG-Only Response" in full_resp or "ORACLE" in full_resp

        # Deactivate engine
        engine.deactivate()
        assert not engine.is_active

    asyncio.run(_run())


# =====================================================================
# 6. Stress Testing Log Scrubber & Telemetry Sanitizer
# =====================================================================

def test_fuzz_telemetry_sanitizer() -> None:
    """Stress test TelemetrySanitizer with edge cases."""
    sanitizer = TelemetrySanitizer()

    # None input
    assert sanitizer.sanitize_text(None) is None
    assert sanitizer.sanitize_text("") == ""

    # Malformed chat history with non-dict elements handled gracefully
    bad_chat = [
        {"role": "user", "content": None},
        "not_a_dict"
    ]
    clean_chat = sanitizer.scrub_chat_log(bad_chat)
    assert len(clean_chat) == 1
    assert clean_chat[0]["content"] == ""


# =====================================================================
# 7. Stress Testing Knowledge Sync & Chunker
# =====================================================================

def test_fuzz_semantic_chunker() -> None:
    """Fuzz semantic chunker with code fences, missing headers, and strange markdown."""
    text_with_code = """# Header 1
Some documentation text.

```python
# This is a comment inside Python code block
def foo() -> None:
    raise NotImplementedError("Implementation pending")
```

## Header 2
More text #tag1 #tag2.
"""
    chunks = semantic_chunker(text_with_code, "test_doc.md")
    assert len(chunks) >= 1
    # Code block hash comment should not split the chunk
    full_chunk_text = " ".join([c["text"] for c in chunks])
    assert "def foo() -> None:" in full_chunk_text
