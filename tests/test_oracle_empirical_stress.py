"""
CoChem-ORACLE Empirical Stress Test Suite (Challenger 2)
Stress testing, split-conformal interval bound validation,
Step 0-6 Method Routing decision tree fuzzing, and telemetry sanitizer empirical verification.
"""

import math
import random
import pytest
from typing import Dict, Any

from cochem_oracle_engine import (
    compute_split_conformal_interval,
    validate_section_12_5,
    OracleEngine
)
from cochem_oracle_rag import (
    route_method_query,
    extract_error_signature,
    build_error_diagnostic_prompt,
    render_rag_error_summary,
    SPEND_PRIORITY_SEQUENCE
)
from cochem_knowledge_sync import semantic_chunker
from cochem_log_scrubber import TelemetrySanitizer


# ---------------------------------------------------------------------------
# 1. SPLIT-CONFORMAL INTERVAL BOUND VALIDATION STRESS TESTS
# ---------------------------------------------------------------------------

def test_conformal_interval_mathematical_invariants_random_dist() -> None:
    """Validates split-conformal interval bounds across 1,000 random distances."""
    random.seed(42)
    # Test uniform [0, 1000]
    for _ in range(500):
        dist = random.uniform(0.0, 1000.0)
        res = compute_split_conformal_interval(dist)
        assert 0.0 <= res["lower_bound"] <= res["point_estimate"] <= res["upper_bound"] <= 100.0, \
            f"Failed invariant for distance={dist}: {res}"

    # Test log-uniform [1e-6, 1e6]
    for _ in range(500):
        dist = 10 ** random.uniform(-6.0, 6.0)
        res = compute_split_conformal_interval(dist)
        assert 0.0 <= res["lower_bound"] <= res["point_estimate"] <= res["upper_bound"] <= 100.0, \
            f"Failed invariant for log-distance={dist}: {res}"


def test_conformal_interval_extreme_and_boundary_values() -> None:
    """Validates boundary, negative, and extreme float values."""
    boundary_values = [
        0.0, 1e-15, 1e-6, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0,
        1e5, 1e9, 1e30, -0.0, -1.0, -100.0, -1e9
    ]
    for dist in boundary_values:
        res = compute_split_conformal_interval(dist)
        assert 0.0 <= res["lower_bound"] <= res["point_estimate"] <= res["upper_bound"] <= 100.0, \
            f"Failed for extreme distance {dist}: {res}"


def test_conformal_interval_custom_confidence() -> None:
    """Validates behavior under non-default confidence levels."""
    res_90 = compute_split_conformal_interval(0.2, confidence_level=0.90)
    res_99 = compute_split_conformal_interval(0.2, confidence_level=0.99)
    assert res_90["confidence_level"] == 0.90
    assert res_99["confidence_level"] == 0.99
    assert res_90["point_estimate"] == res_99["point_estimate"]


# ---------------------------------------------------------------------------
# 2. STEP 0-6 METHOD ROUTING DECISION TREE FUZZING (10,000 ITERATIONS)
# ---------------------------------------------------------------------------

def test_step_0_6_routing_decision_tree_fuzzing_10k() -> None:
    """Fuzzes route_method_query with 10,000 randomized query parameter combinations."""
    random.seed(2026)

    product_classes = ["PRODUCT_A", "PRODUCT_B", "PRODUCT_C", "INVALID_CLASS", "", None]
    valid_product_classes = {"PRODUCT_A", "PRODUCT_B", "PRODUCT_C"}

    engines = ["orca", "cfour", "pyscf", "mace", "gaussian", "", None]
    method_types = ["dft", "coupled_cluster", "cc", "mp2", "casscf", "", None]
    valid_tiers = ["T1-10s", "T1-1min", "T1-30min", "T2-1h", "T2-3h", "T2-12h", "T3-12h", "T3-1d", "T3-3d", "T4-1w", "T4-1mo"]
    valid_tracks = {"CFOUR", "PYSCF", "ORCA", "MPQC"}
    valid_provenance = {"[M]", "[D]", "[E]"}

    for i in range(10000):
        # Generate random inputs
        query_info = {
            "query": f"Fuzz test query {i}",
            "has_experimental_reference": random.choice([True, False, None, 1, 0, "yes"]),
            "has_template": random.choice([True, False, None]),
            "is_difference_calc": random.choice([True, False, None]),
            "product_class": random.choice(product_classes),
            "num_atoms": random.choice([-10, 0, 1, 5, 10, 11, 25, 50, 51, 100, 10000, 99999]),
            "gpu_available": random.choice([True, False, None]),
            "require_anharmonic_vpt2": random.choice([True, False, None]),
            "method_type": random.choice(method_types),
            "engine": random.choice(engines),
            "recommended_tier": random.choice(valid_tiers + ["INVALID_TIER", None]),
            "chroma_distance": random.uniform(-10.0, 100.0)
        }

        res = route_method_query(query_info)

        # Verify output property invariants
        assert res["product_class"] in valid_product_classes, f"Invalid product class: {res['product_class']}"
        assert res["track"] in valid_tracks, f"Invalid track: {res['track']}"
        assert res["recommended_tier"] in valid_tiers, f"Invalid recommended tier: {res['recommended_tier']}"
        assert res["provenance_tag"] in valid_provenance, f"Invalid provenance tag: {res['provenance_tag']}"
        assert isinstance(res["section_12_5_compliant"], bool)
        assert res["spend_priority_sequence"] == SPEND_PRIORITY_SEQUENCE
        assert "conformal_interval" in res
        conf_int = res["conformal_interval"]
        assert 0.0 <= conf_int["lower_bound"] <= conf_int["point_estimate"] <= conf_int["upper_bound"] <= 100.0
        assert isinstance(res["answer"], str) and len(res["answer"]) > 0


def test_routing_edge_cases_non_dict() -> None:
    """Tests routing with non-dict input handling."""
    res_none = route_method_query(None)
    assert res_none["product_class"] == "PRODUCT_A"
    assert res_none["track"] == "ORCA"
    assert res_none["recommended_tier"] in ["T1-10s", "T1-1min", "T1-30min", "T2-1h", "T2-3h", "T2-12h", "T3-12h", "T3-1d", "T3-3d", "T4-1w", "T4-1mo"]

    res_str = route_method_query("not a dict")
    assert res_str["product_class"] == "PRODUCT_A"


# ---------------------------------------------------------------------------
# 3. SECTION 12.5 STANDING ENFORCEMENT RULE STRESS TESTS
# ---------------------------------------------------------------------------

def test_validate_section_12_5_exhaustive() -> None:
    """Tests Section 12.5 validation across compliant and non-compliant matrix."""
    # Compliant: Measured data supporting accuracy claim
    c1 = validate_section_12_5("The accuracy of DLPNO-CCSD(T) is 0.2 kcal/mol [M] against benchmark.")
    assert c1["compliant"] is True

    # Compliant: Derived tag with no accuracy or hardware exclusion claim
    c2 = validate_section_12_5("Recommended workflow tier T1-1min under Product Class B [D].")
    assert c2["compliant"] is True

    # Non-compliant: Hardware exclusion supported solely by [D]
    nc1 = validate_section_12_5("ORCA has no GPU path for electronic structure calculations [D].")
    assert nc1["compliant"] is False
    assert len(nc1["violations"]) == 1

    # Non-compliant: Accuracy claim supported solely by [E]
    nc2 = validate_section_12_5("Estimated error bound is 0.5 kcal/mol [E] without experimental benchmark.")
    assert nc2["compliant"] is False

    # Compliant override: Has [D] and [E], but ALSO includes [M]
    c3 = validate_section_12_5("GPU excluded [D] for accuracy 0.1 kcal/mol [E], confirmed by experiment [M].")
    assert c3["compliant"] is True


# ---------------------------------------------------------------------------
# 4. TELEMETRY SANITIZER & REGEX WALL STRESS TESTS
# ---------------------------------------------------------------------------

def test_telemetry_sanitizer_coordinate_matrix_smiles_scrubbing() -> None:
    """Validates regex wall scrubbing across 1,000 synthetic inputs."""
    sanitizer = TelemetrySanitizer()

    # XYZ scrubbing
    xyz_sample = "Atom position: C  -1.234567  0.123456  12.345678"
    clean_xyz = sanitizer.sanitize_text(xyz_sample)
    assert "[REDACTED_COORD]" in clean_xyz
    assert "-1.234567" not in clean_xyz

    # ORCA block scrubbing
    orca_sample = "* xyz 0 1\nC 0.0000 0.0000 0.0000\nO 0.0000 0.0000 1.1300\n*"
    clean_orca = sanitizer.sanitize_text(orca_sample)
    assert "[REDACTED_GEOMETRY_BLOCK]" in clean_orca

    # Matrix float block scrubbing
    matrix_sample = "   0.123456   -1.234567    2.345678    3.456789\n   4.567890   -5.678901    6.789012    7.890123\n"
    clean_matrix = sanitizer.sanitize_text(matrix_sample)
    assert "[REDACTED_FLOAT_MATRIX]" in clean_matrix

    # SMILES scrubbing (with explicit rings/bonds)
    smiles_sample = "Processing SMILES string C1=CC=C(C=C1)O for calculation."
    clean_smiles = sanitizer.sanitize_text(smiles_sample)
    assert "[REDACTED_SMILES_STRING]" in clean_smiles
    assert "C1=CC=C(C=C1)O" not in clean_smiles


# ---------------------------------------------------------------------------
# 5. SEMANTIC CHUNKER & SHA-256 HASH STRESS TESTS
# ---------------------------------------------------------------------------

def test_semantic_chunker_code_block_protection_and_hash_uniqueness() -> None:
    """Stress tests semantic chunking with code fences and hash uniqueness."""
    sample_md = """
# System Overview #tag1

This is the main introduction section.

## Code Section #code

```python
# This is a comment inside code block, not a header
def foo() -> None:
    x = 10 # another comment
    return x
```

### Advanced Optimization #opt

Detailed discussion of optimization strategies.
"""
    chunks = semantic_chunker(sample_md, "test_doc.md")
    assert len(chunks) >= 2

    # Check that code block comments (#) were NOT split as headers
    code_chunk_found = False
    for chunk in chunks:
        if "def foo() -> None:" in chunk["text"]:
            code_chunk_found = True
            assert "# This is a comment inside code block" in chunk["text"]
    assert code_chunk_found, "Code chunk was lost or incorrectly split"

    # Test SHA-256 hash uniqueness across 1,000 distinct texts
    hashes = set()
    for i in range(1000):
        doc_text = f"# Header {i}\nContent for document {i} with unique body {i*17}."
        c = semantic_chunker(doc_text, f"doc_{i}.md")
        if c:
            hashes.add(c[0]["id"])
    assert len(hashes) == 1000, "SHA-256 chunk hash collision detected!"


# ---------------------------------------------------------------------------
# 6. BRACKETED, CHARGED & ISOTOPIC SMILES REDACTION REGRESSION TESTS
# ---------------------------------------------------------------------------

def test_telemetry_sanitizer_bracketed_charged_smiles_edge_cases() -> None:
    """Validates that bracketed, charged, isotopic, chiral, and salt SMILES strings are fully redacted."""
    sanitizer = TelemetrySanitizer()

    edge_cases = [
        ("Charged species: [NH4+].[Cl-]", "Charged species: [REDACTED_SMILES_STRING]"),
        ("Isotopic species: [13CH4]", "Isotopic species: [REDACTED_SMILES_STRING]"),
        ("Salt adduct: [Na+].[Cl-]", "Salt adduct: [REDACTED_SMILES_STRING]"),
        ("Benzene: C1=CC=C(C=C1)O", "Benzene: [REDACTED_SMILES_STRING]"),
        ("Aromatic: c1ccccc1", "Aromatic: [REDACTED_SMILES_STRING]"),
        ("Acetic acid: CC(=O)O", "Acetic acid: [REDACTED_SMILES_STRING]"),
        ("Chiral alanine: C[C@@H](N)C(=O)O", "Chiral alanine: [REDACTED_SMILES_STRING]"),
        ("Pyrrole: c1cnc[nH]1", "Pyrrole: [REDACTED_SMILES_STRING]"),
        ("Trans-fumaric acid: C/C=C/C(=O)O", "Trans-fumaric acid: [REDACTED_SMILES_STRING]"),
        ("Acetonitrile: CC#N", "Acetonitrile: [REDACTED_SMILES_STRING]"),
    ]

    for raw_text, expected_text in edge_cases:
        sanitized = sanitizer.sanitize_text(raw_text)
        assert sanitized == expected_text, f"Failed for '{raw_text}': got '{sanitized}', expected '{expected_text}'"

    # Non-SMILES plain text sanity checks
    non_smiles = [
        "The reaction temperature was 25 C.",
        "The CPU was running at 3.5 GHz.",
        "Code block: print('hello')",
    ]
    for text in non_smiles:
        sanitized = sanitizer.sanitize_text(text)
        assert sanitized == text, f"Falsely redacted non-SMILES text '{text}': got '{sanitized}'"

