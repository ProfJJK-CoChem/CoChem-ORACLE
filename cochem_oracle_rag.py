#!/usr/bin/env python3
"""
CoChem-ORACLE: Documentation-Grounded RAG Error Diagnostics (Suggestion 80)
Intercepts error tracebacks, queries the ChromaDB vector vault for the top-K
documentation chunks matching the error signature, and enforces manual-grounded
LLM responses for error diagnostics.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

HOME_DIR = os.path.expanduser("~")
BASE_DIR = Path(__file__).resolve().parent


def get_vault_dir() -> str:
    """Resolves the ChromaDB VAULT directory dynamically."""
    artifact_dir = os.environ.get("COCHEM_ARTIFACT_DIR") or os.environ.get("ARTIFACTS_DIR")
    if artifact_dir:
        return str(Path(artifact_dir) / "cochem_vault")
    return os.path.join(HOME_DIR, "CoChem", "cochem_vault")


VAULT_DIR = get_vault_dir()


def extract_error_signature(traceback_str: str) -> str:
    """Extracts a searchable error signature from a Python traceback string.
    
    Extracts the final exception line (e.g., 'ValueError: invalid literal')
    and key module names for RAG query construction.
    """
    if not traceback_str or not isinstance(traceback_str, str) or not traceback_str.strip():
        return ""

    lines = traceback_str.strip().split("\n")

    # Find the last exception line (typically the final line)
    error_line = ""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("File ") and not stripped.startswith("Traceback"):
            error_line = stripped
            break

    # Extract module names from 'File "..." ' lines for context
    module_names = []
    for line in lines:
        match = re.search(r'File\s+"([^"]+)"', line)
        if match:
            filename = Path(match.group(1)).stem
            if filename.startswith("cochem_"):
                module_names.append(filename)

    # Build composite search query
    query_parts = []
    if module_names:
        query_parts.append(" ".join(set(module_names)))
    if error_line:
        query_parts.append(error_line)

    return " ".join(query_parts) if query_parts else traceback_str[-200:]


def query_vault_for_error(traceback_str: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Queries ChromaDB vector vault for documentation chunks relevant to the error.
    
    Args:
        traceback_str: Full Python traceback string.
        top_k: Number of documentation chunks to retrieve.
        
    Returns:
        List of dicts with keys: 'source', 'confidence', 'content'.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        logging.warning("chromadb not installed. Error RAG diagnostics unavailable.")
        return []

    error_query = extract_error_signature(traceback_str)
    if not error_query:
        return []

    try:
        client = chromadb.PersistentClient(
            path=VAULT_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name="cochem_oracle_index")

        results = collection.query(
            query_texts=[error_query],
            n_results=top_k,
            where={"tags": {"$contains": "troubleshooting"}} if top_k <= 3 else None
        )

        if not results["documents"] or not results["documents"][0]:
            # Retry without tag filter
            results = collection.query(
                query_texts=[error_query],
                n_results=top_k
            )

        if not results["documents"] or not results["documents"][0]:
            return []

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            from cochem_oracle_engine import compute_split_conformal_interval
            interval = compute_split_conformal_interval(dist)
            chunks.append({
                "source": meta.get("source", "Unknown"),
                "confidence": f"{interval['lower_bound']}% - {interval['upper_bound']}% (95% Conformal)",
                "conformal_interval": interval,
                "content": doc
            })

        return chunks

    except Exception as e:
        logging.warning(f"Error RAG vault query failed: {e}")
        return []


def build_error_diagnostic_prompt(
    traceback_str: str,
    user_question: Optional[str] = None,
    top_k: int = 3
) -> str:
    """Builds a documentation-grounded LLM prompt for error diagnosis (Suggestion 80).
    
    Intercepts an error traceback, retrieves relevant documentation chunks,
    and constructs a prompt that enforces the LLM to ground its response
    in the retrieved documentation context.
    
    Args:
        traceback_str: The full Python traceback.
        user_question: Optional user question about the error.
        top_k: Number of documentation chunks to retrieve.
        
    Returns:
        Formatted prompt string with documentation context.
    """
    chunks = query_vault_for_error(traceback_str, top_k=top_k)

    prompt_parts = [
        "You are CoChem-ORACLE error diagnostics. You MUST answer using ONLY the "
        "provided documentation context below. If the documentation does not contain "
        "the answer, say 'The local knowledge base does not contain a resolution for this error.'\n"
    ]

    if chunks:
        prompt_parts.append("## Retrieved Documentation Context\n")
        for i, chunk in enumerate(chunks, 1):
            prompt_parts.append(
                f"### Source {i}: {chunk['source']} (Confidence: {chunk['confidence']})\n"
                f"{chunk['content']}\n"
            )
    else:
        prompt_parts.append(
            "## No Documentation Context Available\n"
            "The vector vault returned no matching documentation chunks.\n"
        )

    prompt_parts.append(f"## Error Traceback\n```\n{traceback_str}\n```\n")

    if user_question:
        prompt_parts.append(f"## User Question\n{user_question}\n")

    prompt_parts.append(
        "## Instructions\n"
        "Diagnose this error using ONLY the documentation context above. "
        "Cite the source document for each recommendation using [Source: <filename>]. "
        "If the context is insufficient, state that clearly.\n"
    )

    return "\n".join(prompt_parts)


def render_rag_error_summary(traceback_str: str) -> str:
    """Renders a standalone RAG error summary without LLM inference (for fallback mode).
    
    Used when the LLM is not loaded but the user clicks 'Ask ORACLE About Error'.
    
    Args:
        traceback_str: The full Python traceback.
        
    Returns:
        Formatted markdown string with documentation context.
    """
    chunks = query_vault_for_error(traceback_str, top_k=3)
    error_sig = extract_error_signature(traceback_str)

    parts = [
        "## ORACLE Error Diagnostic (RAG-Only Mode)\n",
        "*LLM not loaded. Showing documentation-grounded context for this error.*\n",
        f"**Detected Error:** `{error_sig}`\n",
    ]

    if chunks:
        parts.append("### Relevant Documentation\n")
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"**{i}. {chunk['source']}** (Confidence: {chunk['confidence']})\n"
                f"> {chunk['content'][:500]}{'...' if len(chunk['content']) > 500 else ''}\n"
            )
    else:
        parts.append(
            "### No Matching Documentation Found\n"
            "> Re-index the VAULT with `cochem_knowledge_sync.sync_knowledge_base()` "
            "to ensure documentation is available for error diagnostics.\n"
        )

    parts.append("---\n*Install and configure a GGUF model for full LLM-powered error diagnosis.*\n")
    return "\n".join(parts)


SPEND_PRIORITY_SEQUENCE = [
    "1. Core-valence basis set expansion",
    "2. Higher-order coupled-cluster correlation (T -> (T) -> Q)",
    "3. Relativistic corrections (DKH2/X2C)",
    "4. Anharmonicity / VPT2",
    "5. DBOC correction",
    "6. CBS extrapolation",
    "7. Frozen-core unfreezing",
    "8. Triple-to-quadruple zeta",
    "9. Diffuse function addition",
    "10. Dense grid integration"
]


def route_method_query(query_info: Dict[str, Any]) -> Dict[str, Any]:
    """Executes the Step 0–6 Method Routing Decision Procedure (§8.5).
    
    Args:
        query_info: Dictionary containing query parameters such as:
            - query (str)
            - product_class (str, optional: 'PRODUCT_A', 'PRODUCT_B', 'PRODUCT_C')
            - has_experimental_reference (bool)
            - has_template (bool)
            - is_difference_calc (bool)
            - num_atoms (int)
            - gpu_available (bool)
            - require_anharmonic_vpt2 (bool)
            - method_type (str)
            - engine (str)
            
    Returns:
        Structured result matching ORACLEResponseV4 schema.
    """
    if not isinstance(query_info, dict):
        query_info = {}

    query_str = query_info.get("query", "Method routing decision query")

    # Step 0: Product Class Selection
    has_exp = query_info.get("has_experimental_reference", False)
    has_tmpl = query_info.get("has_template", False)
    is_diff = query_info.get("is_difference_calc", False)
    explicit_class = query_info.get("product_class")

    if explicit_class in ["PRODUCT_A", "PRODUCT_B", "PRODUCT_C"]:
        product_class = explicit_class
    elif is_diff:
        product_class = "PRODUCT_C"
    elif has_exp or has_tmpl:
        product_class = "PRODUCT_B"
    else:
        product_class = "PRODUCT_A"

    # Step 1: System Size Filter
    raw_atoms = query_info.get("num_atoms", 10)
    try:
        num_atoms = int(raw_atoms) if raw_atoms is not None else 10
    except (ValueError, TypeError):
        num_atoms = 10
    size_category = "small" if num_atoms <= 10 else ("medium" if num_atoms <= 50 else "large")

    # Step 2: Hardware & Engine Capabilities
    gpu_available = query_info.get("gpu_available", False)
    engine_requested = str(query_info.get("engine", "orca")).lower()

    # Step 3: Anharmonicity Track Selection (CFOUR vs ORCA vs PySCF)
    req_vpt2 = query_info.get("require_anharmonic_vpt2", False)
    method_type = str(query_info.get("method_type", "dft")).lower()
    is_vdw = query_info.get("is_vdw_complex", False) or "van der waals" in query_str.lower() or "vdw" in query_str.lower()

    if is_vdw:
        track = "MPQC"
    elif req_vpt2 and ("coupled_cluster" in method_type or "cc" in method_type or "cfour" in engine_requested):
        track = "CFOUR"
    elif gpu_available and "pyscf" in engine_requested:
        track = "PYSCF"
    else:
        track = "ORCA"

    # Step 4: Spend Priority Sequence
    spend_priority = SPEND_PRIORITY_SEQUENCE

    # Step 5: Wall-Clock Budget Matching
    explicit_tier = query_info.get("recommended_tier")
    valid_tiers = ["T1-10s", "T1-1min", "T1-30min", "T2-1h", "T2-3h", "T2-12h", "T3-12h", "T3-1d", "T3-3d", "T4-1w", "T4-1mo"]
    if explicit_tier in valid_tiers:
        recommended_tier = explicit_tier
    elif is_vdw:
        recommended_tier = "T1-30min"
    elif product_class == "PRODUCT_A":
        if size_category == "small":
            recommended_tier = "T3-12h"
        elif size_category == "medium":
            recommended_tier = "T2-3h"
        else:
            recommended_tier = "T1-30min"
    elif product_class in ["PRODUCT_B", "PRODUCT_C"]:
        if size_category == "small":
            recommended_tier = "T1-30min"
        elif size_category == "medium":
            recommended_tier = "T1-1min"
        else:
            recommended_tier = "T1-10s"
    else:
        recommended_tier = "T2-3h"

    # Step 6: Provenance & Section 12.5 Compliance Verification
    provenance_tag = "[M]" if has_exp else ("[D]" if product_class in ["PRODUCT_B", "PRODUCT_C"] else "[E]")

    from cochem_oracle_engine import compute_split_conformal_interval, validate_section_12_5
    raw_dist = query_info.get("chroma_distance", 0.1)
    try:
        dist = float(raw_dist) if raw_dist is not None else 0.1
    except (ValueError, TypeError):
        dist = 0.1
    interval = compute_split_conformal_interval(dist)

    answer_summary = (
        f"Recommended Row: {recommended_tier} under Product Class {product_class} ({track} Track) {provenance_tag}. "
    )
    if is_vdw and track == "MPQC":
        answer_summary += "For van der Waals complexes, using MPQC with cc-pVTZ-F12 as the Tier 1 default."
    else:
        answer_summary += f"VPT2 handling routed to {track} per Section 9."

    val_res = validate_section_12_5(answer_summary)

    return {
        "query": query_str,
        "recommended_tier": recommended_tier,
        "product_class": product_class,
        "track": track,
        "provenance_tag": provenance_tag,
        "section_12_5_compliant": val_res["compliant"],
        "conformal_interval": interval,
        "spend_priority_sequence": spend_priority,
        "answer": answer_summary
    }

