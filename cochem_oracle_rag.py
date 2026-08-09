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
from typing import List, Dict, Optional

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
    if not traceback_str or not traceback_str.strip():
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
            confidence = round(100.0 * (1.0 / (1.0 + float(dist))), 1)
            chunks.append({
                "source": meta.get("source", "Unknown"),
                "confidence": f"{confidence}%",
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
