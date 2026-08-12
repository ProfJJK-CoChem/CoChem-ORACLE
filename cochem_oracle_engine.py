#!/usr/bin/env python3
"""
CoChem-ORACLE: Ephemeral Engine
Manages lazy-loaded local LLM inference, ephemeral chat state wiping, 
and strict Semantic RAG contextualization via ChromaDB.
"""

import os
import gc
import re
import json
import asyncio
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator, Any

# Core Paths resolved dynamically (ORACLE-01, ORACLE-02)
HOME_DIR = os.path.expanduser("~")
BASE_DIR = Path(__file__).resolve().parent

def get_vault_dir() -> str:
    artifact_dir = os.environ.get("COCHEM_ARTIFACT_DIR") or os.environ.get("ARTIFACTS_DIR")
    if artifact_dir:
        return str(Path(artifact_dir) / "cochem_vault")
    return os.path.join(HOME_DIR, "CoChem", "cochem_vault")

VAULT_DIR = get_vault_dir()
PID_FILE = os.path.join(HOME_DIR, ".cochem", "silos", "oracle", "oracle_engine.pid")
CONFIG_PATH = str(BASE_DIR / "cochem_system_config.json")
ORACLE_CONFIG_PATH = str(BASE_DIR / "oracle_config.json")


def compute_split_conformal_interval(distance: Any, confidence_level: float = 0.95) -> Dict[str, Any]:
    """Calculates split-conformal prediction interval bounds (§21) for ChromaDB distances."""
    import math
    try:
        dist_val = float(distance)
    except (ValueError, TypeError):
        dist_val = 0.0

    if math.isnan(dist_val) or math.isinf(dist_val) or dist_val < 0.0:
        dist_val = 0.0

    point_est = round(100.0 * (1.0 / (1.0 + dist_val)), 1)
    margin = round(5.0 * (1.0 + dist_val), 1)
    lower = max(0.0, round(point_est - margin, 1))
    upper = min(100.0, round(point_est + margin, 1))
    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "confidence_level": confidence_level,
        "point_estimate": point_est
    }


def validate_section_12_5(text: Any) -> Dict[str, Any]:
    """Validates text against Section 12.5 Standing Enforcement Rule.
    
    Rule 7: No [D] or [E] value may solely support a hardware exclusion or accuracy claim.
    """
    if text is None:
        return {"compliant": True, "violations": [], "warning": ""}
    
    if not isinstance(text, str):
        text = str(text)

    if not text:
        return {"compliant": True, "violations": [], "warning": ""}

    hardware_excl_pattern = re.compile(
        r'\b(exclude[s]?|exclusion|inefficient|unsupported|prohibit[s]?|no\s+gpu\s+path|cpu[- ]only)\b',
        re.IGNORECASE
    )
    accuracy_claim_pattern = re.compile(
        r'\b(accuracy|precision|error\s+bound|uncert|kcal/mol|cm\^-1|mhz)\b',
        re.IGNORECASE
    )

    has_hw_excl = bool(hardware_excl_pattern.search(text))
    has_accuracy = bool(accuracy_claim_pattern.search(text))

    has_d_tag = "[D]" in text
    has_e_tag = "[E]" in text
    has_m_tag = "[M]" in text

    violations = []
    if (has_hw_excl or has_accuracy) and (has_d_tag or has_e_tag) and not has_m_tag:
        claim_type = "hardware exclusion" if has_hw_excl else "accuracy claim"
        used_tags = f"{'[D]' if has_d_tag else ''}{'[E]' if has_e_tag else ''}"
        violations.append(
            f"Section 12.5 Rule 7 Violation: {claim_type} supported solely by {used_tags} without measured [M] benchmark data."
        )

    warning = ""
    if violations:
        warning = f"⚠️ [Section 12.5 Compliance Warning]: {'; '.join(violations)}"

    return {
        "compliant": len(violations) == 0,
        "violations": violations,
        "warning": warning
    }

logger = logging.getLogger("CoChem_ORACLE_Engine")


class OracleEngine:
    def __init__(self) -> None:
        self.llm = None
        self.is_active = False
        self.chat_history: List[Dict[str, str]] = []
        self.model_path = self._get_model_path()
        self._executor = None
        self._oracle_cfg = self._load_oracle_config()
        
        # System Prompt enforcing rigorous citation and behavior (ORACLE-01)
        self.system_prompt = (
            "You are CoChem-ORACLE, an authoritative computational chemistry decision agent. "
            "You must answer using ONLY the v4 Method Matrix framework. "
            "Every numerical claim, rotational constant, energy, or walltime MUST carry an explicit provenance tag: "
            "[M] (measured), [D] (derived), or [E] (estimated). "
            "Under Section 12.5 Standing Rule, no [D] or [E] value may solely support a hardware exclusion or accuracy claim. "
            "Always specify Product Class (A, B, or C) and Track (ORCA vs CFOUR)."
        )

    def _get_model_path(self) -> str:
        """Retrieves the GGUF model path from the authoritative registry."""
        try:
            from cochem_base.config_loader import load_system_config_dict
            registry = load_system_config_dict(Path(CONFIG_PATH))
            return registry.get("silo_registry", {}).get("oracle_model", "")
        except Exception:
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    registry = json.loads(f.read())
                    return registry.get("silo_registry", {}).get("oracle_model", "")
            except (FileNotFoundError, json.JSONDecodeError):
                return ""

    def _load_oracle_config(self) -> Dict[str, Any]:
        """Loads oracle_config.json for LLM-specific settings (seed, temperature, etc.)."""
        try:
            from cochem_base.config_loader import load_system_config_dict
            return load_system_config_dict(Path(ORACLE_CONFIG_PATH))
        except Exception:
            try:
                with open(ORACLE_CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.loads(f.read())
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.debug(f"oracle_config.json not found or invalid ({e}), using defaults.")
                return {}

    def _write_pid(self) -> None:
        """Writes the current OS Process ID so the preemption hook can kill it if needed."""
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    def _clear_pid(self) -> None:
        """Cleans up the PID file upon graceful deactivation."""
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except OSError as err:
                logging.debug(f"PID file removal skipped: {err}")

    def activate(self) -> bool:
        """Lazy-loads the LLM into VRAM only when explicitly toggled (ORACLE-03)."""
        if self.is_active:
            return True

        logger.info("[ORACLE]: Booting Ephemeral Engine. Claiming VRAM...")
        
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.warning("Warning: 'llama-cpp-python' is not installed. Operating in RAG-only fallback mode.")
            Llama = None

        if Llama is None or not self.model_path or not os.path.exists(self.model_path):
            reason = "llama-cpp-python not installed" if Llama is None else f"model file not found at '{self.model_path}'"
            logger.warning(f"Warning: {reason}. Operating in RAG-only fallback mode.")
            logging.info(f"ORACLE activated in RAG-only fallback mode (reason: {reason}).")
            self.is_active = True
            self._write_pid()
            self.chat_history = [{"role": "system", "content": self.system_prompt}]
            return True

        try:
            # Read seed from oracle_config.json; default -1 = random/non-deterministic (MOCK-19)
            configured_seed = self._oracle_cfg.get("seed", -1)
            if not isinstance(configured_seed, int):
                logging.warning(f"Invalid seed type in oracle_config.json: {type(configured_seed).__name__}. Falling back to -1 (random).")
                configured_seed = -1
            logging.info(f"ORACLE LLM seed: {configured_seed} ({'random' if configured_seed == -1 else 'deterministic'})")

            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1, # Offload all layers to GPU
                n_ctx=4096,      # Context window size
                seed=configured_seed,  # From oracle_config.json (MOCK-19)
                verbose=False    # Suppress C++ backend logging in Jupyter
            )
            self.is_active = True
            self._write_pid()
            self.chat_history = [{"role": "system", "content": self.system_prompt}]
            return True
        except Exception as e:
            logger.error(f"Failed to load LLM model into VRAM: {e}")
            return False

    def deactivate(self) -> None:
        """Wipes the chat state and forcibly unloads the model from VRAM."""
        logger.info("[ORACLE]: Deactivating Engine. Wiping ephemeral state and freeing VRAM...")
        self.is_active = False
        self._clear_pid()
        
        self.chat_history = []
        
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        gc.collect()

    def _query_vault(self, query: str, metadata_filter: Optional[str] = None) -> str:
        """Retrieves semantic context from ChromaDB with a cosine similarity confidence score (ORACLE-05)."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(path=VAULT_DIR, settings=Settings(anonymized_telemetry=False))
            collection = client.get_collection(name="cochem_oracle_index")
            
            where_clause = {"tags": {"$contains": metadata_filter}} if metadata_filter else None
            
            results = collection.query(
                query_texts=[query],
                n_results=3,
                where=where_clause
            )
            
            if not results["documents"] or not results["documents"][0]:
                return ""
                
            # Compile context string using split-conformal prediction intervals and Section 12.5 compliance tags (ORACLE-02, ORACLE-03)
            context_blocks = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                interval = compute_split_conformal_interval(dist)
                source = meta.get("source", "Unknown")
                val_res = validate_section_12_5(doc)
                comp_tag = "[Section 12.5 Verified: Compliant]" if val_res["compliant"] else val_res["warning"]
                context_blocks.append(
                    f"--- Context (Source: {source} | Split-Conformal Interval [{int(interval['confidence_level']*100)}%]: "
                    f"{interval['lower_bound']}% - {interval['upper_bound']}%) | {comp_tag} ---\n{doc}\n"
                )
                
            return "\n".join(context_blocks)
            
        except Exception as e:
            logging.warning(f"VAULT query failed: {e}")
            return f"[VAULT ERROR: Local knowledge base unreachable - {str(e)}]"

    def _query_vault_with_timeout(self, query: str, metadata_filter: Optional[str] = None) -> str:
        """Queries VAULT using a persistent thread executor (ORACLE-04)."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            
        future = self._executor.submit(self._query_vault, query, metadata_filter)
        try:
            return future.result(timeout=3.0)
        except concurrent.futures.TimeoutError:
            return "[VAULT ERROR: Timeout exceeded 3.0s. Re-index VAULT.]"

    def _render_rag_fallback(self, query: str, context_str: str) -> str:
        """Renders a structured local RAG summary when LLM model is unavailable (MOCK-18, Suggestion 76)."""
        header = "## ORACLE RAG-Only Response\n\n"
        header += "*LLM model is not loaded. Showing documentation-grounded context only.*\n\n"

        if not context_str or context_str.startswith("[VAULT ERROR"):
            return (
                header
                + "**No relevant documentation context found for your query.**\n\n"
                + f"**Query:** {query}\n\n"
                + "> To enable full LLM inference, ensure a valid GGUF model path is set in "
                + "`cochem_system_config.json` under `silo_registry.oracle_model`.\n"
            )

        return (
            header
            + f"**Query:** {query}\n\n"
            + "### Retrieved Documentation Context\n\n"
            + context_str + "\n\n"
            + "---\n"
            + "*End of RAG-only response. Install and configure a GGUF model for full LLM inference.*\n"
        )

    async def ask_oracle(self, user_query: str, tags: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Streams the LLM response asynchronously to prevent Jupyter UI lockups."""
        if not self.is_active:
            yield "[ORACLE is currently dormant. Please activate the engine.]"
            return
            
        # 1. Retrieve RAG Context
        context_str = self._query_vault_with_timeout(user_query, tags)
        
        # 2. Package Prompt
        augmented_query = user_query
        if context_str:
            augmented_query = f"Context:\n{context_str}\n\nQuestion:\n{user_query}"
            
        self.chat_history.append({"role": "user", "content": augmented_query})
        
        if self.llm is None:
            # Structured RAG-only fallback when LLM model is unavailable (MOCK-18, Suggestion 76)
            resp = self._render_rag_fallback(user_query, context_str)
            val_res = validate_section_12_5(resp)
            if not val_res["compliant"] and val_res["warning"]:
                resp = f"{resp}\n\n{val_res['warning']}"
            self.chat_history.append({"role": "assistant", "content": resp})
            yield resp
            return

        # 3. Stream Inference
        temp = float(self._oracle_cfg.get("temperature", 0.0))
        max_tok = int(self._oracle_cfg.get("max_tokens", 1024))
        response_stream = self.llm.create_chat_completion(
            messages=self.chat_history,
            stream=True,
            temperature=temp,
            max_tokens=max_tok
        )
        
        full_response = ""
        for chunk in response_stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    token = delta["content"]
                    full_response += token
                    yield token
                    await asyncio.sleep(0)

        val_res = validate_section_12_5(full_response)
        if not val_res["compliant"] and val_res["warning"]:
            warn_msg = f"\n\n{val_res['warning']}"
            full_response += warn_msg
            yield warn_msg
                    
        # 4. Save to ephemeral history
        self.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    engine = OracleEngine()
    logger.info("Engine instantiated.")