#!/usr/bin/env python3
"""
CoChem-ORACLE: Ephemeral Engine
Manages lazy-loaded local LLM inference, ephemeral chat state wiping, 
and strict Semantic RAG contextualization via ChromaDB.
"""

import os
import gc
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

class OracleEngine:
    def __init__(self):
        self.llm = None
        self.is_active = False
        self.chat_history: List[Dict[str, str]] = []
        self.model_path = self._get_model_path()
        self._executor = None
        self._oracle_cfg = self._load_oracle_config()
        
        # System Prompt enforcing rigorous citation and behavior
        self.system_prompt = (
            "You are CoChem-ORACLE, a strict computational chemistry assistant. "
            "You must answer using ONLY the provided local NotebookLM context. "
            "If the context does not contain the answer, say 'I cannot answer this based on the local knowledge base.' "
            "Always append [Source: <filename>] to your response based on the context metadata."
        )

    def _get_model_path(self) -> str:
        """Retrieves the GGUF model path from the authoritative registry."""
        try:
            with open(CONFIG_PATH, "r") as f:
                registry = json.load(f)
                return registry.get("silo_registry", {}).get("oracle_model", "")
        except (FileNotFoundError, json.JSONDecodeError):
            return ""

    def _load_oracle_config(self) -> Dict[str, Any]:
        """Loads oracle_config.json for LLM-specific settings (seed, temperature, etc.)."""
        try:
            with open(ORACLE_CONFIG_PATH, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.debug(f"oracle_config.json not found or invalid ({e}), using defaults.")
            return {}

    def _write_pid(self):
        """Writes the current OS Process ID so the preemption hook can kill it if needed."""
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _clear_pid(self):
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

        print(" [ORACLE]: Booting Ephemeral Engine. Claiming VRAM...")
        
        try:
            from llama_cpp import Llama
        except ImportError:
            print("⚠️ Warning: 'llama-cpp-python' is not installed. Operating in RAG-only fallback mode.")
            Llama = None

        if Llama is None or not self.model_path or not os.path.exists(self.model_path):
            reason = "llama-cpp-python not installed" if Llama is None else f"model file not found at '{self.model_path}'"
            print(f"⚠️ Warning: {reason}. Operating in RAG-only fallback mode.")
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
            print(f"❌ Failed to load LLM model into VRAM: {e}")
            return False

    def deactivate(self) -> None:
        """Wipes the chat state and forcibly unloads the model from VRAM."""
        print(" [ORACLE]: Deactivating Engine. Wiping ephemeral state and freeing VRAM...")
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
                
            # Compile context string using cosine similarity confidence transform (ORACLE-05)
            context_blocks = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                confidence = round(100.0 * (1.0 / (1.0 + float(dist))), 1) # Cosine similarity transform (ORACLE-05)
                source = meta.get("source", "Unknown")
                context_blocks.append(f"--- Context (Source: {source} | Confidence: {confidence}%) ---\n{doc}\n")
                
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
            self.chat_history.append({"role": "assistant", "content": resp})
            yield resp
            return

        # 3. Stream Inference
        response_stream = self.llm.create_chat_completion(
            messages=self.chat_history,
            stream=True,
            temperature=0.1,
            max_tokens=1024
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
                    
        # 4. Save to ephemeral history
        self.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    engine = OracleEngine()
    print("Engine instantiated.")