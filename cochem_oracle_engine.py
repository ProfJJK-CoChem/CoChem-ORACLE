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
import concurrent.futures
from typing import List, Dict, Optional, AsyncGenerator

# Core Paths
HOME_DIR = os.path.expanduser("~")
VAULT_DIR = os.path.join(HOME_DIR, "CoChem", "cochem_vault")
PID_FILE = os.path.join(HOME_DIR, ".cochem", "silos", "oracle", "oracle_engine.pid")
CONFIG_PATH = "cochem_setup/cochem_system_config.json"

class OracleEngine:
    def __init__(self):
        self.llm = None
        self.is_active = False
        self.chat_history: List[Dict[str, str]] = []
        self.model_path = self._get_model_path()
        
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
        except FileNotFoundError:
            return ""

    def _write_pid(self):
        """Writes the current OS Process ID so the preemption hook can kill it if needed."""
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _clear_pid(self):
        """Cleans up the PID file upon graceful deactivation."""
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

    def activate(self) -> bool:
        """Lazy-loads the LLM into VRAM only when explicitly toggled."""
        if self.is_active:
            return True
            
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError("GGUF model not found in registry.")

        print(" [ORACLE]: Booting Ephemeral Engine. Claiming VRAM...")
        
        # Lazy import to prevent PyTorch/CUDA context initialization when dormant
        from llama_cpp import Llama
        
        # Initialize with locked seed for determinism and offload to GPU
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1, # Offload all layers to GPU
            n_ctx=4096,      # Context window size
            seed=42,         # Lock generation determinism
            verbose=False    # Suppress C++ backend logging in Jupyter
        )
        
        self.is_active = True
        self._write_pid()
        
        # Initialize Ephemeral State
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        return True

    def deactivate(self) -> None:
        """Wipes the chat state and forcibly unloads the model from VRAM."""
        print(" [ORACLE]: Deactivating Engine. Wiping ephemeral state and freeing VRAM...")
        self.is_active = False
        self._clear_pid()
        
        # Ephemeral Chat State Wipe
        self.chat_history = []
        
        # Force garbage collection of the C++ pointers
        if self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()

    def _query_vault(self, query: str, metadata_filter: Optional[str] = None) -> str:
        """Retrieves semantic context from ChromaDB with a strict 3-second timeout fallback."""
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
                
            # Compile context string
            context_blocks = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                confidence = max(0, 100 - int(dist * 100)) # Simple distance-to-confidence heuristic
                source = meta.get("source", "Unknown")
                context_blocks.append(f"--- Context (Source: {source} | Confidence: {confidence}%) ---\n{doc}\n")
                
            return "\n".join(context_blocks)
            
        except Exception as e:
            return f"[VAULT ERROR: Local knowledge base unreachable - {str(e)}]"

    def _query_vault_with_timeout(self, query: str, metadata_filter: Optional[str] = None) -> str:
        """Wraps the VAULT query in a 3-second thread timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._query_vault, query, metadata_filter)
            try:
                return future.result(timeout=3.0)
            except concurrent.futures.TimeoutError:
                return "[VAULT ERROR: Timeout exceeded 3.0s. Re-index VAULT.]"

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
        
        # 3. Stream Inference (Temperature 0.1 for strict scientific adherence)
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
                    await asyncio.sleep(0) # Yield control back to Jupyter event loop
                    
        # 4. Save to ephemeral history
        self.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    # Isolated unit test block
    engine = OracleEngine()
    print("Engine instantiated.")