#!/usr/bin/env python3
"""
CoChem-ORACLE: Semantic RAG Sync (CORTEX/VAULT)
Ingests NotebookLM Markdown exports, applies semantic header-based chunking
to preserve rigid physical input blocks (e.g., ORCA matrices), and upserts
the embeddings into the localized ChromaDB vector vault.
"""

import os
import glob
import re
import hashlib
from pathlib import Path
from typing import Any, List, Dict
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

import logging

logger = logging.getLogger("CoChem_Knowledge_Sync")


def print_status(msg: str, status: str = "info") -> None:
    if status == "success":
        logger.info(f"✅ {msg}")
    elif status == "warning":
        logger.warning(f"⚠️ {msg}")
    elif status == "fail":
        logger.error(f"❌ {msg}")
    else:
        logger.info(f"➡️ {msg}")

# Core Paths configured dynamically (ORACLE-10)
HOME_DIR = os.path.expanduser("~")

def get_knowledge_dir() -> str:
    env_dir = os.environ.get("COCHEM_KNOWLEDGE_DIR") or os.environ.get("COCHEM_ARTIFACT_DIR") or os.environ.get("ARTIFACTS_DIR")
    if env_dir:
        return str(Path(env_dir) / "cochem_knowledge_base")
    return os.path.join(HOME_DIR, "CoChem", "cochem_knowledge_base")

def get_vault_dir() -> str:
    env_dir = os.environ.get("COCHEM_ARTIFACT_DIR") or os.environ.get("ARTIFACTS_DIR")
    if env_dir:
        return str(Path(env_dir) / "cochem_vault")
    return os.path.join(HOME_DIR, "CoChem", "cochem_vault")

KNOWLEDGE_DIR = get_knowledge_dir()
VAULT_DIR = get_vault_dir()

def ensure_directories() -> None:
    """Ensures the knowledge drop-zone and vault directories exist."""
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    os.makedirs(VAULT_DIR, exist_ok=True)
    print_status(f"Knowledge drop-zone verified at: {KNOWLEDGE_DIR}")

def semantic_chunker(text: str, source_name: str) -> List[Dict[str, Any]]:
    """
    CORTEX Module: Splits text dynamically at Markdown Headers (H1-H4).
    Protects code fences (```...```) so hash comments (#) inside code blocks are not treated as headers (ORACLE-11).
    Uses SHA-256 for collision-resistant chunk hashes (ORACLE-12).
    """
    # 1. Protect code blocks by replacing headers inside code blocks with placeholder tokens (ORACLE-11)
    code_block_pattern = re.compile(r'```.*?```', re.DOTALL)
    code_blocks = []
    
    def replacer(match: Any) -> str:
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    masked_text = code_block_pattern.sub(replacer, text)
    
    # 2. Split on markdown headers H1-H4
    raw_chunks = re.split(r'\n(?=#{1,4}\s)', '\n' + masked_text)
    
    chunks = []
    for chunk in raw_chunks:
        # Restore code blocks (ORACLE-11)
        for i, code_content in enumerate(code_blocks):
            chunk = chunk.replace(f"__CODE_BLOCK_{i}__", code_content)

        chunk = chunk.strip()
        if len(chunk) < 50:  # Ignore trivial/empty splits
            continue
            
        # Extract tags
        raw_tags = re.findall(r'(?<!^)(?<!\S)#([A-Za-z0-9_]+)', chunk, re.MULTILINE)
        unique_tags = list(set(raw_tags))
        
        # Create SHA-256 hash truncated to 16 chars (ORACLE-12)
        chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()[:16]
        
        metadata = {
            "source": source_name,
            "tags": ",".join(unique_tags) if unique_tags else "none"
        }
        
        chunks.append({
            "id": f"{source_name}_{chunk_hash}",
            "text": chunk,
            "metadata": metadata
        })
        
    return chunks

def sync_knowledge_base() -> None:
    """VAULT Module: Processes all files and upserts to ChromaDB."""
    ensure_directories()
    
    md_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))
    txt_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))
    all_files = md_files + txt_files
    
    if not all_files:
        print_status("No Markdown or Text files found in the knowledge base. Skipping sync.", "warning")
        return

    print_status(f"Initializing Local VAULT (ChromaDB) at {VAULT_DIR}...")
    if not CHROMADB_AVAILABLE:
        print_status("chromadb library is not installed. VAULT functionality disabled.", "warning")
        return
    try:
        client = chromadb.PersistentClient(path=VAULT_DIR, settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(name="cochem_oracle_index")
    except Exception as e:
        print_status(f"Failed to initialize VAULT: {e}", "fail")
        return

    total_chunks = 0
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print_status(f"Parsing {filename}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        chunks = semantic_chunker(content, filename)
        if not chunks:
            continue
            
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        try:
            collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            total_chunks += len(chunks)
            print_status(f"  -> Upserted {len(chunks)} semantic chunks.", "success")
        except Exception as e:
            print_status(f"  -> Failed to upsert {filename}: {e}", "fail")

    print_status(f"Sync Complete. VAULT now holds {total_chunks} updated chunks ready for RAG.", "success")

def main() -> None:
    logger.info("--- CoChem-ORACLE: Knowledge Sync ---")
    sync_knowledge_base()

if __name__ == "__main__":
    main()