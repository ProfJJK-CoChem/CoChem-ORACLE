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
import chromadb
from chromadb.config import Settings

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status: str = "info") -> None:
    if status == "success":
        print(f" {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
    elif status == "warning":
        print(f" {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")
    elif status == "fail":
        print(f" {Colors.FAIL}❌ {msg}{Colors.ENDC}")
    else:
        print(f" {Colors.OKCYAN}➡️ {msg}{Colors.ENDC}")

# Core Paths
HOME_DIR = os.path.expanduser("~")
KNOWLEDGE_DIR = os.path.join(HOME_DIR, "CoChem", "cochem_knowledge_base")
VAULT_DIR = os.path.join(HOME_DIR, "CoChem", "cochem_vault")

def ensure_directories():
    """Ensures the knowledge drop-zone and vault directories exist."""
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    os.makedirs(VAULT_DIR, exist_ok=True)
    print_status(f"Knowledge drop-zone verified at: {KNOWLEDGE_DIR}")

def semantic_chunker(text: str, source_name: str) -> list:
    """
    CORTEX Module: Splits text dynamically at Markdown Headers (H1-H4).
    This guarantees that dense configuration blocks (like ORCA %elprop ... end)
    are never sliced in half arbitrarily by character counts.
    """
    # Lookahead regex: split right before a newline that starts with 1-4 hashes and a space
    raw_chunks = re.split(r'\n(?=#{1,4}\s)', '\n' + text)
    
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if len(chunk) < 50:  # Ignore trivial/empty splits
            continue
            
        # Extract NotebookLM tags (e.g. #troubleshooting, #mace) 
        # Negative lookbehinds ensure we don't grab Python/ORCA comments or Markdown headers
        raw_tags = re.findall(r'(?<!^)(?<!\S)#([A-Za-z0-9_]+)', chunk, re.MULTILINE)
        unique_tags = list(set(raw_tags))
        
        # Create an immutable hash ID for this specific chunk text to prevent duplicate upserts
        chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
        
        metadata = {
            "source": source_name,
            "tags": ",".join(unique_tags) if unique_tags else "none"
        }
        
        chunks.append({
            "id": f"{source_name}_{chunk_hash[:8]}",
            "text": chunk,
            "metadata": metadata
        })
        
    return chunks

def sync_knowledge_base():
    """VAULT Module: Processes all files and upserts to ChromaDB."""
    ensure_directories()
    
    md_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))
    txt_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))
    all_files = md_files + txt_files
    
    if not all_files:
        print_status("No Markdown or Text files found in the knowledge base. Skipping sync.", "warning")
        return

    print_status(f"Initializing Local VAULT (ChromaDB) at {VAULT_DIR}...")
    try:
        # Initialize persistent client. Automatically uses sentence-transformers under the hood.
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

def main():
    print(f"\n{Colors.BOLD}--- CoChem-ORACLE: Knowledge Sync ---{Colors.ENDC}")
    sync_knowledge_base()

if __name__ == "__main__":
    main()