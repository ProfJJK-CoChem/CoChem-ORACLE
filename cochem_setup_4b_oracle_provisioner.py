#!/usr/bin/env python3
"""
CoChem Setup Phase 4b: ORACLE Silo Provisioning
Creates an isolated Python micro-silo for local LLM inference and RAG capabilities.
Downloads the strict Q4_K_M GGUF model and updates the central CoChem registry.
"""

import os
import sys
import subprocess
import json
import urllib.request
import time

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status_type: str = "info") -> None:
    if status_type == "info":
        print(f"{Colors.OKCYAN}[INFO]{Colors.ENDC} {msg}")
    elif status_type == "success":
        print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}")
    elif status_type == "warning":
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {msg}")
    elif status_type == "fail":
        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {msg}")

# Core Paths
HOME_DIR = os.path.expanduser("~")
SILO_DIR = os.path.join(HOME_DIR, ".cochem", "silos", "oracle")
MODEL_DIR = os.path.join(HOME_DIR, ".cochem", "models")
CONFIG_PATH = os.path.join("cochem_setup", "cochem_system_config.json")

# Model Configuration (4-bit Quantized to guarantee < 6GB VRAM footprint)
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def provision_silo() -> str:
    """Creates the isolated virtual environment for ORACLE."""
    print_status(f"Provisioning ORACLE micro-silo at {SILO_DIR}...", "info")
    if not os.path.exists(SILO_DIR):
        try:
            subprocess.run([sys.executable, "-m", "venv", SILO_DIR], check=True)
            print_status("ORACLE micro-silo created.", "success")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to create ORACLE silo: {e}", "fail")
            sys.exit(1)
    else:
        print_status("ORACLE micro-silo already exists. Proceeding...", "success")
    
    # Return the path to the silo's python executable
    return os.path.join(SILO_DIR, "bin", "python")

def install_dependencies(silo_python: str) -> None:
    """Installs llama-cpp-python, chromadb, and supporting RAG libraries."""
    print_status("Installing ORACLE dependencies into micro-silo...", "info")
    
    # We install llama-cpp-python with basic CPU support first for safety, 
    # but in a full CUDA env we would pass CMAKE_ARGS="-DGGML_CUDA=on"
    deps = [
        "llama-cpp-python", 
        "chromadb", 
        "sentence-transformers", 
        "ipywidgets", 
        "psutil"
    ]
    
    try:
        subprocess.run([silo_python, "-m", "pip", "install", "--upgrade", "pip"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([silo_python, "-m", "pip", "install"] + deps, check=True)
        print_status("ORACLE dependencies successfully installed.", "success")
    except subprocess.CalledProcessError as e:
        print_status(f"Dependency installation failed: {e}", "fail")
        sys.exit(1)

def reporthook(count, block_size, total_size):
    """Callback function to display download progress."""
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    if duration == 0:
        duration = 1e-6
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\r{Colors.OKCYAN}[DOWNLOADING]{Colors.ENDC} {percent}% - {progress_size / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB | {speed} KB/s")
    sys.stdout.flush()

def download_model() -> None:
    """Downloads the GGUF model if it does not already exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print_status(f"Model already exists at {MODEL_PATH}. Skipping download.", "success")
        return
    
    print_status(f"Downloading Q4_K_M GGUF model (~4.1 GB). This may take a while...", "info")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook)
        print() # New line after progress bar
        print_status("Model download complete.", "success")
    except Exception as e:
        print()
        print_status(f"Failed to download model: {e}", "fail")
        sys.exit(1)

def update_registry() -> None:
    """Injects the ORACLE silo and model paths into the CoChem config."""
    print_status("Updating cochem_system_config.json registry...", "info")
    if not os.path.exists(CONFIG_PATH):
        print_status("Registry not found. Run previous setup phases first.", "warning")
        return

    with open(CONFIG_PATH, "r") as f:
        registry = json.load(f)

    if "silo_registry" not in registry:
        registry["silo_registry"] = {}
    
    registry["silo_registry"]["oracle_python"] = os.path.join(SILO_DIR, "bin", "python")
    registry["silo_registry"]["oracle_model"] = MODEL_PATH
    registry["silo_registry"]["oracle_vram_limit_gb"] = 6.0

    with open(CONFIG_PATH, "w") as f:
        json.dump(registry, f, indent=4)
    
    print_status("Registry successfully patched with ORACLE parameters.", "success")

def main():
    print(f"\n{Colors.BOLD}--- Phase 4b: ORACLE Silo Provisioning ---{Colors.ENDC}\n")
    silo_python = provision_silo()
    install_dependencies(silo_python)
    download_model()
    update_registry()
    print(f"\n{Colors.BOLD}--- ORACLE Phase 4b Complete ---{Colors.ENDC}\n")

if __name__ == "__main__":
    main()