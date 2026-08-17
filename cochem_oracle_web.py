import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import hashlib
from pathlib import Path

st.set_page_config(page_title="CoChem-ORACLE - Native Pipeline UI", layout="wide")

def kill_zombie_processes() -> None:
    target_procs = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info.get('name')
            if name:
                name = name.lower()
                if any(target in name for target in target_procs):
                    proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-ORACLE Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        module_dir = Path(__file__).resolve().parent
        artifact_dir = os.environ.get('COCHEM_ARTIFACT_DIR', str(Path.home() / 'cochem_artifacts'))
        
        env = os.environ.copy()
        env["COCHEM_TARGET_H5"] = os.path.join(artifact_dir, "landscape.h5")
        
        try:
            cmd = [sys.executable, str(module_dir / 'cochem_oracle_main.py')]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=3600, 
                cwd=str(module_dir),
                env=env
            )
            
            st.code(result.stdout[-3000:], language="text")
            st.success("✅ Execution Completed Natively. CPU load generated.")
        except subprocess.TimeoutExpired:
            st.error("Execution timed out. Purging zombies.")
            kill_zombie_processes()
        except subprocess.CalledProcessError as e:
            st.warning(f"Execution finished with non-zero exit code: {e.returncode}")
            kill_zombie_processes()
        except Exception as e:
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            kill_zombie_processes()
