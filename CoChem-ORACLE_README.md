# **CoChem-ORACLE: Hardware-Aware RAG & Error Interception**

## **Overview**

**CoChem-ORACLE** is the resident AI assistant for the CoChem ecosystem. It operates entirely locally, loading a heavily quantized Large Language Model (LLM) into your VRAM to provide on-the-fly troubleshooting, method recommendations, and error parsing.

ORACLE uses Retrieval-Augmented Generation (RAG) against the cochem\_knowledge\_base (including ORCA manuals, CoChem docs, and past run logs). Furthermore, it injects a global exception hook into Jupyter; if an ORCA run crashes due to a basis set linear dependence, ORACLE automatically pops up an "Ask ORACLE" button directly beneath the traceback.

## **Scientific & Technical Trade-offs**

* **The Preemption Protocol (VRAM Protection):** An LLM requires \~6-8GB of VRAM. A heavy MACE-OFF23 or GPU-accelerated PySCF calculation requires the entire GPU. **Trade-off:** ORACLE is strictly subservient to the physics engines. When a quantum chemistry stage is triggered, the CoChem memory router detects ORACLE, sends a SIGKILL, and entirely unloads the LLM from memory to guarantee the physics calculations do not crash via Out-of-Memory (OOM) errors. You will experience a 10-second delay when "waking" ORACLE back up post-calculation.  
* **Opt-In Telemetry:** Because CoChem operates in sensitive IP environments (pharmaceuticals/defense), ORACLE is 100% air-gapped by default. To share logs, users must manually run the cochem\_log\_scrubber.py to sanitize chemical structures before export.

## **Installation & Setup**

git clone \[https://github.com/CoChem/CoChem-ORACLE.git\](https://github.com/CoChem/CoChem-ORACLE.git)  
cd CoChem-ORACLE

## **How to Run**

1. **Deploy the Jupyter Widget:**  
   In your master Jupyter Notebook, import and deploy the widget:  
   import cochem\_oracle\_widget as oracle  
   oracle.deploy()

2. **Wake the Engine:**  
   Click the "Wake ORACLE" button in the UI. The PULSE monitor will display VRAM allocation.  
3. **Sync Knowledge Base:**  
   Drop PDFs or .md files into ./cochem\_knowledge\_base/ and run python cochem\_oracle\_ingest.py to update the local vector store.