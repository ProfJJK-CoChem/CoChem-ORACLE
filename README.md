# **CoChem-ORACLE: Hardware-Aware RAG & Error Interception**

## **Overview**

**CoChem-ORACLE** is the resident AI troubleshooting assistant for the CoChem ecosystem. Operating entirely locally, it loads a quantized Large Language Model (LLM) to perform on-the-fly debugging, run analysis, and error parsing.

ORACLE leverages Retrieval-Augmented Generation (RAG) against a local vector vault loaded with ORCA manuals, CoChem guidebooks, and past run logs. Additionally, it exposes exception hooks for Jupyter notebooks; if an ORCA run halts due to a configuration or physics error (e.g. SCF convergence issues), ORACLE intercepts the traceback and injects an interactive assistance prompt directly below the cell.

---

## **Scientific & Technical Trade-offs**

* **VRAM Preemption Protocol:** Quantized LLMs require significant GPU memory (~6-8GB VRAM). Because high-tier quantum calculations (such as MACE-OFF23 sweeps or PyTorch potential trainings) also saturate VRAM, ORACLE is dynamically preempted. When a calculation starts, the resource manager unloads ORACLE from VRAM to prevent Out-Of-Memory (OOM) crashes, reloading the LLM model once compute blocks yield.
* **Air-Gapped Privacy Controls:** Designed for secure, pre-publication environments, ORACLE runs entirely local and air-gapped. To prevent accidental leaks of molecular coordinates, users can run the `cochem_log_scrubber.py` utility to sanitize atomic structures before exporting trace logs.

---

## **File Topology & Core Scripts**

ORACLE consists of the following key Python scripts:

1. **[cochem_oracle_main.py](file:///d:/GitHub-Repo/CoChem-ORACLE/cochem_oracle_main.py)** (Central System Orchestrator):
   * Coordinates the database directories initialization and logs compilation.

2. **[cochem_oracle_config.py](file:///d:/GitHub-Repo/CoChem-ORACLE/cochem_oracle_config.py)** (Configuration manager):
   * Tracks model paths, prompt templates, and active local vector databases.

3. **[cochem_oracle_engine.py](file:///d:/GitHub-Repo/CoChem-ORACLE/cochem_oracle_engine.py)** (Local LLM loader):
   * Manages local model loading and querying interfaces (wrapping `llama.cpp` subprocess routes).

4. **[cochem_knowledge_sync.py](file:///d:/GitHub-Repo/CoChem-ORACLE/cochem_knowledge_sync.py)** (Semantic RAG Archivist):
   * Dynamically splits markdown and text manuals using header-based boundary bounds (preventing code matrices from being truncated).
   * Generates and upserts document vector embeddings into local ChromaDB persistent databases.

5. **[cochem_oracle_widget.py](file:///d:/GitHub-Repo/CoChem-ORACLE/cochem_oracle_widget.py)** & **[cochem_oracle_hook.py](file:///d:/GitHub-Repo/CoChem-ORACLE/cochem_oracle_hook.py)**:
   * Build the Jupyter widgets interface and register the global traceback exception listeners.

6. **[cochem_log_scrubber.py](file:///d:/GitHub-Repo/CoChem-ORACLE/cochem_log_scrubber.py)**:
   * Cleans logs of private coordinates prior to external packaging.

---

## **Workflow & How to Run**

To synchronize documents and launch the widgets:

1. **Populate the Knowledge Vault**:
   Place your Markdown reference files inside `~/CoChem/cochem_knowledge_base/` and run the archivist:
   ```bash
   python cochem_knowledge_sync.py
   ```

2. **Deploy the Interactive Notebook Widget**:
   In your active Jupyter notebook cell, import and draw the widget panel:
   ```python
   import cochem_oracle_widget as oracle
   oracle.deploy()
   ```

3. **Sanitize Debugging Trace Logs**:
   To share crash logs securely, clean private coordinates via the scrubber:
   ```bash
   python cochem_log_scrubber.py --file error_trace.log
   ```