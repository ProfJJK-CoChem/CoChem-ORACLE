# **CoChem-ORACLE: Hardware-Aware RAG & Error Interception**

**PI/Developer**: Dr. Joshua John Klaassen
**ORCiD**: [https://orcid.org/0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)
**GitHub Organization**: [https://github.com/ProfJJK-CoChem](https://github.com/ProfJJK-CoChem)

> **Important**: CoChem has recently migrated to the **Valeev Stack (MPQC, F12)**. ORACLE's knowledge base now parses MPQC tracebacks to diagnose configuration errors natively `[E]`.

Please refer to the authoritative [CoChem User Manual](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/CoChem_User_Manual.md) and [Method Matrix](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/Method_Matrix.md) for full execution instructions and basis set provenances.

## **Overview**

**CoChem-ORACLE** is the resident AI troubleshooting assistant for the CoChem ecosystem. It operates entirely locally and air-gapped, leveraging a quantized Large Language Model (LLM) for on-the-fly debugging, run analysis, and error parsing.

Features include:
- **VRAM Preemption Protocol**: ORACLE dynamically unloads its ~8GB `[M]` LLM during high-tier quantum calculations to prevent Out-Of-Memory (OOM) crashes, seamlessly reloading once compute yields.
- **RAG Architecture**: Queries local vector databases constructed from manuals and historical logs.
- **Jupyter Interception**: Global exception hooks automatically catch computational engine crashes, injecting interactive diagnostic prompts directly below the failing cell.

## **Setup and Installation**

1. Clone ORACLE:
   ```bash
   git clone https://github.com/ProfJJK-CoChem/CoChem-ORACLE.git
   cd CoChem-ORACLE
   pip install -r requirements.txt
   ```
2. Download the required quantized GGUF weights into the `models/` directory.

## **Getting Started**

1. **Populate Knowledge Vault**:
   ```bash
   python cochem_knowledge_sync.py
   ```
2. **Deploy Widget**:
   Within a Jupyter notebook cell:
   ```python
   import cochem_oracle_widget as oracle
   oracle.deploy()
   ```
3. **Sanitize Logs**:
   Clean atomic coordinates prior to external sharing:
   ```bash
   python cochem_log_scrubber.py --file error_trace.log
   ```

---
