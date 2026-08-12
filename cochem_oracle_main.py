# cochem_canvas_target: cochem_oracle_main.py
"""
Main orchestrator module for CoChem-ORACLE.
This is the central entry point for the ORACLE (Knowledge Management) system.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from cochem_knowledge_sync import sync_knowledge_base
    from cochem_oracle_config import OracleConfig
except ImportError:
    from CoChem_ORACLE.cochem_knowledge_sync import sync_knowledge_base
    from CoChem_ORACLE.cochem_oracle_config import OracleConfig

import logging

logger = logging.getLogger("CoChem_ORACLE_Main")


class OracleOrchestrator:
    """
    The main orchestrator that coordinates all ORACLE activities (ORACLE-19).
    """
    
    def __init__(self, config_file: str = None) -> None:
        """Initialize the ORACLE orchestrator."""
        self.oracle_config = OracleConfig(config_file)
        self.config_file = self.oracle_config.config_file
        self.config = self.oracle_config.config
        self.is_initialized = False
        
    def _get_artifact_dir(self) -> str:
        """Get the artifact directory for reproducible research."""
        return self.oracle_config._get_artifact_dir()
        
    def initialize(self) -> None:
        """Initialize the ORACLE system."""
        logger.info("Initializing CoChem-ORACLE System...")
        
        artifact_dir = self._get_artifact_dir()
        data_dir = Path(self.config.get('data_dir', os.path.join(artifact_dir, "ORACLE", "data")))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        (data_dir / "knowledge_base").mkdir(parents=True, exist_ok=True)
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)
        (data_dir / "reports").mkdir(parents=True, exist_ok=True)
        (data_dir / "cache").mkdir(parents=True, exist_ok=True)
        
        self.is_initialized = True
        logger.info("CoChem-ORACLE initialized successfully")
        
    def run_knowledge_sync(self, target: str = "default") -> bool:
        """Run knowledge synchronization with live sync_knowledge_base engine (ORACLE-19)."""
        if not self.is_initialized:
            raise RuntimeError("ORACLE system must be initialized before running sync")
            
        logger.info(f"Synchronizing knowledge base (Target: {target})...")
        try:
            sync_knowledge_base()
            logger.info(f"Knowledge sync ({target}) completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Knowledge sync failed: {e}")
            return False
        
    def generate_knowledge_report(self, output_dir: str = "./reports") -> str:
        """Generate comprehensive report of knowledge operations (ORACLE-19)."""
        if not self.is_initialized:
            raise RuntimeError("ORACLE system must be initialized before generating reports")
            
        logger.info(f"Generating ORACLE knowledge report in {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        report_path = Path(output_dir) / "cochem_oracle_report.json"
        
        report_data = {
            "project": "CoChem-ORACLE",
            "status": "OPERATIONAL",
            "config_file": self.config_file,
            "data_dir": str(self.config.get("data_dir", ""))
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        logger.info(f"ORACLE knowledge report generated at {report_path}")
        return str(report_path)

def main() -> None:
    """Main entry point for CoChem-ORACLE."""
    logger.info("Starting CoChem-ORACLE Orchestrator")
    orchestrator = OracleOrchestrator()
    orchestrator.initialize()
    
    orchestrator.run_knowledge_sync("local_vault")
    orchestrator.generate_knowledge_report("./reports")
    
if __name__ == "__main__":
    main()