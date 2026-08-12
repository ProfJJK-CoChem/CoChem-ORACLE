# cochem_canvas_target: cochem_oracle_config.py
"""
Configuration module for CoChem-ORACLE.
Unified under cochem_system_config.json (ORACLE-18).
"""

import json
import os
from pathlib import Path

import logging
from typing import Any

logger = logging.getLogger("CoChem_ORACLE_Config")


class OracleConfig:
    """
    Unified configuration wrapper for CoChem-ORACLE system (ORACLE-18).
    Standardized on cochem_system_config.json.
    """
    
    def __init__(self, config_file: str = None) -> None:
        """Initialize configuration."""
        if config_file is None:
            config_file = str(Path(__file__).resolve().parent / "cochem_system_config.json")
        self.config_file = config_file
        self.config = self._load_config()
        
    def _get_artifact_dir(self) -> str:
        """Get the artifact directory for reproducible research."""
        artifacts_dir = os.environ.get('COCHEM_ARTIFACT_DIR') or os.environ.get('ARTIFACTS_DIR')
        if artifacts_dir:
            return artifacts_dir
        return os.path.join(os.path.expanduser("~"), "CoChem", "artifacts")
        
    def _load_config(self) -> dict:
        """Load configuration from unified cochem_system_config.json file."""
        if not os.path.exists(self.config_file):
            root_config = Path(__file__).resolve().parents[1] / "cochem_system_config.json"
            if root_config.exists():
                self.config_file = str(root_config)
            else:
                return self._get_default_config()
        try:
            from cochem_base.config_loader import load_system_config_dict
            cfg = load_system_config_dict(self.config_file)
            default_cfg = self._get_default_config()
            default_cfg.update(cfg)
            return default_cfg
        except ImportError:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    cfg = json.loads(f.read())
                    default_cfg = self._get_default_config()
                    default_cfg.update(cfg)
                    return default_cfg
            except (FileNotFoundError, json.JSONDecodeError):
                return self._get_default_config()
            
    def _get_default_config(self) -> dict:
        """Get default configuration values."""
        artifact_dir = self._get_artifact_dir()
        return {
            "project_name": "CoChem-ORACLE",
            "version": "0.1.0",
            "data_dir": os.path.join(artifact_dir, "ORACLE", "data"),
            "knowledge_sources": {
                "primary_db": {"enabled": True, "url": "localhost:5432/cochem"},
                "external_api": {"enabled": True, "url": "https://api.example.com"}
            },
            "sync": {
                "frequency_minutes": 60,
                "max_concurrent_syncs": 4,
                "timeout_seconds": 300
            },
            "caching": {
                "enable_cache": True,
                "cache_ttl_hours": 24,
                "max_cache_size_mb": 1000
            },
            "logging": {
                "level": "INFO",
                "file": os.path.join(artifact_dir, "ORACLE", "oracle.log")
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        self._save_config()
        
    def _save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.warning(f"Warning saving config to {self.config_file}: {e}")
            
    def update_from_dict(self, updates: dict) -> None:
        """Update configuration from dictionary."""
        self.config.update(updates)
        self._save_config()

def get_oracle_config(config_file: str = None) -> OracleConfig:
    """Helper method to fetch active Oracle config."""
    return OracleConfig(config_file)

def main() -> None:
    """Main entry point for configuration module."""
    logger.info("Initializing CoChem-ORACLE Configuration")
    config = OracleConfig()
    logger.info(f"Current configuration: {config.config}")

if __name__ == "__main__":
    main()