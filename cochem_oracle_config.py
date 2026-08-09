# cochem_canvas_target: cochem_oracle_config.py
"""
Configuration module for CoChem-ORACLE.
Handles all configuration settings for the ORACLE system.
"""

import json
import os
from pathlib import Path

class OracleConfig:
    """
    Configuration class for CoChem-ORACLE system.
    """
    
    def __init__(self, config_file: str = "cochem_oracle_config.json"):
        """Initialize configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        
    def _get_artifact_dir(self) -> str:
        """Get the artifact directory for reproducible research.
        
        Returns:
            str: The path to the artifact directory, using environment variable 
                 or defaulting to home directory.
        """
        # Check if ARTIFACTS_DIR environment variable is set
        artifacts_dir = os.environ.get('ARTIFACTS_DIR')
        if artifacts_dir:
            return artifacts_dir
        
        # Fallback to home directory
        return os.path.join(os.path.expanduser("~"), "CoChem", "artifacts")
        
    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Error loading config: {e}")
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
        
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        return self.config.get(key, default)
        
    def set(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value
        self._save_config()
        
    def _save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def update_from_dict(self, updates: dict):
        """Update configuration from dictionary."""
        self.config.update(updates)
        self._save_config()

def main():
    """Main entry point for configuration module."""
    print("Initializing CoChem-ORACLE Configuration")
    
    config = OracleConfig()
    print("Current configuration:", config.config)

if __name__ == "__main__":
    main()