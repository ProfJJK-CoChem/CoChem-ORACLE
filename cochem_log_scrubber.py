#!/usr/bin/env python3
"""
CoChem-ORACLE: Telemetry Manifest & Sanitizer (SHIELD)
Acts as a Regex Wall to aggressively strip proprietary molecular geometries, 
SMILES strings, and coordinate arrays from local LLM chat logs before packaging 
them into a shareable ZIP archive for optional telemetry feedback.
"""

import os
import re
import json
import zipfile
import datetime
from typing import List, Dict

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

import logging

logger = logging.getLogger("CoChem_Log_Scrubber")


def print_status(msg: str, status: str = "info") -> None:
    if status == "success":
        logger.info(f"✅ {msg}")
    elif status == "warning":
        logger.warning(f"⚠️ {msg}")
    elif status == "fail":
        logger.error(f"❌ {msg}")
    else:
        logger.info(f"➡️ {msg}")

class TelemetrySanitizer:
    def __init__(self) -> None:
        # 1. Matches standard XYZ coordinate lines: C  -1.23456  0.00000  2.34567
        self.xyz_pattern = re.compile(
            r'([A-Za-z]{1,2})\s+(-?\d+\.\d{3,})\s+(-?\d+\.\d{3,})\s+(-?\d+\.\d{3,})'
        )
        
        # 2. Matches ORCA coordinate blocks (Fixed syntax bug: un-commented definition) (ORACLE-13)
        self.orca_block_pattern = re.compile(
            r'(\*\s*xyz.*?\n)(.*?)(\*)', re.DOTALL | re.IGNORECASE
        )
        
        # 3. SMILES pattern matching candidate chemical sequences without literal commas or rigid {3,} (ORACLE-14)
        # Refined regex using negative lookbehind/lookahead (?<![A-Za-z0-9_]) ... (?![A-Za-z0-9_])
        # to prevent truncation of bracketed, charged, isotopic, or disconnected SMILES strings.
        self.smiles_pattern = re.compile(
            r'(?<![A-Za-z0-9_])((?:Cl|Br|Si|Se|Te|Na|K|Fe|Mg|Ca|Li|Cu|Zn|Al|Pt|Pd|[BCNOPSFIHbcnsoph0-9\(\)\[\]\=\#\@\+\-\\\/\.:%]){2,})(?![A-Za-z0-9_])'
        )
        
        # 4. Matches continuous block matrices of floats
        self.float_matrix_pattern = re.compile(
            r'(\s*-?\d+\.\d{4,}\s+){4,}'
        )

    def _is_valid_smiles(self, candidate: str) -> bool:
        """Validates SMILES candidate via RDKit if available (ORACLE-14)."""
        if not candidate:
            return False
        if not RDKIT_AVAILABLE:
            # Fallback heuristic: check for explicit bond/bracket/ring/aromatic characters
            return any(c in candidate for c in ['=', '#', '[', ']', '@', '\\', '/', '(', ')']) or any(c.isdigit() for c in candidate)
        try:
            mol = Chem.MolFromSmiles(candidate)
            return mol is not None
        except Exception:
            return False

    def sanitize_text(self, text: str) -> str:
        """Passes text through the Regex wall to strip proprietary data."""
        if not text:
            return text
            
        # Strip ORCA Blocks (ORACLE-13)
        clean_text = self.orca_block_pattern.sub(r'\1[REDACTED_GEOMETRY_BLOCK]\n\3', text)
        
        # Strip individual XYZ lines
        clean_text = self.xyz_pattern.sub(r'\1 [REDACTED_COORD] [REDACTED_COORD] [REDACTED_COORD]', clean_text)
        
        # Strip dense matrix dumps
        clean_text = self.float_matrix_pattern.sub(r'\n[REDACTED_FLOAT_MATRIX]\n', clean_text)
        
        # Strip SMILES strings with RDKit syntax validation (ORACLE-14)
        def replace_smiles(match: Any) -> str:
            candidate = match.group(1)
            if self._is_valid_smiles(candidate):
                return '[REDACTED_SMILES_STRING]'
            return match.group(0)

        clean_text = self.smiles_pattern.sub(replace_smiles, clean_text)
        
        return clean_text

    def scrub_chat_log(self, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Iterates over the ephemeral chat array and sanitizes all content."""
        if not isinstance(chat_history, list):
            return []
        scrubbed_history = []
        for msg in chat_history:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "unknown")) if msg.get("role") is not None else "unknown"
            content = msg.get("content", "")
            content_str = str(content) if content is not None else ""
            scrubbed_msg = {
                "role": role,
                "content": self.sanitize_text(content_str)
            }
            scrubbed_history.append(scrubbed_msg)
        return scrubbed_history

def export_telemetry(chat_history: List[Dict[str, str]], export_dir: str = None) -> str:
    """
    Sanitizes the chat, generates the JSON manifest, and zips it.
    Returns the file path to the final .zip archive.
    """
    if export_dir is None:
        export_dir = os.path.join(os.path.expanduser("~"), "CoChem", "cochem_exports")
        
    os.makedirs(export_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_name = f"cochem_oracle_telemetry_{timestamp}.json"
    zip_name = f"cochem_oracle_telemetry_{timestamp}.zip"
    
    manifest_path = os.path.join(export_dir, manifest_name)
    zip_path = os.path.join(export_dir, zip_name)
    
    sanitizer = TelemetrySanitizer()
    print_status("SHIELD Active: Scrubbing proprietary geometries and matrices...", "info")
    clean_history = sanitizer.scrub_chat_log(chat_history)
    
    manifest_payload = {
        "export_timestamp": timestamp,
        "cochem_version": "2026.1",
        "privacy_guarantee": "All identified coordinates, matrices, and SMILES have been stripped.",
        "chat_data": clean_history
    }
    
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, indent=4)
        print_status(f"Manifest generated: {manifest_path}", "success")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(manifest_path, arcname=manifest_name)
        print_status(f"Telemetry packaged successfully: {zip_path}", "success")
        
        return zip_path
        
    except Exception as e:
        print_status(f"Telemetry export failed: {e}", "fail")
        return ""

def main() -> None:
    logger.info("--- CoChem-ORACLE: SHIELD Telemetry Test ---")
    
    mock_chat = [
        {"role": "user", "content": "Why did my ORCA job fail? Here is my input:\n* xyz 0 1\nC -1.23456 0.12345 1.11111\nO 0.00000 1.23456 -1.22222\n*\nIt says SCF failed."},
        {"role": "oracle", "content": "It seems your geometry is poorly optimized. Try increasing SCF convergence."},
        {"role": "user", "content": "Also this smiles C1=CC=C(C=C1)O is failing."}
    ]
    
    export_telemetry(mock_chat)

if __name__ == "__main__":
    main()