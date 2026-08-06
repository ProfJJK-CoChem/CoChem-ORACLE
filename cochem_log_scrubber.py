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

class TelemetrySanitizer:
    def __init__(self):
        # 1. Matches standard XYZ coordinate lines: C  -1.23456  0.00000  2.34567
        # Assumes at least 3 decimal places to distinguish from general version numbers.
        self.xyz_pattern = re.compile(
            r'([A-Za-z]{1,2})\s+(-?\d+\.\d{3,})\s+(-?\d+\.\d{3,})\s+(-?\d+\.\d{3,})'
        )
        
        # 2. Matches ORCA coordinate blocks: * xyz 0 1 ... * self.orca_block_pattern = re.compile(
            r'(\*\s*xyz.*?\n)(.*?)(\*)', re.DOTALL | re.IGNORECASE
        )
        
        # 3. Matches dense SMILES-like strings (Heuristic: >8 chars, contains chemical brackets/symbols)
        self.smiles_pattern = re.compile(
            r'\b([B,C,N,O,P,S,F,Cl,Br,I,c,n,o,s,p]+[\(\)\[\]\=\#]+[A-Za-z0-9\(\)\[\]\=\#\+\-\.\@\:\\]{4,})\b'
        )
        
        # 4. Matches continuous block matrices of just floats (e.g., Hessian/Polarizability matrices)
        self.float_matrix_pattern = re.compile(
            r'(\s*-?\d+\.\d{4,}\s+){4,}'
        )

    def sanitize_text(self, text: str) -> str:
        """Passes text through the Regex wall to strip proprietary data."""
        if not text:
            return text
            
        # Strip ORCA Blocks
        clean_text = self.orca_block_pattern.sub(r'\1[REDACTED_GEOMETRY_BLOCK]\n\3', text)
        
        # Strip individual XYZ lines
        clean_text = self.xyz_pattern.sub(r'\1 [REDACTED_COORD] [REDACTED_COORD] [REDACTED_COORD]', clean_text)
        
        # Strip dense matrix dumps
        clean_text = self.float_matrix_pattern.sub(r'\n[REDACTED_FLOAT_MATRIX]\n', clean_text)
        
        # Strip SMILES-like strings
        clean_text = self.smiles_pattern.sub(r'[REDACTED_SMILES_STRING]', clean_text)
        
        return clean_text

    def scrub_chat_log(self, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Iterates over the ephemeral chat array and sanitizes all content."""
        scrubbed_history = []
        for msg in chat_history:
            scrubbed_msg = {
                "role": msg.get("role", "unknown"),
                "content": self.sanitize_text(msg.get("content", ""))
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
    
    timestamp = datetime.datetime.now().strftime("%Y%md_%H%M%S")
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
        # Write the human-readable JSON Manifest first (for user review)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, indent=4)
        print_status(f"Manifest generated: {manifest_path}", "success")
        
        # Package into a Zip file for sharing
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(manifest_path, arcname=manifest_name)
        print_status(f"Telemetry packaged successfully: {zip_path}", "success")
        
        # Optionally clean up the raw json after zipping, but leaving it allows user inspection
        return zip_path
        
    except Exception as e:
        print_status(f"Telemetry export failed: {e}", "fail")
        return ""

def main():
    print(f"\n{Colors.BOLD}--- CoChem-ORACLE: SHIELD Telemetry Test ---{Colors.ENDC}")
    
    # Mock Data for testing the Regex Wall
    mock_chat = [
        {"role": "user", "content": "Why did my ORCA job fail? Here is my input:\n* xyz 0 1\nC -1.23456 0.12345 1.11111\nO 0.00000 1.23456 -1.22222\n*\nIt says SCF failed."},
        {"role": "oracle", "content": "It seems your geometry is poorly optimized. Try increasing SCF convergence."},
        {"role": "user", "content": "Also this smiles C1=CC=C(C=C1)O is failing."}
    ]
    
    export_telemetry(mock_chat)

if __name__ == "__main__":
    main()