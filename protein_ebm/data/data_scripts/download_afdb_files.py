#!/usr/bin/env python3
"""
Script to download AlphaFold Database PDB files from a list of IDs.

Usage:
    python download_afdb_files.py

The script reads IDs from /home/gridsan/jroney/solab/ProteinEBM/data/proteina/d_FS_index.txt
and downloads the corresponding PDB files from the AlphaFold Database.
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
import time
from tqdm import tqdm

# Configuration
INPUT_FILE = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/d_FS_index.txt"
OUTPUT_DIR = "/home/gridsan/jroney/solab/ProteinEBM/data/proteina/afdb_pdbs"
BASE_URL = "https://alphafold.ebi.ac.uk/files"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def download_pdb_file(afdb_id, output_dir, max_retries=MAX_RETRIES):
    """
    Download a single PDB file from AlphaFold Database.
    
    Args:
        afdb_id (str): AlphaFold Database ID (e.g., AF-P12345-F1)
        output_dir (Path): Directory to save the PDB file
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        tuple: (afdb_id, success, message)
    """
    filename = f"{afdb_id}.pdb"
    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename
    
    # Skip if file already exists
    if output_path.exists():
        return (afdb_id, True, "Already exists")
    
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_path)
            return (afdb_id, True, "Downloaded successfully")
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return (afdb_id, False, f"File not found (404): {url}")
            else:
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return (afdb_id, False, f"HTTP error {e.code}: {e.reason}")
                
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return (afdb_id, False, f"URL error: {e.reason}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return (afdb_id, False, f"Unexpected error: {str(e)}")
    
    return (afdb_id, False, "Max retries exceeded")

def read_afdb_ids(input_file):
    """
    Read AlphaFold Database IDs from the input file.
    
    Args:
        input_file (str): Path to the input file
        
    Returns:
        list: List of AFDB IDs
    """
    try:
        with open(input_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

def main():
    """Main function to orchestrate the download process."""
    print("AlphaFold Database PDB Downloader")
    print("=" * 40)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Read AFDB IDs
    print(f"Reading IDs from: {INPUT_FILE}")
    afdb_ids = read_afdb_ids(INPUT_FILE)
    print(f"Found {len(afdb_ids)} IDs to download")
    
    if not afdb_ids:
        print("No IDs found in input file.")
        sys.exit(0)
    
    # Download files sequentially
    print("Starting downloads...")
    
    successful_downloads = 0
    failed_downloads = 0
    already_exists = 0
    
    # Process downloads with progress bar
    for afdb_id in tqdm(afdb_ids, desc="Downloading"):
        afdb_id, success, message = download_pdb_file(afdb_id, output_dir)
        
        if success:
            if "Already exists" in message:
                already_exists += 1
            else:
                successful_downloads += 1
        else:
            failed_downloads += 1
            print(f"\nFailed to download {afdb_id}: {message}")
    
    # Summary
    print("\n" + "=" * 40)
    print("Download Summary:")
    print(f"  Total IDs: {len(afdb_ids)}")
    print(f"  Successfully downloaded: {successful_downloads}")
    print(f"  Already existed: {already_exists}")
    print(f"  Failed downloads: {failed_downloads}")
    
    if failed_downloads > 0:
        print(f"\nNote: {failed_downloads} files failed to download.")
        print("Check the error messages above for details.")
    
    print(f"\nAll available files have been downloaded to: {output_dir}")

if __name__ == "__main__":
    main() 