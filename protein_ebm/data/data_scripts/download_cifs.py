import os
import requests

# Directory to save the downloaded mmCIF files
output_dir = "mmcifs"  # Replace with your desired directory
#output_dir = "/home/gridsan/jroney/solab/ProteinEBM/data/skempi/dsmbind_pdbs"


# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the list of PDB codes to download
print("Reading PDB codes...")
with open("cath-dataset-nonredundant-S40.list", 'r') as file:
    pdb_codes = sorted(list(set([line[:4].upper() for line in file if line.strip()])))
print(f"Found {len(pdb_codes)} PDB codes.")

#pdb_codes = [x.strip() for x in open("/home/gridsan/jroney/solab/ProteinEBM/data/skempi/filtered_dsmbind_pdbs.txt",'r').readlines()]

# Get list of existing files in output directory
existing_files = set(os.listdir(output_dir))
print(f"Found {len(existing_files)} existing files in output directory.")

# Base URL for downloading mmCIF files
base_url = "https://files.rcsb.org/download"

# Open the log file in append mode
for i, pdb_code in enumerate(pdb_codes, start=1):
    # Check if base file exists
    print(f"[{i}/{len(pdb_codes)}] Downloading {pdb_code}...")
    assem_idx = 0

    if f"{pdb_code}.cif" in existing_files:
        assem_idx += 1
        continue

    while True:
        if assem_idx == 0:
            mmcif_url = f"{base_url}/{pdb_code}.cif"
            output_path = os.path.join(output_dir, f"{pdb_code}.cif")


        else:
            assembly_filename = f"{pdb_code}-assembly{assem_idx}.cif"
            mmcif_url = f"{base_url}/{pdb_code}-assembly{assem_idx}.cif"
            output_path = os.path.join(output_dir, assembly_filename)

        try:
            response = requests.get(mmcif_url, stream=True)
            response.raise_for_status()  # Raise an error for HTTP issues
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  Successfully downloaded {output_path}.")
        except requests.exceptions.RequestException as e:
            break

        assem_idx += 1

print("Download process completed.")
