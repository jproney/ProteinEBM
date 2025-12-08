from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import os
import numpy as np
import re
import math
import torch
from protein_utils import residues_to_features

def calculate_radius_of_gyration(residues):
    """
    Calculate the radius of gyration (Rg) for a given set of residues.
    """
    coords = []
    for residue in residues:
        if "CA" in residue:  # Use alpha-carbon atoms for calculation
            coords.append(residue["CA"].coord)
    if not coords:
        return None  # No valid atoms found

    coords = np.array(coords)
    center_of_mass = coords.mean(axis=0)
    rg = np.sqrt(((coords - center_of_mass) ** 2).sum(axis=1).mean())
    return rg
    

def parse_residue_ranges(range_string):
    """
    Parse a residue range string like '117-140:A,-5-6:A' into individual ranges.

    Args:
        range_string (str): Residue range string.

    Returns:
        list of tuples: List of (start, end, chain_id) for each range.
    """
    ranges = []
    # Split by comma for multiple ranges
    for part in range_string.split(","):
        # Match ranges and chain IDs using a regular expression
        match = re.match(r"(-?\d+)([A-Z]?)-(-?\d+)([A-Z]?):([A-Za-z0-9])", part)
        if not match:
            raise ValueError(f"Invalid residue range format: {part}")
        start_num, start_let, end_num, end_let, chain_id = match.groups()
        ranges.append((int(start_num), start_let, int(end_num), end_let, chain_id))
    return ranges

# Input files and directories
input_file = "proteina/ted_domains_seq_filtered_strict_ff_orphans.txt" # "cath_s40_seq_filtered.txt"
pdb_directory = "proteina/afdb_pdbs"
rg_output_file = "proteina/domain_rgs_proteina_strict_ff_orphans.tsv"
domain_output_file = "proteina/final_ted_training_domains_strict_ff_orphans.txt"
tensor_save_file = "proteina/training_ted_domain_data_strict_ff_orphans.pt"
skipped_domains_file = "proteina/skipped_ted_domains_strict_ff_orphans.txt"

# Write headers
with open(rg_output_file, 'w') as out_f:
    out_f.write("PDB\tDomain\tRg\tResidueCount\tPowerLaw\tLogDiff\n")
with open(skipped_domains_file, 'w') as out_f:
    out_f.write("PDB\tDomain\tReason\n")

# Read the domain definitions
print("Reading domain definitions...")
with open(input_file, 'r') as f:
    domain_list = [line.strip().split() for line in f if line.strip()]

if domain_list[0][0].startswith('AF-'): # AlphaFold domains
    domain_list = [(line[0], '_', '_', '_') if line[0].endswith('v4') else
                   ("_".join(line[0].split("_")[:2]), line[0].split("_")[2], "", ",".join([x+":A" for x in line[0].split("_")[3:]])) for line in domain_list]
    domain_list.sort(key=lambda x: x[0])
    # Initialize PDB parser
    parser_auth = PDBParser(QUIET=True)
    parser_renum = PDBParser(QUIET=True)
else:
    # Sort the domain list by PDB code
    domain_list.sort(key=lambda x: x[0][:4])
    print(f"Found {len(domain_list)} domains to process.")
    # Initialize PDB parser
    parser_auth = MMCIFParser(auth_residues=True, QUIET=True)
    parser_renum = MMCIFParser(auth_residues=False, QUIET=True)

# Process each domain
last_pdb_file = None
structure_auth = None
structure_renum = None

# List of domains that pass the Rg filter

filtered_domains_list = []

# Initialize lists for tensors
accumulated_idx = []
accumulated_aatype = []
accumulated_contactflag = []
accumulated_pos = []
accumulated_pos_mask = []
accumulated_chain_ids = []
accumulated_ids = []
accumulated_residue_mask = []

for i, line in enumerate(domain_list, start=1):
    pdb_code = line[0][:4]
    domain_id = line[0]
    residue_ranges = line[-1]
    if residue_ranges != '_':
        parsed_ranges = parse_residue_ranges(residue_ranges)
    else:
        parsed_ranges = None

    if domain_id.startswith('AF-'):
        pdb_code = domain_id
        pdb_files =  [os.path.join(pdb_directory, pdb_code + ".pdb")]
    else:
        pdb_files = [os.path.join(pdb_directory, f"{pdb_code.upper()}-assembly1.cif")]
        if not os.path.exists(pdb_files[0]):
            pdb_files = [os.path.join(pdb_directory, f"{pdb_code.upper()}.cif")]
        else:
            cnt = 2
            nextfile = os.path.join(pdb_directory, f"{pdb_code.upper()}-assembly{cnt}.cif")
            while os.path.exists(nextfile):
                pdb_files.append(nextfile)
                cnt += 1
                nextfile = os.path.join(pdb_directory, f"{pdb_code.upper()}-assembly{cnt}.cif")

    print(f"[{i}/{len(domain_list)}] Processing {domain_id}...")

    residues = []
    chainids = []

    for pdb_file in pdb_files:

        if pdb_file != last_pdb_file:
            try:
                structure_auth = parser_auth.get_structure(pdb_code, pdb_file)
                structure_renum = parser_renum.get_structure(pdb_code, pdb_file)
                last_pdb_file = pdb_file
            except Exception as e:
                continue

        if not (0 in structure_renum and (parsed_ranges is None or all([rng[-1] in structure_renum[0] for rng in parsed_ranges]))): # make sure all chains in question are in the assembly
            continue

        expected_len = 0

        # Collect residues for all specified ranges
        if parsed_ranges is not None:
            for (start_num, start_let, end_num, end_let, chain_id) in parsed_ranges:
                chain_auth = structure_auth[0][chain_id]
                chain_renum = structure_renum[0][chain_id]
                chain_num = ord(chain_id) - ord('A')


                expected_len += end_num - start_num + 1

                for k, (residue, residue_renum) in enumerate(zip(chain_auth.get_residues(), chain_renum.get_residues())):

                    if ((start_num < int(residue.id[1]) or (start_num == int(residue.id[1]) and start_let <= residue.id[2].strip())) and
                            ((end_num > int(residue.id[1]) or (end_num == int(residue.id[1]) and end_let >= residue.id[2].strip())))  
                                and is_aa(residue, standard=True) and all([atom in residue for atom in ["N", "CA", "C"]])):
                        residues.append(residue_renum)
                        chainids.append(chain_num)
                residue_mask = [1] * len(residues)
                plddt = -1
            break
        else:
            residues = [r for r in structure_renum.get_residues() if is_aa(r, standard=True) and all([atom in r for atom in ["N", "CA", "C"]])]
            residue_mask = [r['CA'].get_bfactor() > 50 for r in residues]
            plddt = np.mean([r['CA'].get_bfactor() for r in residues])
            chainids = [0] * len(residues)


    if len(residues) == 0:
        print(f"  No residues found for ranges {residue_ranges}, skipping.")
        with open(skipped_domains_file, 'a') as out_f:
            out_f.write(f"{pdb_code}\t{domain_id}\tNo residues found for ranges: {residue_ranges}\n")
        continue

    # Calculate the radius of gyration
    rg = calculate_radius_of_gyration(residues)
    if rg is None:
        print(f"  No valid atoms found in residues for {domain_id}, skipping.")
        with open(skipped_domains_file, 'a') as out_f:
            out_f.write(f"{pdb_code}\t{domain_id}\tNo valid atoms found for radius of gyration calculation\n")
        continue

    # Get the residue count
    residue_count = len(residues)

    power_law = 2 * residue_count ** 0.4
    logdiff = math.log10(rg) - math.log10(power_law)

    # Write the result to the output file
    with open(rg_output_file, 'a') as out_f:
        out_f.write(f"{pdb_code}\t{domain_id}\t{rg:.3f}\t{residue_count}\t{power_law}\t{logdiff}\n")
    print(f"  Calculated Rg: {rg:.3f}, Residue Count: {residue_count}, Expected Residues {expected_len}, Log Difference: {logdiff:.3f}, pLDDT: {plddt:.3f}")

    if residue_count < 500: # bumping way up from 0.08

        filtered_domains_list.append(line)
        atom_positions, atom_mask, aatype = residues_to_features(residues)
        accumulated_chain_ids.append(torch.tensor(chainids))

        accumulated_pos.append(atom_positions)
        accumulated_pos_mask.append(atom_mask)
        accumulated_aatype.append(aatype)
        accumulated_idx.append(torch.tensor([r.id[1] for r in residues], dtype=torch.long))
        accumulated_contactflag.append(torch.tensor([0] * len(residues), dtype=torch.long)) # 0 means unknown
        accumulated_ids.append(domain_id)
        accumulated_residue_mask.append(torch.tensor(residue_mask, dtype=torch.bool))

        with open(domain_output_file, 'a') as f:
            f.write('\t'.join(line) + '\n')


torch.save({
    "aatype": accumulated_aatype,
    "idx": accumulated_idx, 
    "contacts": accumulated_contactflag, 
    "atom37": accumulated_pos, 
    "atom37_mask": accumulated_pos_mask,
    "chain_ids": accumulated_chain_ids,
    "residue_mask": accumulated_residue_mask,
    "ids": accumulated_ids
}, tensor_save_file)