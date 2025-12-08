from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import os
import numpy as np
import re
import math
import torch
from protein_ebm.data.protein_utils import residues_to_features
import glob

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

def find_close_residues(target_residue, neighbor_search, distance_cutoff=8.0):

    # Get all atoms in the target residue
    target_atom = target_residue['CB'] if 'CB' in target_residue else target_residue['CA']

    # Find all atoms within the distance cutoff of the target residue's atoms
    close_atoms = neighbor_search.search(target_atom.coord, distance_cutoff)

    # Collect residues from the close atoms, avoiding duplicates
    close_residues = set()
    for atom in close_atoms:
        parent_residue = atom.get_parent()
        if parent_residue != target_residue:
            close_residues.add(parent_residue)

    return list(close_residues)

def get_inter_contacts(residues, neighbor_search, linear_thresh=6):
    has_contact = []
    res_set = set(residues)
    for r in residues:
        inter_contacts = [r2 for r2 in find_close_residues(r, neighbor_search) if r2 not in res_set and abs(int(r2.id[1]) - int(r.id[1])) > linear_thresh]
        has_contact.append(2 if len(inter_contacts) > 0 else 1)

    return has_contact


# Input files and directories
input_file = "cath_s40_seq_filtered_strict_ff_orphans.txt"
pdb_directory = "dompdb"
mmcif_directory = "mmcifs"
rg_output_file = "domain_rgs_refiltered_strict_ff_orphans_contacts.tsv"
domain_output_file = "final_training_domains_strict_ff_orphans_contacts.txt"
tensor_save_file = "training_domain_data_strict_ff_orphans_contacts.pt"

# Write headers
with open(rg_output_file, 'w') as out_f:
    out_f.write("Domain\tRg\tResidueCount\tPowerLaw\tLogDiff\n")

# Initialize PDB parser
parser = PDBParser(QUIET=True)
mmcif_parser = MMCIFParser(QUIET=True)

# Read the domain definitions
print("Reading domain definitions...")
with open(input_file, 'r') as f:
    domain_list = [line.strip().split() for line in f if line.strip()]

# Sort the domain list by PDB code
print(f"Found {len(domain_list)} domains to process.")

# Process each domain
last_pdb_file = None
structure_auth = None
structure_renum = None

# List of domains that pass the Rg filter

filtered_domains_list = []

# Initialize lists for tensors
accumulated_idx = []
accumulated_aatype = []
accumulated_contactflag = [] # 0 means unknown, 1 means no contact, 2 means contact
accumulated_pos = []
accumulated_pos_mask = []
accumulated_chain_ids = []
accumulated_ids = []

for i, line in enumerate(domain_list, start=1):
    
    domain_id = line[0]
    residue_ranges = line[-1]

    if residue_ranges != "_":
        parsed_ranges = parse_residue_ranges(residue_ranges)
        structure = parser.get_structure(domain_id, pdb_directory + '/' + domain_id)
        residues = [r for r in structure.get_residues() if "CA" in r and is_aa(r, standard=True)]

    else:
        pdb_code = domain_id[:4].upper()
        mmcif_files = sorted(glob.glob(f"{mmcif_directory}/{pdb_code}-assembly*.cif")) + [f"{mmcif_directory}/{pdb_code}.cif"]

        stop = False
        for f in mmcif_files:
            structure = mmcif_parser.get_structure(domain_id, f)
            for model in structure:
                for chain in model:
                    if chain.id == domain_id[-1]:
                        residues = [r for r in chain.get_residues() if "CA" in r and is_aa(r, standard=True)]
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
                

    chainids = [0] * len(residues)

    # Calculate the radius of gyration
    rg = calculate_radius_of_gyration(residues)
    if rg is None:
        print(f"  No valid atoms found in residues for {domain_id}, skipping.")
        continue

    # Get the residue count
    residue_count = len(residues)

    power_law = 2 * residue_count ** 0.4
    logdiff = math.log10(rg) - math.log10(power_law)

    # Write the result to the output file
    with open(rg_output_file, 'a') as out_f:
        out_f.write(f"{domain_id}\t{rg:.3f}\t{residue_count}\t{power_law}\t{logdiff}\n")
    print(f"  Calculated Rg: {rg:.3f}, Residue Count: {residue_count}, Log Difference: {logdiff:.3f}")

    if residue_ranges == "_":
        ca_cb = [
            (residue['CB'] if 'CB' in residue else residue['CA']) for residue in structure.get_residues() if
            ('CA' in residue or 'CB' in residue) and is_aa(residue)
        ]
        neighbor_search = NeighborSearch(ca_cb)
        contactflag = torch.tensor(get_inter_contacts(residues, neighbor_search), dtype=torch.long)
    else:
        contactflag = torch.tensor([0] * len(residues), dtype=torch.long)

    if residue_count < 500:

        filtered_domains_list.append(line)
        atom_positions, atom_mask, aatype = residues_to_features(residues)
        accumulated_chain_ids.append(torch.tensor(chainids))

        accumulated_pos.append(atom_positions)
        accumulated_pos_mask.append(atom_mask)
        accumulated_aatype.append(aatype)
        accumulated_idx.append(torch.tensor([r.id[1] for r in residues], dtype=torch.long))
        accumulated_contactflag.append(contactflag)
        accumulated_ids.append(domain_id)

        with open(domain_output_file, 'a') as f:
            f.write('\t'.join(line) + '\n')


torch.save({
    "aatype": accumulated_aatype,
    "idx": accumulated_idx, 
    "contacts": accumulated_contactflag, 
    "atom37": accumulated_pos, 
    "atom37_mask": accumulated_pos_mask,
    "chain_ids": accumulated_chain_ids,
    "ids": accumulated_ids
}, tensor_save_file)
