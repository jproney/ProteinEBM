from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import os
import numpy as np
import re
import math
import torch
from protein_ebm.data.protein_utils import residues_to_features

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
input_file = "skempi/filtered_dsmbind_ids_strict_ff_len.txt" 
pdb_directory = "skempi/dsmbind_pdbs"
rg_output_file = "skempi/complex_rgs_strict_ff_contacts.tsv"
domain_output_file = "skempi/final_dsmbind_training_complexes_strict_ff_contacts.txt"
tensor_save_file = "skempi/training_dsmbind_complex_data_strict_ff_contacts.pt"
skipped_domains_file = "skempi/skipped_dsmbind_complexes_strict_ff_contacts.txt"

# Write headers
with open(rg_output_file, 'w') as out_f:
    out_f.write("PDB\tDomain\tRg\tResidueCount\tPowerLaw\tLogDiff\n")
with open(skipped_domains_file, 'w') as out_f:
    out_f.write("PDB\tDomain\tReason\n")

# Read the domain definitions
print("Reading domain definitions...")
with open(input_file, 'r') as f:
    complex_list = [line.strip().split() for line in f]
    complex_list = [l[0].split("_") + [l[1], l[2]] for l in complex_list]


print(f"Found {len(complex_list)} domains to process.")
# Initialize PDB parser
parser_renum = MMCIFParser(auth_residues=False, QUIET=True)

# Process each domain
last_pdb_file = None
structure_renum = None

# List of domains that pass the Rg filter
filtered_domains_list = []

# Initialize lists for tensors
accumulated_atom37 = []
accumulated_idx = []
accumulated_aatype = []
accumulated_contactflag = [] 
accumulated_atom37_mask = []
accumulated_chain_ids = []
accumulated_ids = []

for i, line in enumerate(complex_list, start=1):
    pdb_code = line[0]
    chaina = line[1]
    chainb = line[2]
    len_a = int(line[3])
    len_b = int(line[4])


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

    print(f"[{i}/{len(complex_list)}] Processing {line}...")

    residues_a = []
    residues_b = []

    for pdb_file in pdb_files:

        if pdb_file != last_pdb_file:
            try:
                structure_renum = parser_renum.get_structure(pdb_code, pdb_file)
                last_pdb_file = pdb_file
            except Exception as e:
                continue

        for c in structure_renum.get_chains():
            if c.id == chaina:
                residues_a = [r for r in c.get_residues() if is_aa(r, standard=True) and all([atom in r for atom in ["N", "CA", "C"]])]
            elif c.id == chainb:
                residues_b = [r for r in c.get_residues() if is_aa(r, standard=True) and all([atom in r for atom in ["N", "CA", "C"]])]

        missing_chains = [c for c,r in zip([chaina, chainb], [residues_a, residues_b]) if len(r) == 0]
        if len(missing_chains) > 0:
            print(f"Chains {missing_chains} missing for {line}, skipping.")
            with open(skipped_domains_file, 'a') as out_f:
                out_f.write(f"{pdb_code}\t{line}\tNo residues found for chain: {missing_chains}\n")
            continue

        break

    if abs(len(residues_a) - len_a) > 10 or abs(len(residues_b) - len_b) > 10:
        print(f"Residue count mismatch for {line}, skipping.")
        with open(skipped_domains_file, 'a') as out_f:
            out_f.write(f"{pdb_code}\t{line}\tResidue count mismatch: {(len(residues_a), len(residues_b))}\n")
        continue

    residues = residues_a + residues_b
    chainids = [0] * len(residues_a) + [1] * len(residues_b)

    ca_cb = [
        (residue['CB'] if 'CB' in residue else residue['CA']) for residue in structure_renum.get_residues() if
        ('CA' in residue or 'CB' in residue) and is_aa(residue)
    ]
    neighbor_search = NeighborSearch(ca_cb)
    contactflag = torch.tensor(get_inter_contacts(residues, neighbor_search), dtype=torch.long)

    # Calculate the radius of gyration
    rg = calculate_radius_of_gyration(residues)
    if rg is None:
        print(f"  No valid atoms found in residues for {line}, skipping.")
        with open(skipped_domains_file, 'a') as out_f:
            out_f.write(f"{pdb_code}\t{line}\tNo valid atoms found for radius of gyration calculation\n")
        continue

    # Get the residue count
    residue_count = len(residues)

    power_law = 2 * residue_count ** 0.4
    logdiff = math.log10(rg) - math.log10(power_law)

    # Write the result to the output file
    with open(rg_output_file, 'a') as out_f:
        out_f.write(f"{pdb_code}\t{line}\t{rg:.3f}\t{residue_count}\t{power_law}\t{logdiff}\n")
    print(f"  Calculated Rg: {rg:.3f}, Residue Counts: {(len(residues_a), len(residues_b))}, Expected Residues {(len_a, len_b)}, Log Difference: {logdiff:.3f}")

    if residue_count < 500: 

        filtered_domains_list.append(line)
        atom_positions, atom_mask, aatype = residues_to_features(residues)
        accumulated_atom37.append(atom_positions)
        accumulated_atom37_mask.append(atom_mask)
        accumulated_aatype.append(aatype)
        accumulated_idx.append(torch.tensor([r.id[1] for r in residues], dtype=torch.long))
        accumulated_contactflag.append(contactflag)
        accumulated_ids.append(f"{pdb_code}_{chaina}_{chainb}")
        accumulated_chain_ids.append(torch.tensor(chainids))

        with open(domain_output_file, 'a') as f:
            f.write('\t'.join(line) + '\n')

torch.save({
    "aatype": accumulated_aatype,
    "idx": accumulated_idx, 
    "contacts": accumulated_contactflag, 
    "atom37": accumulated_atom37, 
    "atom37_mask": accumulated_atom37_mask,
    "chain_ids": accumulated_chain_ids,
    "ids": accumulated_ids
}, tensor_save_file)