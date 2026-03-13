
import torch
import numpy as np
import os
from protein_ebm.data.protein_utils import restype_3to1_with_unk, atom_types, atom_order, atom_type_num, restype_order_with_x

canonical_tokens = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token
]

tokens = [
    "<pad>",
    "-",
    *canonical_tokens,
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]

token_ids = {token: i for i, token in enumerate(tokens)}
num_tokens = len(tokens)

chain_types = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
chain_type_ids = {chain: i for i, chain in enumerate(chain_types)}

def extract_atom_names(name_data):
    return "".join([chr(x+32) for x in name_data if x != 0])

targets = os.listdir("/home/ubuntu/boltz_training_data/rcsb_processed_targets/structures/")



fasta_lines = []
atom_positions_all = []
atom_masks_all = []
aatypes_all = []
residue_idxs_all = []
chain_ids_all = []
present_all = []
target_ids_all = []
chain_names_all = []
external_contacts_all = []

for tidx,t in enumerate(targets):

    print(t)
    if tidx % 100 == 0:
        print(f"Processing {tidx} / {len(targets)}")

    atom_positions = []
    atom_masks = []
    aatypes = []
    residue_idxs = []
    chain_ids = []
    present = []
    chain_names = []
    saved_residues = set()  # Track which residues are being saved


    dat = dict(np.load("/home/ubuntu/boltz_training_data/rcsb_processed_targets/structures/" + t)) # no lazy loading

    for cidx, c in enumerate(dat['chains']):

        start_res = c[-2]
        end_res = c[-2] + c[-1]
        resdata = dat['residues'][start_res:end_res]

        if c[1] != chain_type_ids["PROTEIN"] or not all([tokens[r[1]] in restype_3to1_with_unk for r in resdata]):
            continue
        
        # Extract sequence directly from chain residues
        seq = "".join([restype_3to1_with_unk[tokens[r[1]]] for r in resdata])

        if sum([x == "X" for x in seq]) / len(seq) > 0.1:
            print(f"Skipping {t} {c[0]} because of too many unknown residues")
            continue
        
        chain_name = f"{t.split('.')[0]}_{c[0]}"
        fasta_lines.append(f">{chain_name}\n{seq}")
        
        # Mark residues in this chain as saved
        for res_idx in range(start_res, end_res):
            saved_residues.add(res_idx)

        for ridx, r in enumerate(resdata):
            start_atom = r[3]
            end_atom = start_atom + r[4]

            atomdata = dat['atoms'][start_atom:end_atom]

            pos = torch.zeros((atom_type_num, 3))
            mask = torch.zeros(atom_type_num)
            present_mask = torch.zeros(atom_type_num)
            
            for aidx, a in enumerate(atomdata):
                name = extract_atom_names(a[0])
                if name not in atom_order:
                    continue
                atom_index = atom_order[name]
                pos[atom_index] = torch.from_numpy(a[3])
                mask[atom_index] = 1.
                present_mask[atom_index] = float(a[-2])

            if mask[1] != 1:
                pos[1] = torch.from_numpy(dat['atoms'][r[-4]][3]) # substitue CA atom with "center atom"

            atom_positions.append(pos)
            atom_masks.append(mask)
            aatypes.append(restype_order_with_x[restype_3to1_with_unk[tokens[r[1]]]])
            residue_idxs.append(r[2])
            present.append(present_mask)
        
        chain_names.append(c[0])
        chain_ids.extend([cidx] * len(resdata))
    
    if len(atom_positions) == 0:
        continue

    # Compute external contacts at residue center level
    external_contacts = np.zeros(len(atom_positions), dtype=bool)
    
    if len(atom_positions) > 0:
        # Get center positions of saved residues where CA is present
        stacked_positions = torch.stack(atom_positions)
        stacked_present = torch.stack(present)
        ca_present = stacked_present[:, 1].numpy() > 0  # Check if CA atoms are present
        
        saved_residue_centers = stacked_positions[ca_present, 1, :].numpy()  # CA atoms at index 1
        present_residue_indices = np.where(ca_present)[0]
        
        # Get center positions of all excluded residues (skip NONPOLYMER chains)
        excluded_centers = []
        for chain_idx, chain in enumerate(dat['chains']):
            # Skip NONPOLYMER chains
            if chain[1] == chain_type_ids["NONPOLYMER"]:
                continue
            
            start_res = chain[-2]
            end_res = chain[-2] + chain[-1]
            
            for res_idx in range(start_res, end_res):
                if res_idx not in saved_residues:
                    res = dat['residues'][res_idx]
                    center_atom_idx = res[-4]
                    if dat['atoms'][center_atom_idx][-2]:  # Check if center present
                        center_pos = dat['atoms'][center_atom_idx][3]
                        excluded_centers.append(center_pos)
        
        if len(excluded_centers) > 0 and len(saved_residue_centers) > 0:
            excluded_centers = np.array(excluded_centers)
            
            # Compute distances between saved residue centers and excluded residue centers
            center_distances = np.linalg.norm(
                saved_residue_centers[:, np.newaxis, :] - excluded_centers[np.newaxis, :, :],
                axis=2
            )
            
            # Mark residues with any external residue center within 10A
            has_contact = np.any(center_distances < 9.0, axis=1)
            external_contacts[present_residue_indices] = has_contact

    target_ids_all.append(t.split(".")[0])
    chain_names_all.append(chain_names)
    atom_positions_all.append(torch.stack(atom_positions))
    atom_masks_all.append(torch.stack(atom_masks))
    aatypes_all.append(torch.tensor(aatypes))
    chain_ids_all.append(torch.tensor(chain_ids))
    present_all.append(torch.stack(present))
    external_contacts_all.append(torch.tensor(external_contacts, dtype=torch.bool))
    residue_idxs_all.append(torch.tensor(residue_idxs))

with open("boltz_training_fasta.fasta", "w") as f:
    f.write("\n".join(fasta_lines))

torch.save({
    "target_ids": target_ids_all,
    "chain_names": chain_names_all,
    "atom_positions": atom_positions_all,
    "atom_masks": atom_masks_all,
    "aatypes": aatypes_all,
    "chain_ids": chain_ids_all,
    "present": present_all,
    "external_contacts": external_contacts_all,
    "residue_idxs": residue_idxs_all
}, "boltz_training_data.pt")