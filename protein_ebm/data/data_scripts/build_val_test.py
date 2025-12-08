from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import torch
from protein_ebm.data.protein_utils import restype_order, restype_3to1
from protein_ebm.data.protein_utils import residues_to_features


parser = PDBParser(QUIET=True)


val_aatype = []
val_idx = []
val_contact = []
val_atom37_pos = []
val_atom37_mask = []
val_chainids = []
with open("../data_lists/validation_decoys.txt") as f:
    for line in f:
        pdb = line.strip()
        residues = [r for r in parser.get_structure(pdb, f"../../../eval_data/decoys/natives/{pdb}.pdb")[0].get_residues() if is_aa(r)]
        
        # Use the new function to get all data
        atom_positions, atom_mask, aatype, residue_idx = residues_to_features(residues)
        
        val_aatype.append(aatype)
        val_idx.append(residue_idx)
        val_contact.append(torch.zeros(len(residues), dtype=torch.long))
        val_chainids.append(torch.zeros(len(residues), dtype=torch.long))
        val_atom37_pos.append(atom_positions)
        val_atom37_mask.append(atom_mask)


test_aatype = []
test_idx = []
test_contact = []
test_atom37_pos = []
test_atom37_mask = []
test_chainids = []
with open("../data_lists/test_decoys.txt") as f:
    for line in f:
        pdb = line.strip()
        residues = [r for r in parser.get_structure(pdb, f"../../../eval_data/decoys/natives/{pdb}.pdb")[0].get_residues() if is_aa(r)]
        
        # Use the new function to get all data
        atom_positions, atom_mask, aatype, residue_idx = residues_to_features(residues)
        
        test_aatype.append(aatype)
        test_idx.append(residue_idx)
        test_contact.append(torch.zeros(len(residues), dtype=torch.long))
        test_chainids.append(torch.zeros(len(residues), dtype=torch.long))
        test_atom37_pos.append(atom_positions)
        test_atom37_mask.append(atom_mask)

# Save validation data with all fields
torch.save({
    "aatype": val_aatype,
    "idx": val_idx,
    "contacts": val_contact,
    "atom37": val_atom37_pos,
    "atom37_mask": val_atom37_mask,
    "chain_ids": val_chainids
}, "../../../eval_data/val_protein_data.pt")

# Save test data with all fields
torch.save({
    "aatype": test_aatype,
    "idx": test_idx,
    "contacts": test_contact,
    "atom37": test_atom37_pos,
    "atom37_mask": test_atom37_mask,
    "chain_ids": test_chainids
}, "../../../eval_data/test_protein_data.pt")