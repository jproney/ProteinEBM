from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import torch
from protein_ebm.data.protein_utils import restype_order, restype_3to1
from protein_ebm.data.protein_utils import residues_to_features
import os


parser = PDBParser(QUIET=True)


rmsd = {(l.split()[0], l.split()[1]) : float(l.split()[-1]) for l in open("../../../eval_data/decoy_data/rmsd.txt",'r').readlines()}
tmscore = {(l.split()[0], l.split()[1]) : float(l.split()[-1]) for l in open("../../../eval_data/decoy_data/tmscore.txt",'r').readlines()}
rosetta = {(l.split()[0], l.split()[1]) : float(l.split()[-1]) for l in open("../../../eval_data/decoy_data/rosettascore.txt",'r').readlines()}

all_lines = open("../data_lists/validation_decoys.txt").readlines() + open("../data_lists/test_decoys.txt").readlines()

for line in all_lines:

    print(line)
    pdb = line.strip()
    if os.path.exists(f'../../../eval_data/decoys/{pdb}.pt'):
        continue

    decoy_list = os.listdir(f"../../../eval_data/decoys/{pdb}/")

    native_residues = [r for r in parser.get_structure(pdb, f"../../../eval_data/decoys/natives/{pdb}.pdb")[0].get_residues() if is_aa(r)]
    aatype = torch.tensor([restype_order[restype_3to1[r.get_resname()]] for r in native_residues], dtype=torch.long)
    idx = torch.tensor([r.id[1] for r in native_residues], dtype=torch.long)

    decoy_rms = []
    decoy_tm = []
    decoy_ros = []
    decoy_aatype = []
    decoy_idxs = []
    decoy_atom37 = []
    decoy_atom37_mask = []
    for decoy in decoy_list:
        residues = [r for r in parser.get_structure(pdb, f"../../../eval_data/decoys/{pdb}/{decoy}")[0].get_residues() if is_aa(r)]
        atom_positions, atom_mask, _, residue_idx = residues_to_features(residues)
        decoy_rms.append(rmsd[(pdb, decoy)])
        decoy_tm.append(tmscore[(pdb, decoy)])
        decoy_ros.append(rosetta[(pdb, decoy)])
        decoy_aatype.append("".join([restype_3to1[r.get_resname()] for r in residues]))
        decoy_idxs.append(residue_idx)
        decoy_atom37.append(atom_positions)
        decoy_atom37_mask.append(atom_mask)


    # truncate everything to the min length of all the decoys
    lens = [len(s) for s in decoy_aatype]
    minidx = lens.index(min(lens))

    shifts = [s.index(decoy_aatype[minidx]) for s in decoy_aatype]        

    
    decoy_atom37 = torch.stack([a[s:] for a,s in zip(decoy_atom37, shifts)])
    decoy_atom37_mask = torch.stack([m[s:] for m,s in zip(decoy_atom37_mask, shifts)])

    aatype = aatype[shifts[0]:].unsqueeze(0).expand(decoy_atom37.shape[:-2])
    idx = idx[shifts[0]:].unsqueeze(0).expand(decoy_atom37.shape[:-2])

    torch.save({
        'aatype': aatype,
        'idx': idx,
        'rmsd': torch.tensor(decoy_rms),
        'tmscore': torch.tensor(decoy_tm),
        'rosetta': torch.tensor(decoy_ros),
        'atom37': decoy_atom37,
        'atom37_mask': decoy_atom37_mask,
    }, f'../../../eval_data/decoys/{pdb}.pt')