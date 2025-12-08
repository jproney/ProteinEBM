import mdtraj as md
import numpy as np
import os
import torch
from protein_ebm.data.protein_utils import restype_order, restype_3to1, atom_order

base_md_dir = "/home/gridsan/jroney/solab/ProteinEBM/data/bioemu/MSR_cath2/"
filtered = set([line.strip() for line in open("/home/gridsan/jroney/solab/ProteinEBM/data/bioemu/filtered_msr_cath_ids_strict_ff.txt").readlines()])
cath_traj_dirs = os.listdir(base_md_dir)


alltraj = []
for trajname in cath_traj_dirs:
    if trajname in filtered:
        print("skipping", trajname)
        continue

    traj_dir = base_md_dir + trajname + "/trajs"

    if not os.path.exists(traj_dir):
        continue

    print(trajname)
    traj_files = os.listdir(traj_dir)
    concat_traj = []
    for file in traj_files:
        if file.endswith(".xtc"):
            xtc_file = os.path.join(traj_dir, file)
            traj = md.load_xtc(xtc_file, top=os.path.join(base_md_dir + trajname, "topology.pdb"))
            concat_traj.append(traj.xyz)
    concat_traj = np.concatenate(concat_traj, axis=0)

    reslist = list(traj.topology.residues)
    atom37 = torch.zeros((concat_traj.shape[0], len(reslist), 37, 3))
    atom37_mask = torch.zeros((len(reslist), 37))
    aatype = torch.zeros(len(reslist), dtype=torch.long)
    residue_idx = torch.zeros(len(reslist), dtype=torch.long)
    # Get CA indices

    for i, res in enumerate(reslist):
        for a in res.atoms:
            if a.name not in atom_order:
                if not a.name.startswith("H"):
                    print(a.name)
                continue
            id = atom_order[a.name]
            atom37[:, i, id, :] = torch.from_numpy(concat_traj[:, a.index, :])
            atom37_mask[i, id] = 1
            aatype[i] = restype_order[restype_3to1[res.name]]
            residue_idx[i] = res.index

    alltraj.append((atom37, atom37_mask, aatype, residue_idx, trajname))
torch.save(alltraj, "/home/gridsan/jroney/solab/ProteinEBM/data/bioemu/cath2_md_dataset_strict_ff.pt")


allperms = [torch.randperm(x[0].shape[0]) for x in alltraj]

nshards = 10
for i in range(nshards):
    shard_dict = {
        'atom37': [],
        'atom37_mask': [],
        'aatype': [],
        'idx': [],
        'contacts': [],
        'ids': [],
    }

    for j, (atom37, atom37_mask, aatype, residue_idx, trajname) in enumerate(alltraj):
        print(i,j)
        nframe = atom37.shape[0]
        shard_size = nframe // nshards

        atom37_shard = atom37[allperms[j][i*shard_size:(i+1)*shard_size]]
        B, N, _, _ = atom37_shard.shape

        shard_dict['atom37'].extend(atom37_shard.unbind(dim=0))
        shard_dict['atom37_mask'].extend([atom37_mask] * B)
        shard_dict['aatype'].extend([aatype] * B)
        shard_dict['idx'].extend([residue_idx] * B)
        shard_dict['contacts'].extend([torch.zeros(N)] * B)
        shard_dict['ids'].extend([trajname] * B)

    torch.save(shard_dict, f"/home/gridsan/jroney/solab/ProteinEBM/data/bioemu/cath_md_shards/cath2_md_dataset_strict_ff_{i}.pt")
    del shard_dict