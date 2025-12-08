import yaml
from ml_collections import ConfigDict
import torch
from argparse import ArgumentParser
from protein_ebm.model.r3_diffuser import R3Diffuser
from torch.utils.data import TensorDataset
from protein_ebm.model.ebm import ProteinEBM
from protein_ebm.model.boltz_utils import center_random_augmentation
from scipy.stats import spearmanr

parser = ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('chkpt', type=str)
parser.add_argument('decoy_list', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--bsize', type=int, default=256)
parser.add_argument('--t_min', type=float, default=0.0)
parser.add_argument('--t_max', type=float, default=1.0)
parser.add_argument('--n_samples', type=int, default=401)
parser.add_argument('--ensemble_size', type=int, default=0)
parser.add_argument('--mask_seq', action='store_true')
parser.add_argument('--template_self_condition', action='store_true', help='Use template for self-conditioning')
parser.add_argument('--inference_self_condition', action='store_true', help='Generate the self-conditioning input via the normal path')
parser.add_argument('--log_noise_nrg', action='store_true', help='Generate the self-conditioning input via the normal path')
parser.add_argument('--scdiff_mask_sc', action='store_true', help='Mask out the non-backbone atoms for a sidechain diffusion model')
parser.add_argument('--score_norm_nrg', action='store_true', help='Use the norm of the score instead of the energy')

args = parser.parse_args()

# Load config
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
config = ConfigDict(config)

diffuser = R3Diffuser(config.diffuser)
model = ProteinEBM(config.model, diffuser).cuda()


# Load model weights from regular checkpoint
ckpt = torch.load(args.chkpt, weights_only=False)
model.load_state_dict({k[len("model."):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model')})
model.eval()

all_decoys = [s.strip() for s in open(args.decoy_list).readlines()]

energy_dict = {}
spearman_dict = {}

for d in all_decoys:
    print(d)
    decoys = torch.load(f'../../eval_data/decoys/{d}.pt', weights_only=False)
    decoy_nrg = []

    if decoys['atom37'].shape[1] > 175:
        bsize = args.bsize // 2
    else:
        bsize = args.bsize

    with torch.no_grad():
        for t in torch.linspace(args.t_min, args.t_max, args.n_samples):
            print(t)
            time_nrg = []

            # Process atom37 coordinates to get N,CA,C atoms and flatten to 9-dim features
            if config.model.diffuse_sidechain:
                atoms = decoys['atom37']
            else:
                atoms = decoys['atom37'][..., 1, :]  # Just the CA

            B, N = atoms.shape[:2]

            flat_coords = atoms.reshape([B,-1,3])  # [B, N*(37 or 1), 3]

            if config.model.diffuse_sidechain:
                mask = decoys['atom37_mask']
                if args.scdiff_mask_sc:
                    mask[..., 3:] = 0
                flat_mask = mask.reshape(B, N*37)
            else:
                mask = torch.ones((B, N), device=flat_coords.device)
                flat_mask = mask

            # Apply random augmentation
            aug_coords = center_random_augmentation(flat_coords, flat_mask)

            if config.model.diffuse_sidechain:
                aug_coords = aug_coords * flat_mask[..., None]

            # Reshape to [B, N, 9] for diffusion
            atoms = aug_coords.reshape(B,N, 37*3 if config.model.diffuse_sidechain else 3)

            for i in range(args.ensemble_size if args.ensemble_size > 0 else 1):
                if args.ensemble_size > 0 and t > 0:
                    r_noisy, _ = diffuser.forward_marginal(atoms.numpy(), t.item())
                    r_noisy = torch.tensor(r_noisy, dtype=torch.float)
                else:
                    r_noisy = atoms


                dataset = TensorDataset(decoys['aatype'], decoys['idx'], r_noisy, atoms, mask)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=False)

                nrg_log = []
                for i,batch in enumerate(train_loader):
                    aatype, idx, r_noisy, r_clean, mask = batch

                    residue_mask = torch.ones(aatype.shape)
                    contacts = torch.zeros(aatype.shape, dtype=torch.long)
                    t_model = torch.ones(aatype.shape[0]) * t
                    chain_encoding = torch.zeros_like(residue_mask, dtype=torch.long)


                    input_feats = {
                        'aatype': torch.zeros_like(aatype).cuda() + 20 if args.mask_seq else aatype.cuda(),
                        'mask': residue_mask.cuda(),
                        'residue_idx': idx.cuda(),
                        'chain_encoding': chain_encoding.cuda(),
                        'external_contacts': contacts.cuda(),
                        'selfcond_coords': r_clean.cuda() if args.template_self_condition else torch.zeros_like(r_noisy).cuda(),
                        'r_noisy': r_noisy.cuda(),
                        'atom_mask': mask.cuda(),
                        't': t_model.cuda()
                    }


                    if args.inference_self_condition:
                        sc_output = model.compute_energy(input_feats) # independent dropping
                        input_feats['selfcond_coords'] = sc_output['pred_coords_aux'].view(r_noisy.shape)

                    
                    if args.score_norm_nrg:
                        out = model.compute_energy(input_feats)
                        out['energy'] = out['r_update_aux'].pow(2).sum(dim=(-1,-2))
                    else:
                        out = model.compute_energy(input_feats)

                    if args.ensemble_size > 0 or args.log_noise_nrg:
                        scale = torch.exp(-1/2*diffuser.marginal_b_t(t))
                        noise_energy = (r_noisy - r_clean * scale).pow(2).sum(dim=(-1,-2)) / 100 / 2 / diffuser.conditional_var(t.item())
                        nrg_log.append(torch.cat([out['energy'].unsqueeze(-1).cpu(), noise_energy.unsqueeze(-1)], dim=-1))
                    else:
                        nrg_log.append(out['energy'].unsqueeze(-1).cpu())


                time_nrg.append(torch.cat(nrg_log))
            decoy_nrg.append(torch.stack(time_nrg))
            last = decoy_nrg[-1].squeeze()
            print((last==last).sum() / len(last))
            print(spearmanr(last[last == last].numpy(), -decoys['tmscore'][last == last].numpy())[0])
        energy_dict[d] = torch.stack(decoy_nrg).permute(0,2,1,3) # Time x Decoy x Ensemble x 2

torch.save(energy_dict, args.save_dir)