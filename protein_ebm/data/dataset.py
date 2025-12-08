import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from protein_ebm.model.boltz_utils import center_random_augmentation

class ProteinNoisedDataset(torch.utils.data.Dataset):
    def __init__(self, atom37, atom37_mask, residue_idx, aatype, residue_mask, contacts, diffuser, max_t=1.0, chain_ids=None, diffuse_sidechain=False, sidechain_drop_prob=0.5, max_len=1000, pad_to_max_threshold=None):
        """
        Args:
            atom37 (torch.Tensor): Atom37 positions [B, N, 37, 3]
            atom37_mask (torch.Tensor): Atom37 mask [B, N, 37]
            residue_idx (torch.Tensor): Residue indices [B, N]
            aatype (torch.Tensor): Amino acid types [B, N]
            residue_mask (torch.Tensor): Mask for valid positions [B, N]
            contacts (torch.Tensor): Contact information [B, N, N]
            diffuser (R3Diffuser): Diffuser instance for noising
            max_t (float): Maximum time value for rescaling (default: 1.0)
            chain_ids (torch.Tensor, optional): Chain IDs for each residue [B, N] (default: None)
            diffuse_sidechain (bool): If True, diffuse all 37 atoms instead of just backbone (default: False)
            sidechain_drop_prob (float): Probability of masking out sidechains and just looking at backbone (default: 0.5)
            max_len (int): Maximum sequence length, sequences longer than this will be cropped (default: 1000)
            pad_to_max_threshold (int, optional): If sequence length exceeds this threshold, pad to max_len instead of cropping (default: None)
            contact_drop_prob (float): Probability of returning all-zero pairwise contact map (default: 0.5)
            use_continuous_contacts (bool): If True, use continuous-valued contact maps with tanh distance mapping (default: False)
            continuous_contact_center (float): Center distance for tanh mapping in Angstroms (default: 12.0)
            continuous_contact_scale (float): Scale factor for tanh mapping (default: 2.0)
        """

        self.atom37 = atom37
        self.atom37_mask = atom37_mask
        self.residue_mask = residue_mask  # [B, N]
        self.residue_idx = residue_idx  # [B, N]
        self.aatype = aatype  # [B, N]
        self.num_proteins = len(atom37)
        self.diffuser = diffuser
        self.contacts = contacts
        self.max_t = max_t  # Store the max_t parameter
        self.chain_ids = chain_ids  # Store chain IDs
        self.diffuse_sidechain = diffuse_sidechain
        self.max_len = max_len
        self.pad_to_max_threshold = pad_to_max_threshold
        self.sidechain_drop_prob = sidechain_drop_prob # probability of masking out sidechains and just looking at backbone
        
    def __len__(self):
        return self.num_proteins
        
    def __getitem__(self, idx):
        # Sample noise level for this example, rescaled by max_t
        t = torch.rand(1) * self.max_t  # Rescale the uniform sample
        
        # Determine crop indices
        seq_len = self.residue_mask[idx].shape[0]
        if seq_len > self.max_len:
            start_idx = torch.randint(0, seq_len - self.max_len + 1, (1,)).item()
            end_idx = start_idx + self.max_len
        else:
            start_idx = 0
            end_idx = seq_len
        
        atom37_mask = self.atom37_mask[idx]

        # Extract N, CA, C coordinates from atom37
        if self.diffuse_sidechain:
            coords = self.atom37[idx] # [N, 37, 3]
            if torch.rand(1) < self.sidechain_drop_prob:
                atom37_mask[..., 3:] = 0
                coords[..., 3:, :] = 0
        else:
            coords = self.atom37[idx][:, 1, :]  # [N, 3]
        
        # Reshape to [N*3, 3] for augmentation
        N = coords.shape[0]


        flat_coords = coords.reshape([1,-1,3])  # [1, N*(1 or 37), 3]

        # Create mask - zero outside crop region, ones inside
        if self.diffuse_sidechain:
            mask = torch.zeros_like(self.atom37_mask[idx])
            mask[start_idx:end_idx] = atom37_mask
            mask = mask.reshape(1, -1)
        else:
            mask = torch.zeros(1, self.atom37[idx].shape[0])
            mask[0, start_idx:end_idx] = 1.0
        
        # Apply random augmentation 
        aug_coords, atom37_aug = center_random_augmentation(flat_coords, mask, second_coords=self.atom37[idx].reshape([1,-1,3]), return_second_coords=True)

        # Reshape back to [N, 9] for diffusion
        coords = aug_coords.reshape(N, 37*3 if self.diffuse_sidechain else 3)
        atom37_aug = atom37_aug.reshape(N, 37, 3)


        # Create noised version
        r_noisy, trans_score = self.diffuser.forward_marginal(
            coords.unsqueeze(0).numpy(),  # Add batch dimension
            t.item()
        )

        if self.diffuse_sidechain:
            r_noisy = (torch.tensor(r_noisy, dtype=torch.float).reshape(N, 37, 3) * atom37_mask[..., None]).reshape(N, 37*3)
            trans_score = (torch.tensor(trans_score, dtype=torch.float).reshape(N, 37, 3) * atom37_mask[..., None]).reshape(N, 37*3)
        else:
            r_noisy  = torch.tensor(r_noisy[0], dtype=torch.float)
            trans_score = torch.tensor(trans_score[0], dtype=torch.float)

        trans_score_scaling = self.diffuser.score_scaling(t.item())
        
        
        output = {
            'r_noisy': r_noisy,  # [N, 9 or 37*3]
            'trans_score': trans_score,  # [N, 9 or 37*3]
            'trans_score_scaling': torch.tensor(trans_score_scaling, dtype=torch.float),
            'mask': self.residue_mask[idx],
            'residue_idx': self.residue_idx[idx],
            'aatype': self.aatype[idx],
            'contacts': self.contacts[idx],
            'atom37': atom37_aug,
            'atom37_mask': atom37_mask,
            't': t,
            'chain_encoding': self.chain_ids[idx]
        }


        # Apply padding if threshold is exceeded
        if self.pad_to_max_threshold is not None and seq_len > self.pad_to_max_threshold and seq_len < self.max_len:
            #print(f"Padding protein {idx} from length {seq_len} to {self.max_len}")
            pad_amount = self.max_len - seq_len
            
            for key in output:
                tensor = output[key]
                if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 0:
     
                    # Create padding with same shape as tensor except first dimension
                    pad_shape = list(tensor.shape)
                    pad_shape[0] = pad_amount
                    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                    
                    # Concatenate original tensor with padding
                    output[key] = torch.cat([tensor, padding], dim=0)
            
            # Skip cropping logic since we've padded to max_len
            return output
        
        # Apply cropping to all relevant fields except 't'
        if start_idx != 0 or end_idx != seq_len:
            for key in output:
                if len(output[key].shape) > 0 and output[key].shape[0] == seq_len:
                    output[key] = output[key][start_idx:end_idx]
            
        return output

def stack_and_pad(batch, padding_value=0):
    batch = [b for b in batch if b is not None]
    collated_batch = {}
    for key in batch[0].keys():
        # Extract values for the current key from the batch
        values = [item[key] for item in batch]
        
        if isinstance(values[0], torch.Tensor) and len(values[0].shape) > 0:
            # Pad the sequences for tensor keys with more than 1 dimension
            collated_batch[key] = pad_sequence(values, batch_first=True, padding_value=padding_value)
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors with no padding needed
            collated_batch[key] = torch.stack(values)
        else:
            # Directly store non-tensor values (e.g., scalars or lists)
            collated_batch[key] = values


    return collated_batch
