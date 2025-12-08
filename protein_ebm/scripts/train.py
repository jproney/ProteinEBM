import yaml
from ml_collections import ConfigDict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import os
from datetime import datetime
import glob
import shutil
import random

from torch.utils.data import TensorDataset
from scipy.stats import spearmanr
from protein_ebm.model.boltz_utils import center_random_augmentation


from protein_ebm.model.r3_diffuser import R3Diffuser
from protein_ebm.data.dataset import ProteinNoisedDataset, stack_and_pad
import subprocess
from torch.serialization import add_safe_globals
from protein_ebm.model.ebm import ProteinEBM
from protein_ebm.data.data_source import (
    DataSourceManager,
    StaticDataSource,
    SubsettedDataSource,
    RotatingFileDataSource
)

add_safe_globals([ConfigDict])

class ProteinScoreMatchingTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        

        # Create SE3 diffuser
        self.diffuser = R3Diffuser(config.diffuser)
        self.model = ProteinEBM(config.model, self.diffuser)   

        # Freeze layers if specified in config
        if getattr(config.model, 'freeze_middle_layers', False):
            self.freeze_middle_layers()

    
    def freeze_middle_layers(self):
        """Freeze all layers except the first (input embeddings) and last (output projections) layers."""
        print("Freezing middle layers of ProteinEBM model...")
        
        # Keep track of frozen/unfrozen parameters
        frozen_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Define first layer components (input embeddings)
            first_layer_components = [
                'sequence_embedding',
                'noisy_coord_embedding', 
                'contact_embedding',
                'self_conditioning_embedding',
                'atom_mask_embedding'
            ]
            
            # Define last layer components (output projections)  
            last_layer_components = [
                'r_update_proj',
                'r_update_proj_aux',
                'sidechain_proj'
            ]
            
            # Define conditioning components (keep trainable)
            conditioning_components = [
                'single_conditioner',
                'rel_pos',
                'pairwise_conditioner'
            ]
            
            # Check if parameter belongs to first or last layer
            is_first_layer = any(component in name for component in first_layer_components)
            is_last_layer = any(component in name for component in last_layer_components)
            is_conditioning = any(component in name for component in conditioning_components)
            
            # Check if parameter belongs to first or last transformer block
            is_first_transformer_block = 'token_transformer.layers.0.' in name
            is_last_transformer_block = False
            
            # Determine the last transformer block index dynamically
            if 'token_transformer.layers.' in name:
                # Extract layer number from parameter name
                import re
                layer_match = re.search(r'token_transformer\.layers\.(\d+)\.', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    # Get total number of transformer layers from config
                    total_layers = self.config.model.token_transformer_depth
                    is_last_transformer_block = (layer_idx == total_layers - 1)
            
            if is_first_layer or is_last_layer or is_conditioning or is_first_transformer_block or is_last_transformer_block:
                # Keep these layers trainable
                param.requires_grad = True
                print(f"  Keeping trainable: {name} ({param.numel()} params)")
            else:
                # Freeze middle layers
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"  Freezing: {name} ({param.numel()} params)")
        
        print(f"Frozen {frozen_params}/{total_params} parameters ({100*frozen_params/total_params:.1f}%)")
        print(f"Trainable parameters: {total_params - frozen_params}")
        
    def flatten_atom37_with_mask(self, coords, mask):
        """
        Flatten atom37 coordinates by applying mask and concatenating batch elements.
        
        Args:
            coords: [B, N, 37, 3] atom coordinates
            mask: [B, N, 37] atom mask
            
        Returns:
            flattened_coords: [B, max_atoms, 3] padded coordinates
            flattened_mask: [B, max_atoms] padded mask
        """
        B, N, A, _ = coords.shape
        batch_coords = []

        # Process each batch element
        for b in range(B):
            # Get valid atoms for this batch element
            valid_mask = mask[b].flatten() > 0.5   # [N*37] - convert to bool
            valid_coords = coords[b].view(-1, 3)[valid_mask]  # [num_valid_atoms, 3]
            
            batch_coords.append(valid_coords)
        
        # Find max number of atoms across batch
        max_atoms = max(len(c) for c in batch_coords)
        
        # Pad all sequences to max length
        padded_coords = torch.zeros(B, max_atoms, 3, device=coords.device)
        padded_masks = torch.zeros(B, max_atoms, device=coords.device)
        
        for b in range(B):
            num_atoms = len(batch_coords[b])
            if num_atoms > 0:
                padded_coords[b, :num_atoms] = batch_coords[b]
                padded_masks[b, :num_atoms] =  1
        
        return padded_coords, padded_masks

    def generic_step(self, batch, batch_idx, is_training):
        # Unpack batch
        r_noisy = batch['r_noisy']
        gt_trans_scores = batch['trans_score']
        trans_score_scalings = batch['trans_score_scaling']  # [B]


        S_batch = batch['aatype']  # [B, N]
        mask_batch = batch['mask']  # [B, N]
        residue_idx_batch = batch['residue_idx']  # [B, N]
        contacts = batch['contacts']  # [B, N]
        atom37 = batch['atom37']  # [B, N, 37, 3]
        atom37_mask = batch['atom37_mask']  # [B, N, 37]
        t = batch['t']  # [B]
        chain_encoding_batch = batch['chain_encoding'] # [B, N]

        # Get loss masks (assuming all residues can be diffused for now)
        bb_mask = mask_batch
        diffuse_mask = torch.ones_like(mask_batch)  # [B, N]
        loss_mask = bb_mask * diffuse_mask
        batch_loss_mask = torch.any(bb_mask, dim=-1)  # [B]


        # Apply sequence dropping with probability
        if getattr(self.config.training, 'drop_seq_prob', 0.1) > 0:
            drop_mask = torch.rand(S_batch.shape[0], device=S_batch.device) < self.config.training.drop_seq_prob
            S_batch_dropped = torch.where(drop_mask[:, None], torch.zeros_like(S_batch) + 20, S_batch)
        else:
            S_batch_dropped = S_batch

        # Prepare input features dictionary
        input_feats = {
            'r_noisy': r_noisy,
            'aatype': torch.zeros_like(S_batch) if getattr(self.config.training, 'mask_sequence', False) else S_batch_dropped,
            'mask': mask_batch,
            'residue_idx': residue_idx_batch,
            't': t[..., 0], # should make the t dimensions consistent at some point
            'chain_encoding': chain_encoding_batch.to(torch.long),
            'external_contacts': contacts,
            'selfcond_coords': torch.zeros_like(r_noisy),  # For self-conditioning
            'atom_mask': atom37_mask
        }



        # If self-conditioning is enabled, run a no-grad forward pass to get clean predictions
        if getattr(self.config.training, 'self_conditioning', False) and random.random() > 0.5:
            with torch.no_grad():
                if getattr(self.config.training, 'drop_seq_prob', 0.1) > 0:
                    drop_mask = torch.rand(S_batch.shape[0], device=S_batch.device) < self.config.training.drop_seq_prob
                    S_batch_dropped_sc = torch.where(drop_mask[:, None], torch.zeros_like(S_batch) + 20, S_batch)

                if self.config.model.direct_score:
                    sc_output = self.model.compute_score({**input_feats, 'aatype' : S_batch_dropped_sc}) # independent dropping
                else:
                    sc_output = self.model.compute_energy({**input_feats, 'aatype' : S_batch_dropped_sc}) # independent dropping
                
                selfcond_coords = sc_output['pred_coords'].view(r_noisy.shape) if self.config.model.direct_score else sc_output['pred_coords_aux'].view(r_noisy.shape)
                input_feats['selfcond_coords'] = selfcond_coords

        # Get model predictions
        print(input_feats['r_noisy'].shape)
        output = self.model.compute_score(input_feats)
        
        # Extract scores and energy
        pred_trans_score = output['trans_score'] * diffuse_mask[..., None]
        energy = output['energy']

        B, N, _, _ = atom37.shape

        # Always compute translation score loss for logging purposes
        trans_score_mse = (gt_trans_scores - pred_trans_score)**2 * loss_mask[..., None]
        if self.config.model.diffuse_sidechain:
            trans_score_mse = (trans_score_mse.reshape(B, N, 37, 3) * atom37_mask[..., None]).reshape(B, N, 37*3) / (atom37_mask.sum(dim=-1, keepdim=True) + 1e-10)
        trans_loss = torch.sum(
            trans_score_mse / trans_score_scalings[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Auxiliary translation score loss
        aux_trans_loss = torch.tensor(0.0, device=self.device)
        if 'trans_score_aux' in output and self.config.training.get('aux_score_weight', 0.25):
            aux_trans_score_mse = (gt_trans_scores - output['trans_score_aux'])**2 * loss_mask[..., None]
            aux_trans_loss = torch.sum(
                aux_trans_score_mse / trans_score_scalings[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Compute losses
        final_loss = torch.tensor(0.0, device=self.device)
        
        # Get loss weights from config (with defaults)
        trans_score_weight = self.config.training.get('trans_score_weight', 1.0)
        aux_weight = self.config.training.get('aux_score_weight', 0.25)
        sidechain_weight = self.config.training.get('sidechain_weight', 0.0)
        
        # Translation score loss
        trans_score_loss = torch.tensor(0.0, device=self.device)
        if trans_score_weight > 0:
            trans_score_loss = trans_loss.mean()
            final_loss += trans_score_weight * trans_score_loss

        mse_loss = torch.tensor(0.0, device=self.device)

        # Sidechain loss
        sidechain_loss = torch.tensor(0.0, device=self.device)
        if sidechain_weight > 0 and 'sidechain_coords' in output:
            # Get predicted sidechain coordinates [B, N, 37, 3]
            pred_sidechain = output['sidechain_coords']
            
            # Compute MSE loss between predicted and true atom37 coordinates
            atom37_centered = atom37 - atom37[:,:, 1].unsqueeze(2) # local CA
            true_sidechain = torch.cat([atom37_centered[:,:,:1], atom37_centered[:,:,2:]], dim=-2) 

            sidechain_diff = (pred_sidechain - true_sidechain) ** 2  # [B, N, 34, 3]
            sidechain_mse = sidechain_diff.sum(dim=-1)  # [B, N, 34]
            

            # Apply atom37 mask to only compute loss on valid atoms
            sidechain_mask = torch.cat([atom37_mask[:,:,:1], atom37_mask[:,:,2:]], dim=-1)

            masked_sidechain_mse = sidechain_mse * sidechain_mask   # [B, N, 34]
            valid_atoms = sidechain_mask.sum(dim=(-1, -2))  # [B]

            sidechain_loss = masked_sidechain_mse.sum(dim=(-1, -2)) / (valid_atoms + 1e-10) 
            
            final_loss += sidechain_weight * sidechain_loss.mean()
        
        # Auxiliary losses
        aux_mse_loss = torch.tensor(0.0, device=self.device)
        
        if aux_weight > 0:
            # Auxiliary translation score loss
            if 'trans_score_aux' in output and trans_score_weight > 0:
                aux_trans_score_mse = (gt_trans_scores - output['trans_score_aux'])**2 * loss_mask[..., None]
                if self.config.model.diffuse_sidechain:
                    aux_trans_score_mse = (aux_trans_score_mse.reshape(B, N, 37, 3) * atom37_mask[..., None]).reshape(B, N, 37*3)  / (atom37_mask.sum(dim=-1, keepdim=True) + 1e-10)
                aux_trans_loss = torch.sum(
                    aux_trans_score_mse / trans_score_scalings[:, None, None]**2,
                    dim=(-1, -2)
                ) / (loss_mask.sum(dim=-1) + 1e-10)
                final_loss += aux_weight * trans_score_weight * aux_trans_loss.mean()
            
        if is_training:
            prefix = "train"
        else:
            prefix = "val"

        # Log metrics
        self.log(f'{prefix}_loss', final_loss)
        # Always log translation score loss for comparison
        self.log(f'{prefix}_trans_loss', trans_loss.mean())
        self.log(f'{prefix}_trans_score_loss', trans_score_loss)
        self.log(f'{prefix}_sidechain_loss', sidechain_loss.mean())
        self.log(f'{prefix}_aux_trans_loss', aux_trans_loss.mean())



        print(f"Step {batch_idx} Loss {final_loss}")

        return final_loss

    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, True)
    
    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, False)


    def configure_optimizers(self):
        if getattr(self.config.training, 'yo_hparams', False):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.training.learning_rate, weight_decay=0.01)
            
            # Use linear warmup + cosine annealing
            warmup_steps = 10000
            total_steps = self.trainer.estimated_stepping_batches
            
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return step / warmup_steps
                else:
                    # Cosine annealing
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.config.training.lr_scheduler.pct_start,
                div_factor=self.config.training.lr_scheduler.div_factor,
                final_div_factor=self.config.training.lr_scheduler.final_div_factor,
            )
            
        print(f"Total steps: {self.trainer.estimated_stepping_batches}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


    def compute_validation_spearman(self, decoy_data):
        """Helper function to compute validation metrics for a set of decoy data.
        
        Args:
            decoy_data: List of dictionaries containing decoy data
        """

        # Initialize metrics collection
        spearman_correlations = []
        # Process each decoy set
        for decoy_data_item in decoy_data:

            atoms = decoy_data_item['atom37'][..., 1, :]
            
            # Reshape to [B, N*3, 3] for augmentation
            B, N = atoms.shape[:2]
            
            # Create dummy mask of all ones
            dummy_mask = torch.ones((B, N), device=flat_coords.device)
            
            # Apply random augmentation
            aug_coords = center_random_augmentation(atoms, dummy_mask)
            
            r_noisy = aug_coords.reshape(B, N, 9)


            dataset = TensorDataset(
                r_noisy,
                decoy_data_item['aatype'],
                decoy_data_item['idx'],
                decoy_data_item['tmscore'],
                decoy_data_item['rmsd']
            )

            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=False
            )
            
            # Process batches
            all_energies = []
            all_tmscores = []
            
            print(f"decoy: {decoy_data_item['name']}")
            
            for batch in dataloader:
                r_noisy, aatype, residue_idx, tmscore, rmsd = batch
                
                input_feats = {
                    'r_noisy': r_noisy.cuda(),
                    'aatype': aatype.cuda(),
                    'mask': torch.ones(aatype.shape).cuda(),
                    'residue_idx': residue_idx.cuda(),
                    'chain_encoding': torch.zeros_like(residue_idx).cuda(),
                    'external_contacts': torch.ones_like(residue_idx).cuda(), # one means no external contact
                    'selfcond_coords' : r_noisy.cuda(),
                    't' : getattr(self.config.training, 'eval_time', 0.15) * torch.ones(residue_idx.shape[0]).cuda()
                }

                with torch.no_grad():
                        energies = self.model.compute_energy(input_feats)['energy']
                
                all_energies.append(energies.cpu())
                all_tmscores.append(tmscore)
            
            # Concatenate results
            energies = torch.cat(all_energies)
            tmscores = torch.cat(all_tmscores)
            
            # Compute metrics for this decoy set
            # Compute Spearman correlation
            spearman = spearmanr(energies.numpy(), -tmscores.numpy())[0]
            print(spearman)
            spearman_correlations.append(spearman)
            
        
        # Compute aggregate metrics
        avg_spearman = torch.tensor(spearman_correlations).mean()

        # Log aggregate metrics with prefix
        self.log(f'val_spearman_mean', avg_spearman)

        # Print validation metrics
        print(f"\nVal Metrics:")
        print(f"Spearman Correlation: {avg_spearman:.3f}")
        print()  # Add blank line for readability


parser = ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('--resume_run', type=str, help='Full experiment name to resume (e.g., "my_experiment_20240315_123456")')
parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained model weights (.ckpt file) to initialize from')
parser.add_argument('--md_offset', type=int, default=0, help='Offset for which MD file to load initially (default: 0)')
parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility (default: 12345)')
args = parser.parse_args()

# Load config
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
config = ConfigDict(config)


# Check required config parameters
if config.data.train_data_path is None:
    raise ValueError("Please specify a training data file in the config")
if config.data.validation_data_path is None:
    raise ValueError("Please specify a validation data file in the config")
if config.training.run_name is None and not args.resume_run:
    raise ValueError("Please specify a run_name in the training config or provide --resume_run")

# Check that pretrained_weights and resume_run are not used together
if args.pretrained_weights and args.resume_run:
    raise ValueError("Cannot use both --pretrained_weights and --resume_run. Use --resume_run to continue training or --pretrained_weights to start fresh with pretrained weights.")

def load_blocked_ids(blocked_ids_file):
    """Load blocked training IDs from file."""
    if not blocked_ids_file or not os.path.exists(blocked_ids_file):
        return set()
    
    blocked_ids = set()
    with open(blocked_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                blocked_ids.add(line)
    
    print(f"Loaded {len(blocked_ids)} blocked training IDs from {blocked_ids_file}")
    return blocked_ids

def filter_data_by_blocked_ids(data_dict, blocked_ids, data_name="data"):
    """Filter data dictionary by removing entries with blocked IDs."""
    if not blocked_ids or not data_dict:
        return data_dict, 0
    
    # Filter by 'ids' field as specified
    if 'ids' not in data_dict or data_dict['ids'] is None:
        print(f"Warning: No 'ids' field found in {data_name}, skipping filtering")
        return data_dict, 0
    
    original_count = len(data_dict['ids'])
    
    # Create mask for entries to keep (not in blocked list)
    keep_mask = []
    for i, entry_ids in enumerate(data_dict['ids']):
        should_keep = str(entry_ids) not in blocked_ids
        keep_mask.append(should_keep)
    
    # Filter all fields based on the mask
    filtered_data = {}
    for key, values in data_dict.items():
        if values is None:
            filtered_data[key] = None
        elif isinstance(values, list):
            filtered_data[key] = [values[i] for i in range(len(values)) if keep_mask[i]]
        else:
            # Handle non-list data structures (shouldn't happen in this context)
            filtered_data[key] = values
    
    filtered_count = len(filtered_data['ids']) if filtered_data['ids'] is not None else 0
    discarded_count = original_count - filtered_count
    
    print(f"Filtered {data_name}: kept {filtered_count}/{original_count} entries, discarded {discarded_count}")
    
    return filtered_data, discarded_count


# Create or load experiment name
if args.resume_run:
    exp_name = args.resume_run
    log_dir = os.path.join(config.training.log_dir, exp_name)

    # Load config from the log directory if resuming
    with open(os.path.join(log_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config)


    print(f"Resuming training from experiment: {exp_name}")
else:
    # Create unique experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.training.run_name}_{timestamp}"
    log_dir = os.path.join(config.training.log_dir, exp_name)
    # Save a copy of the config file in the log directory
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(args.config_path, os.path.join(log_dir, 'config.yaml'))

    print(f"Starting new experiment: {exp_name}")

# log git commit hash
commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
with open(os.path.join(log_dir, 'commithash.txt'), 'w') as f:
    f.write(commit_hash)

model = ProteinScoreMatchingTrainer(config)

# Load pretrained weights if specified
if args.pretrained_weights:
    print(f"Loading pretrained weights from: {args.pretrained_weights}")
    
    # Check if file exists
    if not os.path.exists(args.pretrained_weights):
        raise FileNotFoundError(f"Pretrained weights file not found: {args.pretrained_weights}")
    
    # Load the checkpoint
    checkpoint = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
    
    # Extract only the model state dict, ignoring optimizer and scheduler state
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # If it's just a state dict directly
        state_dict = checkpoint
    
    # Load the state dict into the model
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights (strict mode)")
    except RuntimeError as e:
        print(f"Failed to load with strict=True, trying strict=False: {e}")
        try:
            # Try loading with strict=False to handle minor mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys in pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in pretrained weights: {unexpected_keys}")
            print("Successfully loaded pretrained weights (non-strict mode)")
        except RuntimeError as e:
            print(f"Failed to load pretrained weights: {e}")
            raise
    
    print("Model initialized with pretrained weights. Training will start from scratch with fresh optimizer state.")

# Check for existing checkpoints
checkpoint_dir = os.path.join(log_dir, 'checkpoints')
latest_checkpoint=None
if os.path.exists(checkpoint_dir) and not args.pretrained_weights:
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if existing_checkpoints:
        latest_checkpoint=max(existing_checkpoints, key=os.path.getmtime)

# Note: When using pretrained_weights, we intentionally don't load any existing checkpoints
# to ensure we start training from scratch with fresh optimizer/scheduler state
if args.pretrained_weights and os.path.exists(checkpoint_dir):
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if existing_checkpoints:
        print(f"Found {len(existing_checkpoints)} existing checkpoints in {checkpoint_dir}, but ignoring them due to --pretrained_weights")
        print("Training will start from epoch 0 with fresh optimizer state")


# Setup logging
logger = TensorBoardLogger(
    save_dir=log_dir,
    name='tensorboard',
    default_hp_metric=False
)
logger.log_hyperparams(config)

# Setup checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(log_dir, 'checkpoints'),
    filename='{epoch:02d}-{val_loss:.2f}' if not getattr(config.training, 'compute_spearman_val', False) else '{epoch:02d}-{step:06d}-{val_spearman_mean:.2f}',
    save_top_k=1,
    monitor='val_loss' if not getattr(config.training, 'compute_spearman_val', False) else 'val_spearman_mean',
    mode='min' if not getattr(config.training, 'compute_spearman_val', False) else 'max',
    save_last=True

)

# Load blocked training IDs if specified
blocked_ids = set()
if hasattr(config.data, 'blocked_training_ids') and config.data.blocked_training_ids:
    blocked_ids = load_blocked_ids(config.data.blocked_training_ids)

# Load protein data
train_proteins = torch.load(config.data.train_data_path, weights_only=False)
val_proteins = torch.load(config.data.validation_data_path, weights_only=False)

# Filter training data by blocked IDs (don't filter validation data)
train_proteins, train_discarded = filter_data_by_blocked_ids(train_proteins, blocked_ids, "training data")
val_discarded = 0  # No validation filtering

# Get number of base training proteins for computing data source sizes
num_train_proteins = len(train_proteins['atom37'])

# Create data source manager with base training data
data_source_manager = DataSourceManager(train_proteins, seed=args.seed)

# Add AFDB data source if specified (subsetted - cycles through data across epochs)
if config.data.get("afdb_data_path", False):
    afdb_multiplier = getattr(config.data, 'afdb_multiplier', 1.0)
    afdb_proteins_per_epoch = int(num_train_proteins * afdb_multiplier)
    afdb_source = SubsettedDataSource(
        name="AFDB",
        data_path=config.data.afdb_data_path,
        proteins_per_epoch=afdb_proteins_per_epoch,
        blocked_ids=blocked_ids
    )
    data_source_manager.add_source(afdb_source)

# Add AFDB data source if specified (subsetted - cycles through data across epochs)
if config.data.get("complex_data_path", False):
    complexes_multiplier = getattr(config.data, 'complex_multiplier', 1.0)
    complexes_per_epoch = int(num_train_proteins * complexes_multiplier)
    
    # Create transform function for complexes
    def complex_transform(data):
        """Transform complex data after loading."""
        if config.training.get("zero_complex_chain_ids", False):
            print(f"Zeroing out chain_ids for complexes dataset and offsetting chain 1 residue indices")
            
            # Process each protein in the dataset
            new_idx_list = []
            for i, (chain_ids, idx) in enumerate(zip(data['chain_ids'], data['idx'])):
                # Find residues in chain 0 and chain 1
                chain0_mask = chain_ids == 0
                chain1_mask = chain_ids == 1
                
                # If we have chain 1 residues, offset their indices
                if chain1_mask.any():
                    # Get max index in chain 0
                    if chain0_mask.any():
                        max_chain0_idx = idx[chain0_mask].max()
                        min_chain1_idx = idx[chain1_mask].min()
                        
                        # Calculate offset so min(chain1) = max(chain0) + 64
                        offset = max_chain0_idx + 64 - min_chain1_idx
                        
                        # Create new index tensor
                        new_idx = idx.clone()
                        new_idx[chain1_mask] = idx[chain1_mask] + offset
                        new_idx_list.append(new_idx)
                    else:
                        # No chain 0, just keep original
                        new_idx_list.append(idx)
                else:
                    # No chain 1, keep original
                    new_idx_list.append(idx)
            
            # Update data with new indices and zeroed chain_ids
            data['idx'] = new_idx_list
            data['chain_ids'] = [torch.zeros_like(x) for x in data['chain_ids']]
        return data
    
    complex_source = SubsettedDataSource(
        name="Complexes",
        data_path=config.data.complex_data_path,
        proteins_per_epoch=complexes_per_epoch,
        blocked_ids=blocked_ids,
        transform_fn=complex_transform
    )
    data_source_manager.add_source(complex_source)

# Add MD data source if specified (rotating - loads different file each super-epoch)
if config.data.get("md_data_prefix", False):
    # Find all files matching the MD data prefix
    md_data_files = sorted(glob.glob(config.data.md_data_prefix + "*"))
    if md_data_files:
        print(f"Found {len(md_data_files)} MD data files matching prefix: {config.data.md_data_prefix}")
        
        md_multiplier = getattr(config.data, 'md_multiplier', 1.0)
        md_proteins_per_epoch = int(num_train_proteins * md_multiplier)
        
        def md_transform(data):
            """Transform MD data after loading."""
            if config.data.get("rescale_md_coords", False):
                data['atom37'] = [x*10 for x in data['atom37']]
                print(f"Rescaled MD coordinates by factor of 10")
            data['contacts'] = [x.to(torch.long) for x in data['contacts']]
            data['chain_ids'] = [torch.zeros_like(x) for x in data['idx']]
            return data
        
        md_source = RotatingFileDataSource(
            name="MD",
            data_files=md_data_files,
            proteins_per_epoch=md_proteins_per_epoch,
            blocked_ids=blocked_ids,
            transform_fn=md_transform,
            initial_offset=args.md_offset
        )
        data_source_manager.add_source(md_source)
    else:
        print(f"WARNING: No MD data files found matching prefix: {config.data.md_data_prefix}")

# Backwards compatibility: single MD file path
elif config.data.get("md_data_path", False):
    md_multiplier = getattr(config.data, 'md_multiplier', 1.0)
    md_proteins_per_epoch = int(num_train_proteins * md_multiplier)
    
    def md_transform_single(data):
        """Transform MD data after loading."""
        if config.data.get("rescale_md_coords", False):
            data['atom37'] = [x*10 for x in data['atom37']]
            print(f"Rescaled MD coordinates by factor of 10")
        data['contacts'] = [x.to(torch.long) for x in data['contacts']]
        data['chain_ids'] = [torch.zeros_like(x) for x in data['idx']]
        return data
    
    md_source = SubsettedDataSource(
        name="MD",
        data_path=config.data.md_data_path,
        proteins_per_epoch=md_proteins_per_epoch,
        blocked_ids=blocked_ids,
        transform_fn=md_transform_single
    )
    data_source_manager.add_source(md_source)

# Initialize all data sources
data_source_manager.initialize()

# Create a simplified data module using the DataSourceManager
class ProteinEBMDataModule(pl.LightningDataModule):
    def __init__(self, config, data_source_manager, val_proteins, diffuser, val_decoy_data=None):
        super().__init__()
        self.config = config
        self.data_source_manager = data_source_manager
        self.val_proteins = val_proteins
        self.diffuser = diffuser
        self.current_epoch = 0
        self.val_decoy_data = val_decoy_data
    def setup(self, stage=None):
        """Called at the beginning of each epoch."""
        if stage == 'fit' or stage is None:
            # Get combined data for current epoch from manager
            combined_proteins = self.data_source_manager.get_combined_data(self.current_epoch)
                
            # Create training dataset with combined proteins
            self.train_dataset = ProteinNoisedDataset(
                combined_proteins['atom37'],
                combined_proteins['atom37_mask'],
                combined_proteins['idx'],
                combined_proteins['aatype'],
                [torch.ones(r.shape[0]) for r in combined_proteins['atom37']], 
                combined_proteins['contacts'],
                self.diffuser,
                max_t=self.config.training.max_t,
                chain_ids=combined_proteins['chain_ids'],
                diffuse_sidechain=getattr(self.config.model, 'diffuse_sidechain', False),
                max_len=getattr(self.config.training, 'max_len', 1000),
                pad_to_max_threshold=getattr(self.config.training, 'pad_to_max_threshold', None)
            )
            
            # Create validation dataset (static)
            self.val_dataset = ProteinNoisedDataset(
                self.val_proteins['atom37'],
                self.val_proteins['atom37_mask'],
                self.val_proteins['idx'],
                self.val_proteins['aatype'],
                [torch.ones(r.shape[:-2]) for r in self.val_proteins['atom37']], 
                [torch.ones_like(c) for c in self.val_proteins['contacts']], # one means no external contact
                self.diffuser,
                max_t=self.config.training.max_t,
                chain_ids=self.val_proteins['chain_ids'],
                diffuse_sidechain=getattr(self.config.model, 'diffuse_sidechain', False),
                max_len=getattr(self.config.training, 'max_len', 1000),
                pad_to_max_threshold=getattr(self.config.training, 'pad_to_max_threshold', None),
            )
                
            print(f"Training dataset size: {len(self.train_dataset)}")
    
    def train_dataloader(self):
        print("Reloading Train Dataloader!")   
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.get('num_workers', 4),
            collate_fn=stack_and_pad
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.get('num_workers', 4),
            collate_fn=stack_and_pad
        )



if config.training.get('compute_spearman_val', False):

    if getattr(config.data, 'val_decoy_ids_file', None) is not None:
        with open(config.data.val_decoy_ids_file, 'r') as f:
            decoy_files = [os.path.join(config.data.decoy_dir, f"{line.strip()}.pt") for line in f if line.strip()]
    else:
        decoy_files = glob.glob(os.path.join(config.data.decoy_dir, "*.pt"))

    decoy_data = []
    for decoy_file in decoy_files:
        data = torch.load(decoy_file, weights_only=False)
        data['name'] = os.path.basename(decoy_file).split('.')[0]
        decoy_data.append(data)
else:
    decoy_data = None


# Determine start epoch
if args.resume_run:
    ckpt = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0

# Create the simplified data module
data_module = ProteinEBMDataModule(
    config,
    data_source_manager,
    val_proteins,
    model.diffuser,
    decoy_data
)
data_module.current_epoch = start_epoch

# Print summary of blocked ID filtering (from training data only - sources handle their own)
if blocked_ids:
    print(f"\n=== BLOCKED ID FILTERING SUMMARY ===")
    print(f"Training data entries discarded: {train_discarded}")
    print(f"Validation data: 0 (not filtered)")
    print(f"Additional source filtering reported above by each DataSource")
    print(f"=====================================\n")

# Create a callback to handle epoch changes
class EpochChangeCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Update epoch and reload data for the data module
        trainer.datamodule.current_epoch = trainer.current_epoch
        print(f"Starting epoch {trainer.datamodule.current_epoch} - reloading datasets")
        trainer.datamodule.setup('fit')

class ValidationCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(pl_module.config.training, 'compute_spearman_val', False):
            pl_module.compute_validation_spearman(trainer.datamodule.val_decoy_data)

# Create trainer
trainer = pl.Trainer(
    max_epochs=config.training.max_epochs,
    logger=logger,
    callbacks=[checkpoint_callback, EpochChangeCallback(), ValidationCallback()],
    devices=config.training.gpus,
    accelerator='gpu' if config.training.gpus > 0 else None,
    num_nodes=getattr(config.training, 'num_nodes', 1),
    gradient_clip_val=10.0,  # Add gradient clipping
    accumulate_grad_batches=config.training.grad_accum,
    strategy="ddp_find_unused_parameters_true" if config.training.gpus > 1 else "auto",
    val_check_interval=getattr(config.training, 'val_check_interval', 1.0),  # Validate multiple times per epoch if specified
    precision="bf16" if getattr(config.training, 'use_bf16', False) else "32",  # Enable full bf16 precision if specified
    reload_dataloaders_every_n_epochs=1,
    use_distributed_sampler=True,
    limit_train_batches=getattr(config.training, 'limit_train_batches', None)
)

# Train model
trainer.fit(model, data_module, ckpt_path=latest_checkpoint)
