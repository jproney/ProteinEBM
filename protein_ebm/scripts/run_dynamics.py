import torch
import yaml
from ml_collections import ConfigDict
from argparse import ArgumentParser
import os
import numpy as np
from datetime import datetime
import shutil

from protein_ebm.model.r3_diffuser import R3Diffuser
from protein_ebm.model.ebm import ProteinEBM
from protein_ebm.model.boltz_utils import center_random_augmentation

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--experiment_name', type=str, required=True, help='Name for the experiment')
parser.add_argument('--config', type=str, required=True, help='Path to model config file')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--scoring_config', type=str, default="", help='Path to model config file')
parser.add_argument('--scoring_checkpoint', type=str, default="", help='Path to model checkpoint')
parser.add_argument('--final_score_checkpoint', type=str, default="", help='Path to checkpoint used for scoring in final round only')
parser.add_argument('--resample_dynamics_checkpoint', type=str, default="", help='Path to checkpoint used for dynamics from round 2 onwards')
parser.add_argument('--steps', type=int, default=100, help='Number of dynamics steps for first round')
parser.add_argument('--resample_steps', type=int, default=-1, help='Number of dynamics steps for rounds 2+ (default: use --steps)')
parser.add_argument('--min_steps', type=int, default=-1, help='Minimum number of steps for ramping (default: no ramping)')
parser.add_argument('--resample_min_steps', type=int, default=-1, help='Minimum number of steps for ramping in rounds 2+ (default: use --min_steps)')
parser.add_argument('--peak_time', type=float, default=-1, help='Time at which to reach maximum steps (default: no ramping)')
parser.add_argument('--ramp_start', type=float, default=-1, help='Time to start ramping from min_steps (default: start ramping from t_max)')
parser.add_argument('--quad_ramp', action='store_true', help='Use quadratic ramp-up and linear ramp-down (default: linear both ways)')
parser.add_argument('--step_function_ramp', action='store_true', help='Switch immediately from min_steps to steps at ramp_start (no gradual ramping)')
parser.add_argument('--interval_schedule', type=int, default=-1, help='Alternate between min_steps and steps every N reverse steps (default: no alternating)')
parser.add_argument('--last_t_steps', type=int, default=-1, help='Number of dynamics steps for the last time step')
parser.add_argument('--scoring_time', type=float, default=-1, help='Time level for additional energy scoring on last time step (default: no scoring)')
parser.add_argument('--load_previous', type=str, default=None, help='Path to previous dynamics_trajectory.pt file to load and resample from')
parser.add_argument('--rescore_previous', action='store_true', help='Rescore the previous trajectory')
parser.add_argument('--resample_noise_time', type=float, default=0.2, help='Time level to noise resampled structures back to (default: 0.2)')
parser.add_argument('--num_resample_rounds', type=int, default=1, help='Number of sequential resampling rounds to perform (default: 1)')
parser.add_argument('--resample_energy_scaling', type=float, default=200.0, help='Energy scaling factor for resampling weights (default: 200, set to -1 for uniform weighting)')
parser.add_argument('--cluster_resampling_firstround', action='store_true', help='Cluster the sturctures before resampling in the first round')
parser.add_argument('--cluster_resampling_allrounds', action='store_true', help='Cluster the structures before resampling in all rounds')
parser.add_argument('--quantile_thresh_firstround', type=float, default=0.05, help='Quantile threshold for filtering structures in first round (default: 0.05)')
parser.add_argument('--cluster_quantile_thresh_firstround', type=float, default=0.25, help='Quantile threshold for filtering clusters in first round (default: 0.25)')
parser.add_argument('--quantile_thresh_resample', type=float, default=0.05, help='Quantile threshold for filtering structures in resampling rounds (default: 0.05)')
parser.add_argument('--cluster_quantile_thresh_resample', type=float, default=0.05, help='Quantile threshold for filtering clusters in resampling rounds (default: 0.05)')
parser.add_argument('--t_min', type=float, default=0.001, help='Minimum time level for reverse diffusion')
parser.add_argument('--t_max', type=float, default=1.0, help='Minimum time level for reverse diffusion')
parser.add_argument('--dt', type=float, default=0.001, help='Time step size')
parser.add_argument('--reverse_steps', type=int, default=200, help='Number of reverse steps for first round')
parser.add_argument('--resample_reverse_steps', type=int, default=-1, help='Number of reverse steps for rounds 2+ (default: use --reverse_steps)')
parser.add_argument('--temp_scaling', type=float, default=1.0, help='Temperature scaling factor for first round')
parser.add_argument('--resample_temp_scaling', type=float, default=-1, help='Temperature scaling factor for rounds 2+ (default: use --temp_scaling)')
parser.add_argument('--batch_size', type=int, default=400, help='Batch size for first round')
parser.add_argument('--resample_batch_size', type=int, default=-1, help='Batch size for rounds 2+ (default: use --batch_size)')
parser.add_argument('--total_samples', type=int, default=400, help='Total number of samples to generate in first round')
parser.add_argument('--resample_total_samples', type=int, default=-1, help='Total number of samples for rounds 2+ (default: use --total_samples)')
parser.add_argument('--metropolis', action='store_true', help='Use Metropolis-Hastings sampling')
parser.add_argument('--use_aux_score_initial', action='store_true', help='Use auxiliary score for sampling in first round only')
parser.add_argument('--use_aux_score_resample', action='store_true', help='Use auxiliary score for sampling in rounds 2+ only')
parser.add_argument('--pdb_file', type=str, required=True, help='Input PDB file')
parser.add_argument('--log_dir', type=str, default='../../eval/dynamics/', help='Base directory for saving experiment results')
args = parser.parse_args()

# Create experiment directory with timestamp  
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = f"{args.experiment_name}_{timestamp}"
log_dir = os.path.join(args.log_dir, exp_name)
os.makedirs(log_dir, exist_ok=True)

print(f"Starting dynamics experiment: {exp_name}")
print(f"Results will be saved to: {log_dir}")

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
config = ConfigDict(config)

# Save a copy of the config file in the log directory
shutil.copy(args.config, os.path.join(log_dir, 'config.yaml'))

def np_rmsd(true, pred):
  def sigmoid(z):
    return 1/(1 + np.exp(-z))
  
  def kabsch(P, Q):
    V, S, W = np.linalg.svd(np.swapaxes(P, -1,-2) @ Q, full_matrices=False)
    flip = sigmoid(-10 * np.linalg.det(V) * np.linalg.det(W))
    S = flip[:,None] * np.concatenate([S[:-1], -S[-1:]]) + (1-flip)[:,None] * S
    V = flip[:, None, None] * np.concatenate([V[:,:-1], -V[:,-1:]], axis=1) + (1-flip)[:, None, None] * V
    return V@W
  p = true - true.mean(1,keepdims=True)
  q = pred - pred.mean(1,keepdims=True)
  p = p @ kabsch(p,q)
  loss = np.sqrt(np.square(p-q).sum(-1).mean(-1) + 1e-8)
  return loss


def get_steps_for_time(t, args, step_index=None, round_idx=0):
    """Calculate number of steps for given time point with optional ramping or interval scheduling."""
    # Get round-specific step parameters
    if round_idx == 0:  # First round
        current_steps = args.steps
        current_min_steps = args.min_steps
    else:  # Subsequent rounds
        current_steps = args.resample_steps if args.resample_steps != -1 else args.steps
        current_min_steps = args.resample_min_steps if args.resample_min_steps != -1 else args.min_steps
    
    # Check for interval scheduling first
    if args.interval_schedule != -1 and step_index is not None:
        if current_min_steps == -1:
            return current_steps  # Can't alternate without min_steps defined
        # Alternate between min_steps and steps based on step index
        if (step_index+1) % args.interval_schedule == 0:
            return current_steps
        else:
            return current_min_steps
    
    # Check for step function ramping
    if args.step_function_ramp and current_min_steps != -1 and args.ramp_start != -1:
        # Determine ramp start time with clamping
        ramp_start = max(args.t_min, min(args.t_max, args.ramp_start))
        # Step function: immediate switch at ramp_start
        if t >= ramp_start:
            return current_min_steps
        else:
            return current_steps
    
    # If no ramping specified, use constant steps
    if current_min_steps == -1 or args.peak_time == -1:
        return current_steps
    
    # Clamp peak_time to be within [t_min, t_max]
    peak_time = max(args.t_min, min(args.t_max, args.peak_time))
    
    # Determine ramp start time
    if args.ramp_start == -1:
        ramp_start = args.t_max
    else:
        ramp_start = max(peak_time, min(args.t_max, args.ramp_start))
    
    if t >= ramp_start:
        # Before ramp starts: use min_steps
        return current_min_steps
    elif t >= peak_time:
        # Ramp-up phase: interpolate from min_steps (at ramp_start) to steps (at peak_time)
        if ramp_start == peak_time:
            return current_steps
        ratio = (t - peak_time) / (ramp_start - peak_time)
        # Apply quadratic ramp-up if specified
        if args.quad_ramp:
            ratio = 1-(1-ratio)**2  
        return int(current_steps + ratio * (current_min_steps - current_steps))
    else:
        # Ramp-down phase: interpolate from steps (at peak_time) to min_steps (at t_min) - always linear
        if peak_time == args.t_min:
            return current_steps
        ratio = (peak_time - t) / (peak_time - args.t_min)
        return int(current_steps + ratio * (current_min_steps - current_steps))

# Create models
diffuser = R3Diffuser(config.diffuser)
model = ProteinEBM(config.model, diffuser).cuda()

ckpt = torch.load(args.checkpoint, weights_only=False)

if "diffuse_sidechain" not in config.model:
    config.model.diffuse_sidechain = False

model.load_state_dict({k[len("model."):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model')})

if not args.scoring_config:
    args.scoring_config = args.config

if args.scoring_checkpoint:
    with open(args.scoring_config, 'r') as f:
        scoring_config = yaml.safe_load(f)
    scoring_config = ConfigDict(scoring_config)
    scoring_model = CartesianDiffuser(scoring_config.model, diffuser).cuda()
    scoring_ckpt = torch.load(args.scoring_checkpoint, weights_only=False)
    scoring_model.load_state_dict({k[len("model."):]: v for k, v in scoring_ckpt['state_dict'].items() if k.startswith('model')})
else:
    scoring_model = model

# Store checkpoint paths for later loading (to save memory)
final_score_checkpoint_path = args.final_score_checkpoint if args.final_score_checkpoint else None
resample_dynamics_checkpoint_path = args.resample_dynamics_checkpoint if args.resample_dynamics_checkpoint else None
final_checkpoint_loaded = False
resample_dynamics_checkpoint_loaded = False

def load_final_scoring_checkpoint():
    """Load final scoring checkpoint into scoring_model to save memory"""
    global final_checkpoint_loaded
    if final_score_checkpoint_path and not final_checkpoint_loaded:
        print(f"Loading final round scoring checkpoint: {final_score_checkpoint_path}")
        final_scoring_ckpt = torch.load(final_score_checkpoint_path, weights_only=False)
        scoring_model.load_state_dict({k[len("model."):]: v for k, v in final_scoring_ckpt['state_dict'].items() if k.startswith('model')})
        print("Final round scoring checkpoint loaded into scoring model")
        final_checkpoint_loaded = True
        return True
    return final_checkpoint_loaded

def load_resample_dynamics_checkpoint():
    """Load resample dynamics checkpoint into main model for rounds 2+"""
    global resample_dynamics_checkpoint_loaded
    if resample_dynamics_checkpoint_path and not resample_dynamics_checkpoint_loaded:
        print(f"Loading resample dynamics checkpoint: {resample_dynamics_checkpoint_path}")
        resample_dynamics_ckpt = torch.load(resample_dynamics_checkpoint_path, weights_only=False)
        model.load_state_dict({k[len("model."):]: v for k, v in resample_dynamics_ckpt['state_dict'].items() if k.startswith('model')})
        print("Resample dynamics checkpoint loaded into main model")
        resample_dynamics_checkpoint_loaded = True
        return True
    return resample_dynamics_checkpoint_loaded

def calculate_importance_weights(filtered_energies, energy_scaling):
    """Calculate importance weights for resampling"""
    if energy_scaling == -1:
        # Uniform weighting - all trajectories have equal probability
        importance_weights = torch.ones_like(filtered_energies) / len(filtered_energies)
        print(f"Using uniform importance weights (all trajectories weighted equally)")
    else:
        # Energy-based weighting with temperature scaling
        importance_weights = torch.nn.functional.softmax(-filtered_energies / energy_scaling)
        print(f"Using energy-based importance weights with scaling factor {energy_scaling}")
    return importance_weights

def cluster_structures_for_resampling(final_structures, min_traj_nrg, min_traj_idx, nclust=40, quantile_thresh=0.25):
    """Apply clustering to structures before resampling"""
    print("Clustering structures before resampling")
    
    # Get minimum energy structure from each trajectory
    minnrg_structs = final_structures[torch.arange(final_structures.shape[0]), min_traj_idx]
    
    # Compute pairwise RMSD matrix
    pairwise_rms = []
    for i in range(minnrg_structs.shape[0]):
        pairwise_rms.append(np_rmsd(minnrg_structs[...,1,:].numpy(), minnrg_structs[i,...,1,:].unsqueeze(0).numpy()))
    pairwise_rms = np.vstack(pairwise_rms)

    # Perform hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    np.fill_diagonal(pairwise_rms, 0)
    pairwise_rms = pairwise_rms + pairwise_rms.T

    condensed = squareform(pairwise_rms + pairwise_rms.T)
    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=nclust, criterion='maxclust')-1

    # Find best structure from each cluster
    cluster_min_nrgs = torch.zeros(nclust)
    cluster_min_structs = torch.zeros((nclust,) + final_structures.shape[1:])
    cluster_min_idxs = torch.zeros(nclust, dtype=torch.long)

    for i in range(nclust):
        minval, minidx = min_traj_nrg[labels == i].min(dim=-1)
        cluster_min_nrgs[i] = minval
        cluster_min_structs[i] = final_structures[labels == i][minidx]
        cluster_min_idxs[i] = min_traj_idx[labels == i][minidx]
        
    # Apply threshold filtering
    thresh = torch.quantile(min_traj_nrg, quantile_thresh)
    filtered_nrgs = cluster_min_nrgs[cluster_min_nrgs < thresh]
    filtered_structs = cluster_min_structs[cluster_min_nrgs < thresh]
    filtered_idxs = cluster_min_idxs[cluster_min_nrgs < thresh]

    print(f"Filtered {filtered_structs.shape[0]} clusters for resampling")
    return filtered_nrgs, filtered_structs, filtered_idxs

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from protein_ebm.data.protein_utils import residues_to_features

#Load the structure
parser = PDBParser(QUIET=True)

structure = parser.get_structure("my_structure", args.pdb_file)

chain = [c for c in structure.get_chains()][0]
atom_positions, atom_mask, aatype, residue_idx = residues_to_features([r for r in chain.get_residues() if is_aa(r)])
residue_idx = torch.tensor([r.id[1] for r in chain.get_residues() if is_aa(r)])
residue_mask = torch.ones(atom_positions.shape[0]) 
aatype = torch.tensor(aatype, dtype=torch.long)
chain_encoding = torch.zeros_like(aatype)
contacts = torch.zeros(aatype.shape, dtype=torch.long)

# Load and resample from previous run if specified
if args.load_previous:
    print(f"Loading previous trajectory from {args.load_previous}")
    prev_data = torch.load(args.load_previous, weights_only=False)

    # Find batches with different shapes (indicates multiple stages)
    diffs = [i for i, x in enumerate(prev_data['all_pos']) if x.shape[0] != prev_data['all_pos'][0].shape[0]]
    start = min(diffs) if len(diffs) > 0 else len(prev_data['all_pos'])
    
    print(f"Detected {len(diffs)} batches with different shapes")
    if len(diffs) > 0:
        print(f"Loading only first {start} batches (first stage)")
    else:
        print(f"Loading all {start} batches (single stage)")
    
    nbatch = start
    
    # Extract structures from final time level (last element in trajectory) - only first stage
    final_structures = torch.stack([x for x in prev_data['all_t0'][:start]]).transpose(1,2).reshape((nbatch * prev_data['all_t0'][0].shape[1],-1) + prev_data['all_t0'][0].shape[-3:])
    final_energies = torch.stack([x[0] for x in prev_data['all_scoring_energies'][:start]]).transpose(1,2).reshape((nbatch * prev_data['all_scoring_energies'][0][0].shape[1],-1))


    print(f"Extracted {final_structures.shape} structures from previous run")
    print(final_energies.shape)

    # Rescore with scoring model if requested
    if args.rescore_previous:
        print("Rescoring loaded conformations with scoring model...")
        
        # Rescore all final structures
        rescored_energies = []
        
        for i in range(0, len(final_structures)):
            batch_structures = final_structures[i][...,1,:]
            rescoring_batch_size = batch_structures.shape[0]
            # Prepare input features for rescoring
            rescore_input_feats = {
                'r_noisy': batch_structures.reshape([rescoring_batch_size, -1, 3]).cuda(),
                'aatype': aatype.unsqueeze(0).expand([rescoring_batch_size,-1]).cuda(),
                'mask': residue_mask.unsqueeze(0).expand([rescoring_batch_size,-1]).cuda(),
                'residue_idx': residue_idx.unsqueeze(0).expand([rescoring_batch_size,-1]).cuda(),
                't': torch.tensor([args.scoring_time]*rescoring_batch_size, dtype=torch.float).cuda(),
                'chain_encoding': torch.zeros_like(aatype).unsqueeze(0).expand([rescoring_batch_size,-1]).cuda(),
                'contacts': torch.zeros_like(aatype).unsqueeze(0).expand([rescoring_batch_size,-1]).cuda(),
                'selfcond_coords': batch_structures.reshape([rescoring_batch_size, -1, 3]).cuda(),
                'atom_mask': atom_mask.unsqueeze(0).expand([rescoring_batch_size,-1,-1]).cuda()
            }
            
            # Compute energies with scoring model
            with torch.no_grad():
                rescore_out = scoring_model.compute_energy(rescore_input_feats)
                batch_energies = rescore_out['energy'].detach().cpu()
                rescored_energies.append(batch_energies)
            
            print(f" Rescored batch {i} / {len(final_structures)}")
        
        # Combine all rescored energies
        final_energies = torch.stack(rescored_energies, dim=0)
        print(f"Rescored {final_energies.shape} structures")
        print(f"Rescored energy range: {final_energies.min():.3f} to {final_energies.max():.3f}")
        

    # Prepare resampling data for each batch to use independently
    min_traj_nrg, min_traj_idx = final_energies.min(dim=-1)
    
    if args.cluster_resampling_firstround or args.cluster_resampling_allrounds:
        filtered_nrgs, filtered_structs, filtered_idxs = cluster_structures_for_resampling(
            final_structures, min_traj_nrg, min_traj_idx, nclust=40, quantile_thresh=args.cluster_quantile_thresh_firstround)
    else:
        thresh = torch.quantile(min_traj_nrg, args.quantile_thresh_firstround)
        filtered_nrgs = min_traj_nrg[min_traj_nrg < thresh]
        filtered_structs = final_structures[min_traj_nrg < thresh]
        filtered_idxs = min_traj_idx[min_traj_nrg < thresh]
    
    importance_weights = calculate_importance_weights(filtered_nrgs, -1)
        
    print(f"Importance weights range: {importance_weights.min():.6f} to {importance_weights.max():.6f}")
    print(f"Energy range: {filtered_nrgs.min():.3f} to {filtered_nrgs.max():.3f}")
    print(f"Each batch will independently resample {args.resample_batch_size if args.resample_batch_size != -1 else args.batch_size} structures from {len(filtered_structs)} candidates")

    # Update t_max to start from resample_noise_time
    print(f"Updating t_max from {args.t_max} to {args.resample_noise_time}")
    args.t_max = args.resample_noise_time

#Build overdamped Langevin sampler 

dt = args.dt
iters = args.steps
metropolis = args.metropolis

align_steps=False
self_condition=False
aux_score=False

# Initialize storage for all rounds
all_rounds_pos = []
all_rounds_t0 = []
all_rounds_scoring_energies = []


# Main loop for multiple resampling rounds
for round_idx in range(args.num_resample_rounds):
    print(f"\n=== STARTING DYNAMICS ROUND {round_idx + 1}/{args.num_resample_rounds} ===")
    
    # Determine whether to use auxiliary score for this round
    use_aux_for_this_round = (args.use_aux_score_initial and round_idx == 0) or (args.use_aux_score_resample and round_idx >= 1)
    if use_aux_for_this_round:
        print(f"Using auxiliary score for round {round_idx + 1}")
    
    # Setup round-specific batch size and calculate number of batches
    if round_idx == 0:  # First round
        bsize = args.batch_size
        current_total_samples = args.total_samples
    else:  # Subsequent rounds
        bsize = args.resample_batch_size if args.resample_batch_size != -1 else args.batch_size
        current_total_samples = args.resample_total_samples if args.resample_total_samples != -1 else args.total_samples
    
    # Calculate number of batches from total samples and batch size
    current_num_batches = (current_total_samples + bsize - 1) // bsize  # Ceiling division
    print(f"Round {round_idx + 1}: Generating {current_total_samples} total samples in {current_num_batches} batches of size {bsize}")
    
    # Setup coordinates with appropriate batch size
    ca_coords = center_random_augmentation(atom_positions[...,1,:].unsqueeze(0).expand([bsize,-1,-1]), torch.ones([bsize, atom_positions.shape[0]])).view([bsize,-1,3])

    # Update t_max for subsequent resampling rounds
    if round_idx > 0:  # Rounds 2+ (0-indexed)
        if args.t_max != args.resample_noise_time:
            print(f"Updating t_max from {args.t_max} to {args.resample_noise_time} for resampling round {round_idx + 1}")
            args.t_max = args.resample_noise_time
    
    # Load resample dynamics checkpoint before round 2 if specified
    if round_idx == 1:  # Round 2 (0-indexed)
        dynamics_checkpoint_loaded = load_resample_dynamics_checkpoint()
        if dynamics_checkpoint_loaded:
            print("*** Using resample dynamics checkpoint from this round onwards ***")
        
        # Print step count changes for subsequent rounds
        if args.resample_steps != -1 and args.resample_steps != args.steps:
            print(f"*** Using {args.resample_steps} steps for this round onwards (was {args.steps}) ***")
        if args.resample_min_steps != -1 and args.resample_min_steps != args.min_steps:
            print(f"*** Using {args.resample_min_steps} min_steps for this round onwards (was {args.min_steps}) ***")
        if args.resample_reverse_steps != -1 and args.resample_reverse_steps != args.reverse_steps:
            print(f"*** Using {args.resample_reverse_steps} reverse_steps for this round onwards (was {args.reverse_steps}) ***")
        if args.resample_total_samples != -1 and args.resample_total_samples != args.total_samples:
            print(f"*** Using {args.resample_total_samples} total_samples for this round onwards (was {args.total_samples}) ***")
        if args.resample_temp_scaling != -1 and args.resample_temp_scaling != args.temp_scaling:
            print(f"*** Using {args.resample_temp_scaling} temp_scaling for this round onwards (was {args.temp_scaling}) ***")
    
    # Load final checkpoint before final round if specified
    if round_idx == args.num_resample_rounds - 1:
        checkpoint_loaded = load_final_scoring_checkpoint()
        if checkpoint_loaded:
            print("*** Using final round scoring checkpoint for this round ***")
    
    # Setup round-specific diffusion parameters
    if round_idx == 0:  # First round
        current_reverse_steps = args.reverse_steps
        current_temp_scaling = args.temp_scaling
    else:  # Subsequent rounds
        current_reverse_steps = args.resample_reverse_steps if args.resample_reverse_steps != -1 else args.reverse_steps
        current_temp_scaling = args.resample_temp_scaling if args.resample_temp_scaling != -1 else args.temp_scaling
    
    reverse_steps = np.linspace(args.t_min, args.t_max, current_reverse_steps)[::-1]
    dt_rev = (args.t_max - args.t_min) / current_reverse_steps
    
    # Initialize for this round
    all_pos = [] 
    all_t0 = []
    all_scoring_energies = [[] for _ in range(current_num_batches)] if args.scoring_time != -1 else None
    score = None

    with torch.no_grad():
        for b in range(current_num_batches):

            # Initialize noisy structures
            if args.load_previous and round_idx == 0:
                # Independently resample structures for this batch
                batch_resample_indices = torch.multinomial(importance_weights, bsize, replacement=True)
                batch_resampled_structures = filtered_structs[batch_resample_indices, filtered_idxs[batch_resample_indices]]
                
                print(f"Round {round_idx + 1} Batch {b}: Resampled {bsize} structures, indices: {batch_resample_indices[:5].tolist()}...")
                
                # Apply forward noising to resample_noise_time
                r_noisy_batch, _ = diffuser.forward_marginal(batch_resampled_structures[...,1,:].unsqueeze(-2).numpy(), args.resample_noise_time)
                pos_t = torch.tensor(r_noisy_batch, dtype=torch.float).view([bsize, -1, 3]).cuda()
                
                print(f"Round {round_idx + 1}: Starting from independently resampled structures at noise level {args.t_max}")
            elif round_idx > 0:
                # Independently resample structures for this batch from previous round
                batch_resample_indices = torch.multinomial(round_importance_weights, bsize, replacement=True)
                batch_resampled_structures = round_filtered_structs[batch_resample_indices, round_filtered_idxs[batch_resample_indices]]
                
                print(f"Round {round_idx + 1} Batch {b}: Resampled {bsize} structures from previous round, indices: {batch_resample_indices[:5].tolist()}...")
                
                # Apply forward noising to resample_noise_time
                r_noisy_batch, _ = diffuser.forward_marginal(batch_resampled_structures[...,1,:].unsqueeze(-2).numpy(), args.resample_noise_time)
                pos_t = torch.tensor(r_noisy_batch, dtype=torch.float).view([bsize, -1, 3]).cuda()
                
                print(f"Round {round_idx + 1}: Starting from independently resampled structures from previous round")
            else:
                # Generate new noisy structures
                if args.t_max == 1.0:
                    r_noisy = torch.randn_like(ca_coords).view([bsize, -1, 3]).cuda()
                else: 
                    r_noisy, trans_score = diffuser.forward_marginal(
                        ca_coords.reshape([-1,3]).numpy(),
                        args.t_max)
                    r_noisy = torch.as_tensor(r_noisy, dtype=torch.float, device='cuda').view([bsize, -1, 3])
                pos_t = r_noisy
                print(f"Round {round_idx + 1}: Starting from fresh random noise at level {args.t_max}")
                
            align_mask = torch.ones(pos_t.reshape([bsize,-1,3]).shape[:-1]).cuda()
            prev0 = torch.zeros(pos_t.shape).cuda()



            for step_idx, t in enumerate(reverse_steps):
                print(f"Round {round_idx + 1} Batch {b} Time {t}")
                pos_log = []
                t0_log = []


                # Calculate steps for this time point with optional ramping or interval scheduling
                if t == reverse_steps[-1] and args.last_t_steps != -1:
                    current_iters = args.last_t_steps
                else:
                    current_iters = get_steps_for_time(t, args, step_idx, round_idx)
                
                if args.interval_schedule != -1:
                    interval_info = f" (interval {step_idx // args.interval_schedule})"
                else:
                    interval_info = ""
                print(f"Using {current_iters} steps for time {t:.4f}{interval_info}")

                for i in range(current_iters + 1):

                    input_feats = {
                        'r_noisy': pos_t,
                        'aatype': aatype.unsqueeze(0).expand([bsize,-1]).cuda(),
                        'mask': residue_mask.unsqueeze(0).expand([bsize,-1]).cuda(),
                        'residue_idx': residue_idx.unsqueeze(0).expand([bsize,-1]).cuda(),
                        't': torch.tensor([t]*bsize, dtype=torch.float).cuda(),
                        'chain_encoding': torch.zeros_like(aatype).unsqueeze(0).expand([bsize,-1]).cuda(),
                        'contacts' : torch.zeros_like(aatype).unsqueeze(0).expand([bsize,-1]).cuda(),
                        'sc_ca_t' : prev0 if self_condition else torch.zeros_like(prev0),
                        'atom_mask' : atom_mask.unsqueeze(0).expand([bsize,-1,-1]).cuda()
                    }


                    if metropolis and i < current_iters:

                        if score is None:

                            if use_aux_for_this_round:
                                out = model.compute_energy(input_feats)
                                score = out['r_update_aux']
                                energy = out['energy']
                            else:
                                out = model.compute_score(input_feats)
                                score = out['trans_score']
                                energy = out['energy']

                            ca_atoms = out['pred_coords' if not use_aux_for_this_round else 'pred_coords_aux'].reshape([bsize,-1,1,3])
                            non_ca = ca_atoms +  out['sidechain_coords'].squeeze()
                            pos_allatom = torch.cat([non_ca[...,:1,:], ca_atoms, non_ca[...,1:,:]], dim=-2)

                            del out
                            


                        next_mean = pos_t + diffuser.diffusion_coef(t)**2 * score * current_temp_scaling / diffuser.config.coordinate_scaling * dt
                        next_noise = torch.sqrt(torch.tensor(2.0)) * diffuser.diffusion_coef(t) * torch.sqrt(torch.tensor(dt)) * torch.randn_like(pos_t) / diffuser.config.coordinate_scaling

                        pos_t_proposed = next_mean + next_noise

                        input_feats_proposed = {**input_feats, 'r_noisy': pos_t_proposed}
                        out_proposed = model.compute_score(input_feats_proposed)
                        score_proposed = out_proposed['trans_score']

                        ca_atoms = out_proposed['pred_coords' if not use_aux_for_this_round else 'pred_coords_aux'].reshape([bsize,-1,1,3])
                        non_ca = ca_atoms +  out_proposed['sidechain_coords'].squeeze()
                        pos_allatom_proposed = torch.cat([non_ca[...,:1,:], ca_atoms, non_ca[...,1:,:]], dim=-2)
                            

                        proposed_next_mean = pos_t_proposed + diffuser.diffusion_coef(t)**2 * score_proposed * current_temp_scaling / diffuser.config.coordinate_scaling * dt

                        energy_ratio = torch.exp(-(out_proposed['energy'] - energy) * current_temp_scaling)
                        kernel_ratio = torch.exp(-((pos_t - proposed_next_mean).pow(2).sum(dim=(-1,-2)) - next_noise.pow(2).sum(dim=(-1,-2))) / (2 * diffuser.diffusion_coef(t)**2 * dt / diffuser.config.coordinate_scaling**2) / 2)          

                        accept_ratio = energy_ratio * kernel_ratio

                        accept = torch.rand(bsize, device=pos_t.device) < accept_ratio

                        pos_t = torch.where(accept.view([-1, 1, 1]), pos_t_proposed, pos_t)
                        # Update score to match the accepted/rejected positions
                        score = torch.where(accept.view([-1, 1, 1]), score_proposed, score)
                        energy = torch.where(accept, out_proposed['energy'], energy)
                        pos_allatom = torch.where(accept.view([-1, 1, 1, 1]), pos_allatom_proposed, pos_allatom)

                        prev0_new = out_proposed['pred_coords' if not use_aux_for_this_round else 'pred_coords_aux'].view(prev0.shape)
                        prev0 = torch.where(accept.view([-1, 1, 1]), prev0_new, prev0)
                        print("Step ", i, "Acceptance ratio", accept.sum() / bsize)
                        print(energy)


                        pos_log.append(pos_t.detach().cpu())
                        t0_log.append(pos_allatom.detach().cpu())

                    else:

                        #Get model predictions
                        if use_aux_for_this_round:
                            out = model.compute_energy(input_feats)
                            score = out['r_update_aux']
                            energy = out['energy']
                        else:
                            out = model.compute_score(input_feats)
                            score = out['trans_score']
                            energy = out['energy']                

                        if i < current_iters:
                            pos_t = pos_t - (diffuser.drift_coef(pos_t, t) - diffuser.diffusion_coef(t)**2 * score * current_temp_scaling /  diffuser.config.coordinate_scaling) * dt + diffuser.diffusion_coef(t) * np.sqrt(dt) * torch.randn_like(pos_t) / diffuser.config.coordinate_scaling
                        else:
                            pos_t = pos_t - (diffuser.drift_coef(pos_t, t) - diffuser.diffusion_coef(t)**2 * score /  diffuser.config.coordinate_scaling) * dt_rev + diffuser.diffusion_coef(t) * np.sqrt(dt_rev) * torch.randn_like(pos_t) / diffuser.config.coordinate_scaling



                        pos_t, prev0_new = center_random_augmentation(pos_t.reshape([bsize,-1,3]), torch.ones([bsize, atom_positions.shape[0]], device=pos_t.device), rotate=False, second_coords=out['pred_coords' if not use_aux_for_this_round else 'pred_coords_aux'].reshape([bsize,-1,3]),return_second_coords=True)

                        pos_t = pos_t.view([bsize,-1,3])
                    
                        prev0 = prev0_new.view(pos_t.shape)

                        ca_atoms = out['pred_coords' if not use_aux_for_this_round else 'pred_coords_aux'].reshape([bsize,-1,1,3])
                        non_ca = ca_atoms +  out['sidechain_coords'].squeeze()
                        pos_allatom = torch.cat([non_ca[...,:1,:], ca_atoms, non_ca[...,1:,:]], dim=-2)
                        
                        t0_log.append(pos_allatom.detach().cpu())

                        if i < current_iters: # for the last step just run the backward and transport to the next time level
                            # now run forward ito step:
                            pos_t = pos_t + diffuser.drift_coef(pos_t, t-dt) * dt + diffuser.diffusion_coef(t-dt) * torch.sqrt(torch.tensor(dt)) * torch.randn_like(pos_t) / diffuser.config.coordinate_scaling

                        pos_log.append(pos_t.detach().cpu())

                        print(f"Step {i} Energy {out['energy']}")

                # Perform energy scoring on last time level if requested
                if args.scoring_time != -1 and t == reverse_steps[-1]:
                    print(f"Performing energy scoring at time {args.scoring_time}")
                    scoring_energies = []
                    
                    for pos_step in pos_log:
                        # Create input features for energy scoring
                        scoring_input_feats = {
                            'r_noisy': pos_step.cuda(),
                            'aatype': aatype.unsqueeze(0).expand([bsize,-1]).cuda(),
                            'mask': residue_mask.unsqueeze(0).expand([bsize,-1]).cuda(),
                            'residue_idx': residue_idx.unsqueeze(0).expand([bsize,-1]).cuda(),
                            't': torch.tensor([args.scoring_time]*bsize, dtype=torch.float).cuda(),
                            'chain_encoding': torch.zeros_like(aatype).unsqueeze(0).expand([bsize,-1]).cuda(),
                            'contacts': torch.zeros_like(aatype).unsqueeze(0).expand([bsize,-1]).cuda(),
                            'sc_ca_t': pos_step.cuda(),
                            'atom_mask': atom_mask.unsqueeze(0).expand([bsize,-1,-1]).cuda()
                        }
                        
                        # Compute energy at scoring time
                        scoring_out = scoring_model.compute_energy(scoring_input_feats)
                        scoring_energies.append(scoring_out['energy'].detach().cpu())
                    
                    all_scoring_energies[b].append(torch.stack(scoring_energies))
                    print(f"Scored {len(scoring_energies)} steps with energies: {[e.mean().item() for e in scoring_energies[:3]]}...")

            all_pos.append(torch.stack(pos_log))
            all_t0.append(torch.stack(t0_log))

    # Store results for this round
    all_rounds_pos.append(all_pos)
    all_rounds_t0.append(all_t0)
    all_rounds_scoring_energies.append(all_scoring_energies)

    print(f"=== COMPLETED DYNAMICS ROUND {round_idx + 1}/{args.num_resample_rounds} ===")

    # If not the last round, prepare resampling for next round
    if round_idx < args.num_resample_rounds - 1:
        print(f"Preparing resampling for round {round_idx + 2}")
        
        # Extract structures from final time level of this round
        round_final_structures = torch.stack([x for x in all_t0]).transpose(1,2).reshape((len(all_t0) * all_t0[0].shape[1],-1) + all_t0[0].shape[-3:])
        
        # Get energies for resampling weights
        round_final_energies = torch.stack([x[0] for x in all_scoring_energies]).transpose(1,2).reshape((len(all_scoring_energies) * all_scoring_energies[0][0].shape[1],-1))
        
        round_min_traj_nrg, round_min_traj_idx = round_final_energies.min(dim=-1)
        
        if args.cluster_resampling_allrounds:
            round_filtered_nrgs, round_filtered_structs, round_filtered_idxs = cluster_structures_for_resampling(
                round_final_structures, round_min_traj_nrg, round_min_traj_idx, nclust=40, quantile_thresh=args.cluster_quantile_thresh_resample)
        else:
            round_thresh = torch.quantile(round_min_traj_nrg, args.quantile_thresh_resample)
            round_filtered_nrgs = round_min_traj_nrg[round_min_traj_nrg < round_thresh]
            round_filtered_structs = round_final_structures[round_min_traj_nrg < round_thresh]
            round_filtered_idxs = round_min_traj_idx[round_min_traj_nrg < round_thresh]
            
        round_importance_weights = calculate_importance_weights(round_filtered_nrgs, args.resample_energy_scaling)
        
        print(f"Prepared resampling data for round {round_idx + 2}: {len(round_filtered_structs)} candidates")
        print(f"Each batch in next round will independently resample {bsize} structures")

# Stack results from all rounds
print(f"Stacking results from {args.num_resample_rounds} rounds")
all_pos = [pos for round_pos in all_rounds_pos for pos in round_pos]
all_t0 = [t0 for round_t0 in all_rounds_t0 for t0 in round_t0]
all_scoring_energies = [energies for round_energies in all_rounds_scoring_energies for energies in round_energies]

# Score native structure if scoring_time is specified
native_energy = None
if args.scoring_time != -1:
    print(f"Scoring native structure at time {args.scoring_time}")
    
    # Load final checkpoint for native scoring if not already loaded
    load_final_scoring_checkpoint()
    
    # Prepare native structure input features
    native_input_feats = {
        'r_noisy': ca_coords.reshape([bsize, -1, 3]).cuda(),
        'aatype': aatype.unsqueeze(0).expand([bsize,-1]).cuda(),
        'mask': residue_mask.unsqueeze(0).expand([bsize,-1]).cuda(),
        'residue_idx': residue_idx.unsqueeze(0).expand([bsize,-1]).cuda(),
        't': torch.tensor([args.scoring_time]*bsize, dtype=torch.float).cuda(),
        'chain_encoding': torch.zeros_like(aatype).unsqueeze(0).expand([bsize,-1]).cuda(),
        'contacts': torch.zeros_like(aatype).unsqueeze(0).expand([bsize,-1]).cuda(),
        'selfcond_coords': ca_coords.reshape([bsize, -1, 3]).cuda(),
        'atom_mask': atom_mask.unsqueeze(0).expand([bsize,-1,-1]).cuda()
    }
    
    # Compute energy for native structure
    with torch.no_grad():
        native_scoring_out = scoring_model.compute_energy(native_input_feats)
        native_energy = native_scoring_out['energy'].detach().cpu()
    
    print(f"Native structure energy at t={args.scoring_time}: {native_energy.mean().item():.3f}")
    if final_checkpoint_loaded:
        print("(Scored with final round checkpoint)")

# Save results
print(f"Saving results to {log_dir}")
save_dict = {
    'all_pos': all_pos,
    'all_t0': all_t0,
    'args': vars(args)
}

# Add scoring energies
save_dict['all_scoring_energies'] = all_scoring_energies
print(f"Saved scoring energies for {len([e for batch in all_scoring_energies for e in batch])} time levels")

# Add native structure energy if computed
if native_energy is not None:
    save_dict['native_energy'] = native_energy
    print(f"Saved native structure energy: {native_energy.mean().item():.3f}")

# Add resampling info if used
if args.load_previous:
    save_dict['resampled_from'] = args.load_previous
    save_dict['resample_noise_time'] = args.resample_noise_time
    save_dict['rescore_previous'] = args.rescore_previous

# Add number of resample rounds
save_dict['num_resample_rounds'] = args.num_resample_rounds

# Add final scoring checkpoint info if used
if args.final_score_checkpoint:
    save_dict['final_score_checkpoint'] = args.final_score_checkpoint

# Add resample dynamics checkpoint info if used
if args.resample_dynamics_checkpoint:
    save_dict['resample_dynamics_checkpoint'] = args.resample_dynamics_checkpoint

# Add resample step parameters
if args.resample_steps != -1:
    save_dict['resample_steps'] = args.resample_steps
if args.resample_min_steps != -1:
    save_dict['resample_min_steps'] = args.resample_min_steps
if args.resample_reverse_steps != -1:
    save_dict['resample_reverse_steps'] = args.resample_reverse_steps
if args.resample_total_samples != -1:
    save_dict['resample_total_samples'] = args.resample_total_samples
if args.resample_temp_scaling != -1:
    save_dict['resample_temp_scaling'] = args.resample_temp_scaling

# Add clustering resampling options
save_dict['cluster_resampling_firstround'] = args.cluster_resampling_firstround
save_dict['cluster_resampling_allrounds'] = args.cluster_resampling_allrounds

if args.load_previous:
    rescore_info = ", rescored with scoring model" if args.rescore_previous else ""
    print(f"Saved resampling metadata: loaded from {args.load_previous}, noised to {args.resample_noise_time}, {args.num_resample_rounds} rounds{rescore_info}")
else:
    print(f"Saved metadata: {args.num_resample_rounds} rounds")

torch.save(save_dict, os.path.join(log_dir, 'dynamics_trajectory.pt'))

print("Dynamics simulation complete!")