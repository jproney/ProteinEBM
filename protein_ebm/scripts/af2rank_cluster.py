#!/usr/bin/env python3
"""
AF2 Rank Cluster Script

Loads dynamics results from run_dynamics.py, clusters them, and uses low-energy
cluster centers as templates for AlphaFold2 prediction. The template sequence
is set to all gap tokens while the target sequence is the correct sequence.

Usage:
    python af2rank_cluster.py --trajectory dynamics_trajectory.pt \\
        --pdb_file protein.pdb --data_dir /path/to/af2/params \\
        --output_dir ./af2_reranked --num_clusters 40 --energy_quantile 0.25
"""

import argparse
import sys
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

# Add alphafold to path
sys.path.append('/home/ubuntu/alphafold')

from alphafold.model import config
from alphafold.model import data
from alphafold.model import model as af_model
from alphafold.model import modules
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
import haiku as hk
import jax
import jax.numpy as jnp


# Compile regex patterns for TM-score parsing
tm_re = re.compile(r'TM-score[\s]*=[\s]*(\d.\d+)')
ref_len_re = re.compile(r'Length=[\s]*(\d+)[\s]*\(by which all scores are normalized\)')
common_re = re.compile(r'Number of residues in common=[\s]*(\d+)')
super_re = re.compile(r'\(":" denotes the residue pairs of distance < 5\.0 Angstrom\)\\n([A-Z\-]+)\\n[" ", :]+\\n([A-Z\-]+)\\n')


def np_rmsd(true, pred):
    """Compute RMSD between two sets of coordinates with Kabsch alignment."""
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    def kabsch(P, Q):
        V, S, W = np.linalg.svd(np.swapaxes(P, -1,-2) @ Q, full_matrices=False)
        flip = sigmoid(-10 * np.linalg.det(V) * np.linalg.det(W))
        S_flip = flip[:,None] * np.concatenate([S[:,:-1], -S[:,-1:]], axis=1) + (1-flip)[:,None] * S
        V_flip = flip[:, None, None] * np.concatenate([V[:,:-1], -V[:,-1:]], axis=1) + (1-flip)[:, None, None] * V
        return V_flip@W
    
    p = true - true.mean(1,keepdims=True)
    q = pred - pred.mean(1,keepdims=True)
    p = p @ kabsch(p,q)
    loss = np.sqrt(np.square(p-q).sum(-1).mean(-1) + 1e-8)
    return loss


def cluster_structures(structures, energies, nclust=40, quantile_thresh=0.25):
    """
    Cluster structures and return low-energy cluster centers.
    
    Args:
        structures: Tensor of shape [N, T, 3, 3] (CA coordinates)
        energies: Tensor of shape [N] (energies per structure)
        nclust: Number of clusters
        quantile_thresh: Quantile threshold for filtering
    
    Returns:
        filtered_structures: Low-energy cluster centers
        filtered_energies: Corresponding energies
    """
    print(f"Clustering {structures.shape[0]} structures into {nclust} clusters...")
    
    # Compute pairwise RMSD matrix on CA coordinates
    ca_coords = structures[:, :, 1, :]  # [N, T, 3] - CA atoms
    pairwise_rms = []
    for i in range(ca_coords.shape[0]):
        pairwise_rms.append(np_rmsd(ca_coords.numpy(), ca_coords[i].unsqueeze(0).numpy()))
    pairwise_rms = np.vstack(pairwise_rms)
    
    # Perform hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    np.fill_diagonal(pairwise_rms, 0)
    pairwise_rms = pairwise_rms + pairwise_rms.T
    
    condensed = squareform(pairwise_rms)
    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=nclust, criterion='maxclust') - 1
    
    print(f"Found {len(np.unique(labels))} clusters")
    
    # Find best structure from each cluster
    cluster_structures = []
    cluster_energies = []
    
    for i in range(nclust):
        cluster_mask = labels == i
        if not cluster_mask.any():
            continue
        
        cluster_nrgs = energies[cluster_mask]
        cluster_structs = structures[cluster_mask]
        
        # Get minimum energy structure from this cluster
        min_idx = cluster_nrgs.argmin()
        cluster_structures.append(cluster_structs[min_idx])
        cluster_energies.append(cluster_nrgs[min_idx])
    
    cluster_structures = torch.stack(cluster_structures)
    cluster_energies = torch.tensor(cluster_energies)
    
    # Filter by quantile threshold
    thresh = torch.quantile(energies, quantile_thresh)
    filtered_mask = cluster_energies < thresh
    
    filtered_structures = cluster_structures[filtered_mask]
    filtered_energies = cluster_energies[filtered_mask]
    
    print(f"Filtered to {filtered_structures.shape[0]} low-energy cluster centers")
    print(f"Energy range: {filtered_energies.min():.3f} to {filtered_energies.max():.3f}")
    
    return filtered_structures, filtered_energies


def load_dynamics_results(trajectory_path):
    """Load dynamics trajectory and extract final structures, energies, and PDB path."""
    print(f"Loading dynamics trajectory from {trajectory_path}...")
    saved = torch.load(trajectory_path)
    
    # Extract PDB file path from saved args
    if hasattr(saved['args'], 'pdb_file'):
        pdb_file = saved['args'].pdb_file
    elif isinstance(saved['args'], dict) and 'pdb_file' in saved['args']:
        pdb_file = saved['args']['pdb_file']
    else:
        raise ValueError(f"Could not find pdb_file in trajectory args: {saved['args']}")
    
    print(f"  PDB file from trajectory: {pdb_file}")
    
    # Handle multi-stage trajectories (take last stage)
    all_pos = saved['all_pos']
    all_t0 = saved['all_t0']
    all_nrgs = saved['all_scoring_energies']
    
    # Detect multi-stage and use last stage
    diffs = [i for i,x in enumerate(all_pos) if x.shape[0] != all_pos[0].shape[0]]
    if len(diffs) > 0:
        start = max(diffs)
        print(f"Multi-stage trajectory detected, using stage {start} onwards")
        all_pos = all_pos[start:]
        all_t0 = all_t0[start:]
        all_nrgs = all_nrgs[start:]
    
    # Extract final structures and energies
    # all_t0 shape: list of [num_batches, timesteps, num_samples, residues, atoms, 3]
    # We want the final time level (t0) for each sample
    nbatch = len(all_t0)
    final_structures = torch.stack([x for x in all_t0]).transpose(1,2).reshape(
        (nbatch * all_t0[0].shape[1], -1) + all_t0[0].shape[-3:]
    )
    
    # Extract energies (use first scoring energy if multiple)
    final_energies = torch.stack([x[0] for x in all_nrgs]).transpose(1,2).reshape(
        (nbatch * all_nrgs[0][0].shape[1], -1)
    )
    
    # Get minimum energy for each trajectory
    min_energies, min_indices = final_energies.min(dim=-1)
    
    # Get the structures at minimum energy
    min_structures = final_structures[torch.arange(final_structures.shape[0]), min_indices]
    
    print(f"Loaded {min_structures.shape[0]} structures")
    print(f"Energy range: {min_energies.min():.3f} to {min_energies.max():.3f}")
    
    return min_structures, min_energies, pdb_file


def extend(a, b, c, L, A, D):
    """
    Add C-Beta to glycine residues.
    
    Args:
        a, b, c: 3D coordinates
        L: Length
        A: Angle
        D: Dihedral
    
    Returns:
        4th coordinate
    """
    def normalize(x):
        return x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-8)
    
    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    
    return c + sum([m_i * d_i for m_i, d_i in zip(m, d)])


def compute_tmscore(pdb_pred, pdb_native, tm_exec, test_len=True):
    """
    Compute TM Scores between two PDBs and parse outputs
    
    Args:
        pdb_pred: Path to the predicted PDB
        pdb_native: Path to the native PDB
        tm_exec: Path to TMscore executable
        test_len: Run asserts that the input and output should have the same length
    
    Returns:
        TM-score (float)
    """
    cmd = [tm_exec, pdb_pred, pdb_native, '-seq']
    tmscore_output = str(subprocess.check_output(cmd))
    
    try:
        tm_out = float(tm_re.search(tmscore_output).group(1))
        reflen = int(ref_len_re.search(tmscore_output).group(1))
        common = int(common_re.search(tmscore_output).group(1))
        
        seq1 = super_re.search(tmscore_output).group(1)
        seq2 = super_re.search(tmscore_output).group(2)
    except Exception as e:
        print("Failed on: " + " ".join(cmd))
        raise e
    
    if test_len:
        assert reflen == common, cmd
        assert seq1 == seq2, cmd
        assert len(seq1) == reflen, cmd
    
    return tm_out


def extract_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Get first chain
    chain = next(structure.get_chains())
    
    sequence = []
    for residue in chain:
        if is_aa(residue):
            resname = residue.get_resname()
            restype_3to1 = {v: k for k, v in residue_constants.restype_1to3.items()}
            if resname in restype_3to1:
                sequence.append(restype_3to1[resname])
            else:
                sequence.append('X')  # Unknown residue
    
    return ''.join(sequence)


def create_template_features(template_coords, target_sequence, num_templates):
    """
    Create template features with gap tokens for all template sequences.
    Masks sidechains and adds projected C-beta atoms.
    
    Args:
        template_coords: np.array of shape [num_templates, seq_length, 37, 3]
        target_sequence: Target protein sequence (string) - used to identify glycines
        num_templates: Number of templates
    
    Returns:
        Dictionary of template features
    """
    seq_length = len(target_sequence)
    
    # Template sequence is all gap tokens
    decoy_seq = '-' * seq_length
    
    # Template aatype: use HHBLITS_AA_TO_ID for gap tokens
    template_aatype = residue_constants.sequence_to_onehot(
        decoy_seq, 
        residue_constants.HHBLITS_AA_TO_ID
    )[None].astype(np.int32)
    template_aatype = np.repeat(template_aatype, num_templates, axis=0)
    
    # Template all atom positions
    template_all_atom_positions = template_coords.astype(np.float32)
    
    # Template all atom mask: auto-detect which atoms are present (non-zero)
    atom_present = np.any(template_coords != 0, axis=-1)  # [num_templates, seq_length, 37]
    template_all_atom_masks = atom_present.astype(np.float32)
    
    # Template domain names
    template_domain_names = np.array(["None"] * num_templates, dtype=object)
    
    return {
        'template_aatype': template_aatype,
        'template_all_atom_positions': template_all_atom_positions,
        'template_all_atom_masks': template_all_atom_masks,
        'template_domain_names': template_domain_names,
    }


def convert_coords_to_atom37(coords, target_sequence):
    """
    Convert minimal backbone coords to atom37 format with projected C-beta.
    Masks sidechains and adds CB to all residues.
    
    Args:
        coords: Tensor of shape [num_templates, seq_length, 3 or 4, 3] (N, CA, C) or (N, CA, C, O)
        target_sequence: Target protein sequence (string) - used to identify glycines
    
    Returns:
        atom37_coords: np.array of shape [num_templates, seq_length, 37, 3]
    """
    num_templates, seq_length, num_atoms = coords.shape[:3]
    atom37 = np.zeros((num_templates, seq_length, 37, 3), dtype=np.float32)
    
    # Convert to numpy
    coords_np = coords.numpy()
    
    # Fill in backbone atoms (N=0, CA=1, C=2)
    atom37[:, :, 0, :] = coords_np[:, :, 0, :]  # N
    atom37[:, :, 1, :] = coords_np[:, :, 1, :]  # CA
    atom37[:, :, 2, :] = coords_np[:, :, 2, :]  # C
    atom37[:, :, 3, :] = coords_np[:, :, 3, :]  # O
    
    # Project CB atoms for each template and residue
    for t in range(num_templates):
        for i in range(seq_length):
            n = atom37[t, i, 0, :]
            ca = atom37[t, i, 1, :]
            c = atom37[t, i, 2, :]
            
            # Project CB atom (atom index 4)
            # CB projection parameters from the reference: extend(c, n, ca, 1.522, 1.927, -2.143)
            cb = extend(c, n, ca, 1.522, 1.927, -2.143)
            atom37[t, i, 4, :] = cb
    
    return atom37


def run_af2_with_templates(sequence, template_coords, model_runner):
    """
    Run AlphaFold2 with template structures.
    
    Args:
        sequence: Target protein sequence (string)
        template_coords: Template coordinates [num_templates, seq_length, 3, 3]
        model_params: AlphaFold2 model parameters
        model_config: AlphaFold2 model configuration
    
    Returns:
        prediction_result: Dictionary with model outputs
    """
    seq_length = len(sequence)
    num_templates = template_coords.shape[0]
    
    print(f"Running AlphaFold2 with {num_templates} templates...")
    
    # Create sequence features
    sequence_features = pipeline.make_sequence_features(
        sequence=sequence,
        description='target',
        num_res=seq_length,
    )
    
    # Create minimal MSA (just the query sequence)
    msa_str = f">target\n{sequence}\n"
    msa_object = pipeline.parsers.parse_a3m(msa_str)
    msa_features = pipeline.make_msa_features([msa_object])
    
    # Create template features with gap tokens
    atom37_coords = convert_coords_to_atom37(template_coords, sequence)
    template_features = create_template_features(atom37_coords, sequence, num_templates)
    
    # Combine all features
    feature_dict = {**sequence_features, **msa_features, **template_features}
    
    processed_features = model_runner.process_features(
        feature_dict, random_seed=0
    )


    # Run prediction
    result = model_runner.predict(processed_features, 0)
    
    
    return result, processed_features


def process_single_trajectory(trajectory_path, output_dir, args, model_runner):
    """Process a single trajectory file."""
    print(f"\n{'='*80}")
    print(f"Processing trajectory: {trajectory_path}")
    print(f"{'='*80}\n")
    
    # Load dynamics results (includes PDB file path)
    structures, energies, pdb_file = load_dynamics_results(trajectory_path)
    
    # Get template structures and energies
    if args.use_min_energy_only:
        # Skip clustering, just use the single minimum energy structure
        print("Using minimum energy structure only (no clustering)")
        min_idx = torch.argmin(energies)
        template_structures = [structures[min_idx]]  # List of single structures
        template_energies = [energies[min_idx]]
        
        print(f"\nMinimum energy structure:")
        print(f"  Energy = {template_energies[0]:.3f}")
    else:
        # Cluster structures and use ALL filtered cluster centers
        cluster_centers, cluster_energies = cluster_structures(
            structures, energies, 
            nclust=args.num_clusters,
            quantile_thresh=args.energy_quantile
        )
        
        # Sort by energy and use ALL cluster centers
        sorted_indices = torch.argsort(cluster_energies)
        template_structures = [cluster_centers[i] for i in sorted_indices]
        template_energies = [cluster_energies[i] for i in sorted_indices]
        
        print(f"\nUsing all {len(template_structures)} cluster centers as templates:")
        for i, energy in enumerate(template_energies[:5]):  # Show first 5
            print(f"  Template {i+1}: Energy = {energy:.3f}")
        if len(template_energies) > 5:
            print(f"  ... and {len(template_energies) - 5} more")
    
    # Extract target sequence from PDB
    sequence = extract_sequence_from_pdb(pdb_file)
    print(f"\nTarget sequence length: {len(sequence)}")
    
    # Run AlphaFold2 with each template separately
    all_results = []
    for template_idx, (template_structure, template_energy) in enumerate(zip(template_structures, template_energies)):
        print(f"\n--- Processing template {template_idx + 1}/{len(template_structures)} (Energy: {template_energy:.3f}) ---")
        
        # Run AF2 with single template (add batch dimension)
        single_template = template_structure.unsqueeze(0)  # [1, T, 3, 3]
        result, processed_features = run_af2_with_templates(
            sequence, single_template, model_runner
        )
        
        # Extract results
        plddt = result['plddt']
        mean_plddt = float(np.mean(plddt))
        ptm = float(result.get('ptm', -1.0))
        
        print(f"  Mean pLDDT: {mean_plddt:.2f}")
        print(f"  PTM score: {ptm:.3f}")
        
        # Save predicted structure
        unrelaxed_protein = protein.from_prediction(
            features=processed_features,
            result=result,
            b_factors=np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1),
            remove_leading_feature_dimension=True,
        )
        
        output_pdb = output_dir / f'predicted_structure_template{template_idx:03d}.pdb'
        with open(output_pdb, 'w') as f:
            f.write(protein.to_pdb(unrelaxed_protein))
        
        # Compute TM-score if executable provided
        tmscore = None
        if args.tm_exec:
            tmscore = compute_tmscore(str(output_pdb), pdb_file, args.tm_exec, test_len=False)
            print(f"  TM-score: {tmscore:.4f}")

        
        all_results.append({
            'template_idx': template_idx,
            'template_energy': float(template_energy),
            'mean_plddt': mean_plddt,
            'ptm': ptm,
            'tmscore': tmscore,
            'output_pdb': str(output_pdb)
        })
    
    # Save summary CSV
    import csv
    csv_path = output_dir / 'predictions_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['template_idx', 'template_energy', 'mean_plddt', 'ptm', 'tmscore', 'output_pdb']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            writer.writerow(res)
    
    print(f"\nSaved predictions summary to {csv_path}")
    
    # Find best prediction by TM-score or pLDDT
    if any(r['tmscore'] is not None for r in all_results):
        best_result = max([r for r in all_results if r['tmscore'] is not None], key=lambda x: x['tmscore'])
        best_metric = 'TM-score'
        best_value = best_result['tmscore']
    else:
        best_result = max(all_results, key=lambda x: x['mean_plddt'])
        best_metric = 'pLDDT'
        best_value = best_result['mean_plddt']
    
    print(f"\nBest prediction (by {best_metric}):")
    print(f"  Template {best_result['template_idx'] + 1}, Energy: {best_result['template_energy']:.3f}")
    print(f"  {best_metric}: {best_value:.4f}")
    
    return {
        'trajectory': trajectory_path,
        'pdb_file': pdb_file,
        'num_templates': len(template_structures),
        'best_template_idx': best_result['template_idx'],
        'best_template_energy': best_result['template_energy'],
        'best_mean_plddt': best_result['mean_plddt'],
        'best_ptm': best_result['ptm'],
        'best_tmscore': best_result['tmscore'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cluster dynamics results and rank with AlphaFold2 templates'
    )
    
    # Input - either trajectory list file or individual trajectories
    parser.add_argument('--trajectory_list', type=str,
                       help='Path to file containing list of trajectory paths (one per line)')
    parser.add_argument('--trajectories', type=str, nargs='+',
                       help='Path(s) to dynamics_trajectory.pt file(s)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to AlphaFold2 parameters directory')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for predictions (default: same as trajectory directory)')
    
    # Clustering parameters
    parser.add_argument('--num_clusters', type=int, default=40,
                       help='Number of clusters for hierarchical clustering')
    parser.add_argument('--energy_quantile', type=float, default=0.25,
                       help='Energy quantile threshold for filtering clusters')
    parser.add_argument('--use_min_energy_only', action='store_true',
                       help='Skip clustering and use only the single minimum energy structure as template')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='model_1_ptm',
                       choices=['model_1_ptm', 'model_2_ptm'],
                       help='AlphaFold2 model to use')
    parser.add_argument('--tm_exec', type=str, default='/home/ubuntu/Tmscore/TMscore',
                       help='Path to TMscore executable')
    
    args = parser.parse_args()
    
    # Get list of trajectories
    trajectory_files = []
    if args.trajectory_list:
        print(f"Reading trajectory list from {args.trajectory_list}")
        with open(args.trajectory_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    trajectory_files.append(line)
    
    if args.trajectories:
        trajectory_files.extend(args.trajectories)
    
    if not trajectory_files:
        parser.error("Must provide either --trajectory_list or --trajectories")
    
    print(f"Found {len(trajectory_files)} trajectory file(s) to process")
    
    # Load AlphaFold2 model (once for all trajectories)
    print(f"\nLoading AlphaFold2 model {args.model_name}...")
    model_config = config.model_config(args.model_name)
    model_config.data.eval.num_ensemble = 1
    model_config.model.embeddings_and_evoformer.num_msa = 1
    model_config.model.embeddings_and_evoformer.num_extra_msa = 1
    model_config.data.common.max_extra_msa = 1
    model_config.data.eval.max_msa_clusters = 1
    model_config.data.common.reduce_msa_clusters_by_max_templates = False
    model_config.data.common.use_templates = True
    model_config.model.num_recycle = 3
    
    model_params = data.get_model_haiku_params(
        model_name=args.model_name,
        data_dir=args.data_dir
    )

    model_runner = af_model.RunModel(model_config, model_params)

    
    # Process each trajectory
    results = []
    for i, trajectory_file in enumerate(trajectory_files):
        print(f"\n\nProcessing trajectory {i+1}/{len(trajectory_files)}")
        
        
        # If no output_dir specified, use the trajectory's parent directory
        if args.output_dir is None:
            trajectory_output_dir = Path(trajectory_file).parent
        else:
            trajectory_output_dir = Path(args.output_dir)
        

        trajectory_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = process_single_trajectory(
                trajectory_file, 
                trajectory_output_dir, 
                args, 
                model_runner
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {trajectory_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed {len(results)}/{len(trajectory_files)} trajectories\n")
    
    if results:
        # Check if any results have TM-scores
        has_tmscore = any(r.get('best_tmscore') is not None for r in results)
        
        if has_tmscore:
            print(f"{'Trajectory':<30} {'Best TM':>10} {'Best pLDDT':>11} {'Best PTM':>9} {'#Templates':>11} {'Best Tmpl#':>11}")
            print(f"{'-'*30} {'-'*10} {'-'*11} {'-'*9} {'-'*11} {'-'*11}")
            for r in results:
                traj_name = Path(r['trajectory']).stem[:28]
                tmscore_str = f"{r['best_tmscore']:>10.4f}" if r.get('best_tmscore') is not None else f"{'N/A':>10}"
                print(f"{traj_name:<30} {tmscore_str} {r['best_mean_plddt']:>11.2f} {r['best_ptm']:>9.3f} {r['num_templates']:>11} {r['best_template_idx']+1:>11}")
        else:
            print(f"{'Trajectory':<35} {'Best pLDDT':>11} {'Best PTM':>9} {'#Templates':>11} {'Best Tmpl#':>11}")
            print(f"{'-'*35} {'-'*11} {'-'*9} {'-'*11} {'-'*11}")
            for r in results:
                traj_name = Path(r['trajectory']).stem[:33]
                print(f"{traj_name:<35} {r['best_mean_plddt']:>11.2f} {r['best_ptm']:>9.3f} {r['num_templates']:>11} {r['best_template_idx']+1:>11}")
    
    if args.output_dir is None:
        print(f"\nResults saved to trajectory directories")
    else:
        print(f"\nAll results saved to {args.output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()

