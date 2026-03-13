# ProteinEBM

This repository houses the open source code for ProteinEBM! ProteinEBM uses denoising score matching to learn an approximation of the energy landscape $$E_\theta(x,s)$$ of a protein structure given its sequence. This has numerious applications, including structure prediction, simulation, and stability prediction. This is the second version of ProteinEBM, released on 3/13/2026 and corresponding to the updated preprint: https://www.biorxiv.org/content/10.64898/2025.12.09.693073v2. 
For the original version of the code, see the `preprint_v1` branch.

[![Diffusion Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/jproney/ProteinEBM/blob/main/proteinebm_diffusion.ipynb)

## Installation

To install in a new conda environment do:

```
conda create --name protebm python=3.11
conda activate protebm

git clone https://github.com/jproney/ProteinEBM-public
cd ProteinEBM-public
pip install .
```

After this, you should be able to do `from protein_ebm.model.ebm import ProteinEBM` 

## Downloading Evaluation Data and Model Weights

In order to download the evaluation data for decoy ranking, stability ranking, conformational baising, and fast-folder simulation, do:

```
cd download_scripts
./download_eval_data.sh
```

To parse the decoy dataset into the tensor data format used by ProteinEBM, do:

```
cd protein_ebm/data/data_scripts
python build_decoy_sets.py
```

This will populate `eval_data/decoys` with a set of tensors that can be used for fast inference with ProteinEBM, as is done in `protein_ebm/scripts/score_decoys.py`

To download the model weights do:

```
cd download_scripts
./download_weights.sh
```

The `download_weights.sh` script will download the parameters for 4 models, all with the same underlying architecture and pretraining regiment. These models are as follows:
- `model_1_frozen_1m_md.pt`: Model finetuned on 1M frames of MD with middle layers frozen. Used for sampling the `t>0.1` noise levels during fast-folder Langevin annealing. **Use this model if you need to do reverse diffusion from a high noise level.**
- `model_2_frozen_1m_md.pt`: Replicate of model 1.
- `model_3_unfrozen_1m_md.pt`: Same pretraining checkpoint as model 1 finetuned on 1M frames of MD with all weights unfrozen.
- `model_4_unfrozen_3m_md.pt`: Same pretraining checkpoint as model 1 fintuned on 3M frames of MD with all weights unfrozen. Used for inital sampling stage in structure prediction protocol.
- `model_5_expert_pretrained.pt`: Expert model pretrained with only `t<.15`. Used in refinement stage in structure prediction protocol.
- `model_6_expert_frozen_1m_md.pt`: **This is ProteinEBM-x from the paper**. Derived from Model 5 finetuned on 1M frames of MD with middle layers frozen. Used for decoy ranking, stability prediction, conformational biasing, direct folding simulation, and sampling the `t<0.1` noise levels during fast-folder Langevin annealing. **Use this checkpoint for scoring applications.**

All of these checkpoints are stored at https://huggingface.co/jproney/ProteinEBM/tree/main.

## Loading the Model and Scoring Structures

Here is a minimal example of loading a model, parsing a protein from a PDB, and computing an energy:

```
import torch
import yaml
from ml_collections import ConfigDict

from protein_ebm.model.r3_diffuser import R3Diffuser
from protein_ebm.data.protein_utils import residues_to_features, plot_protein_frame
from protein_ebm.model.ebm import ProteinEBM
from protein_ebm.model.boltz_utils import center_random_augmentation
import numpy as np

with open("protein_ebm/config/expert_model_config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
config = ConfigDict(config)

# Create models
diffuser = R3Diffuser(config.diffuser)
model = ProteinEBM(config.model, diffuser).cuda()


# Load checkpoint
ckpt = torch.load("weights/model_6_expert_frozen_1m_md.pt", weights_only=False)
model.load_state_dict({k[len("model."):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model')})

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

pdb_path = 'eval_data/confbiasing/3a7r.pdb'
parser = PDBParser(QUIET=True)

structure = parser.get_structure("my_structure", pdb_path) 
chain = list(structure.get_chains())[0]
atom_positions, atom_mask, aatype, residue_idx = residues_to_features([r for r in chain.get_residues() if is_aa(r)])
nres = atom_positions.shape[0]
ca_coords = center_random_augmentation(atom_positions[...,1,:].unsqueeze(0), torch.ones([1, nres])).view([1,nres,3])

t=0.05
input_feats = {
    'r_noisy': ca_coords.cuda(), # coordinates
    'aatype': aatype.unsqueeze(0).cuda(), # amino acid types
    'mask': torch.ones([1, nres]).cuda(), # amino acid mask (for multiple different-length proteins)
    'residue_idx': residue_idx.unsqueeze(0).cuda(), # residue indices
    't': torch.tensor([t], dtype=torch.float).cuda(), # diffusion time (set to 0.1 for scoring)
    'selfcond_coords' : ca_coords.cuda() # optional self-conditioning coordinate channel
}

with torch.no_grad():
    out = model.compute_energy(input_feats)

print(out['energy'])
```

**:exclamation::exclamation:Important Note:exclamation::exclamation::** Make sure that your coordinates are cenetered before passing them to ProteinEBM! This is what the line 
`ca_coords = center_random_augmentation(atom_positions[...,1,:].unsqueeze(0), torch.ones([1, nres])).view([1,nres,3])
` 
does! Do not skip this.

## Notebooks

This repository contains several notebooks for using the model and reproducing the results in the paper. There are as follows:

- `notebooks/ddg_prediction.ipynb`: Runs ddG predictions with ProteinEBM on the Megascale data set. Reproduces results of Figure 3 in the paper.
- `notebooks/diffusion_inference.ipynb`: Runs reverse diffusion with the ProteinEBM to predict structures of arbitrary input proteins and visualize the results.
- `notebooks/confbiasing.ipynb`: Uses ProteinEBM to compute the relative energies of open and closed states of LplA mutants, reproducing Figure S6.
- `notebooks/rank_decoys.ipynb`: Analyzes the results of decoy ranking with ProteinEBM. See the Decoy Ranking section for detailed usage instructions.

## Decoy Ranking

To score the Rosetta decoys, make sure you have run `download_eval_data.sh`, `download_weights.sh`, and `build_decoy_sets.py` as mentioned above. Then, do:

```
cd protein_ebm/scripts
python score_decoys.py ../config/expert_model_config.yaml ../../weights/model_6_expert_frozen_1m_md.pt ../data/data_lists/validation_decoys.txt ../../eval_data/model_6_val_decoy_scores.pt --decoy_dir ../../eval_data/decoys/ --n_samples 16 --t_max 0.15 --bsize 64 --template_self_condition
```


After running this code you can analyze the results in `notebooks/rank_decoys.ipynb`

## Running Dynamics

To run a fast-folder simuation using a combination of the base model and the expert model, you can do:

```
python run_dynamics.py --pdb_file ../../eval_data/fastfolders/experimental_structures/chignolin_cln025.pdb \
--config ../config/base_model_config.yaml --checkpoint ../../weights/model_1_frozen_1m_md.pt \
--lo_time_config ../config/expert_model_config.yaml --lo_time_checkpoint ../../weights/model_6_expert_frozen_1m_md.pt \
--lo_time_threshold 0.1 --steps 100 --min_steps 0 \
--t_min .01 --t_max 1.0 --ramp_start 0.5 --step_function_ramp --dt .001 --reverse_steps 200 --total_samples 400 \
--temp_scaling 0.85714 --temp_scaling_after_lo_time 1.0 --scoring_time 0.05 --use_aux_score_initial --batch_size 400 --log_dir ../../dynamics/ \
--experiment_name chignolin_dynamics
```

To run a two-stage structure prediction run, you can do:

```
cd protein_ebm/scripts
python run_dynamics.py --pdb_file ../../eval_data/decoys/natives/2chf.pdb --config ../config/base_model_config.yaml \
 --checkpoint ../../weights/model_4_unfrozen_3m_md.pt --resample_dynamics_config ../config/expert_model_config.yaml\
 --resample_dynamics_checkpoint ../../weights/model_5_expert_pretrained.pt \
 --steps 100 --resample_steps 10 --min_steps 0 --ramp_start 0.5 --step_function_ramp --dt .001 --reverse_steps 200 \
 --resample_reverse_steps 20 --t_min .01 --t_max 1.0 --total_samples 400 --resample_total_samples 800 --temp_scaling 0.85714 \
 --resample_temp_scaling 1.0 --resample_noise_time 0.1 --scoring_time 0.05 --num_resample_rounds 4 --use_aux_score_initial \
 --batch_size 100 --resample_batch_size 10 --log_dir ../../dynamics/ --experiment_name 2chf_structure_prediction
```

To run a direct folding simumation, you can do:

```
python run_dynamics.py --experiment_name nug2_folding_sim --pdb_file ../../eval_data/fastfolders/experimental_structures/proteing_1mi0.pdb \
--config ../config/expert_model_config.yaml --checkpoint ../../weights/model_6_expert_frozen_1m_md.pt \
--dt .001 --steps 10000 --reverse_steps 1 --t_min 0.05 --t_max 0.05 --scoring_time 0.05 --save_stride 10 \
 --total_samples 400 --batch_size 400 --use_aux_score_initial --start_unfolded --num_resample_rounds 1 --log_dir ../../dynamics/ 
```

After running these commands, you can analyzes the results in `notebooks/analyze_dynamics.ipynb`

## Training

To download the pretraining data form ProteinEBM, you can do

```
cd download_scripts
./download_data.sh
```
The training data are stored at: https://zenodo.org/records/17871696.

This will populate `weights/training_data` with three zipped data files. You well need about 10GB of space for the compressed data files, and about 50GB to uncompress them. After downloading and uncompressing the data files, you can run:

```
cd protein_ebm/scripts
python train.py ../config/expert_model_config.yaml
```

This will pretrain the model and save the results in `training_logs/`. Before running this, make sure to update `protein_ebm/config/expert_model_config.yaml` to correctly reflect your GPU setup.

## Raw Results from Paper

Raw results from the paper figures for decoy ranking, ddG prediction, and structure prediction, as well as the model weights, can be found at https://huggingface.co/jproney/ProteinEBM/tree/main.
