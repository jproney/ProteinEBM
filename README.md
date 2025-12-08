# ProteinEBM

This repository houses the open source code for ProteinEBM! ProteinEBM uses denoising score matching to learn an approximation of the energy landscape $$E_\theta(x,s)$$ of a protein structure given its sequence. This has numerious applications, including structure prediction, simulation, and stability prediction. For more information, see our preprint.

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

Note that this will take 40GB of space. Feel free to download a subset of these files.

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
- `model_1_frozen_1m_md.pt`: Finetuned on 1M frames of MD with middle layers frozen. Used for decoy ranking, ddG ranking, fast folders, and conformational biasing in an ensemble with model 2.
- `model_2_frozen_1m_md.pt`: Replicate of model 1, used with model 1 as an ensemble for the above tasks
- `model_3_unfrozen_1m_md.pt`: Finetuned on 1M frames of MD with all weights unfrozen. Used for refinement stage in structure prediction protocol.
- `model_4_unfrozen_3m_md.pt`: Fintuned on 3M frames of MD with all weights unfrozen. Used for inital sampling stage in structure prediction protocol. 

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

with open("protein_ebm/config/base_pretrain.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
config = ConfigDict(config)

# Create models
diffuser = R3Diffuser(config.diffuser)
model = ProteinEBM(config.model, diffuser).cuda()


# Load checkpoint
ckpt = torch.load("weights/model_1_frozen_1m_md.pt", weights_only=False)
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

t=0.1
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

## Notebooks

This repository contains several notebooks for using the model and reproducing the results in the paper. There are as follows:

- `notebooks/ddg_prediction.ipynb`: Runs ddG predictions with ProteinEBM on the Megascale data set. Reproduces results of Figure 3 in the paper.
- `notebooks/diffusion_inference.ipynb`: Runs reverse diffusion with the ProteinEBM to predict structures of arbitrary input proteins and visualize the results.
- `notebooks/confbiasing.ipynb`: Uses ProteinEBM to compute the relative energies of open and closed states of LplA mutants, reproducing Figure S3.
- `notebooks/rank_decoys.ipynb`: Analyzes the results of decoy ranking with ProteinEBM. See the Decoy Ranking section for detailed usage instructions.
- `notebooks/analyze_dynamics.ipynb`: Analyzes the results of sampling trajectories ProteinEBM. See the Running Dynamics section for detailed usage instructions.


## Decoy Ranking

To score the Rosetta decoys, make sure you have run `download_eval_data.sh`, `download_weights.sh`, and `build_decoy_sets.py` as mentioned above. Then, do:

```
cd protein_ebm/scripts
python score_decoys.py ../config/base_pretrain.yaml ../../weights/model_1_frozen_1m_md.pt ../data/data_lists/validation_decoys.txt ../../eval_data/model_1_val_decoy_scores.pt --n_samples 21 --bsize 64 --template_self_condition
```

After running this code you can analyze the results in `notebooks/rank_decoys.ipynb`

## Running Dynamics

To run a fast-folder simuation, you can do:


```
python run_dynamics.py --pdb_file ../../eval_data/fastfolders/experimental_structures/chignolin_cln025.pdb \
--config ../config/base_pretrain.yaml --checkpoint ../../weights/model_1_frozen_1m_md.pt --steps 100 --min_steps 0 \
--t_min .01 --t_max 1.0 --ramp_start 0.5 --step_function_ramp --dt .001 --reverse_steps 200 --total_samples 400 \
--temp_scaling 0.85714 --scoring_time 0.1 --use_aux_score_initial --batch_size 400 --log_dir ../../dynamics/ \
--experiment_name chignolin_dynamics
```

To run a two-stage structure prediction run, you can do:

```
cd protein_ebm/scripts
python run_dynamics.py --pdb_file ../../eval_data/decoys/natives/2chf.pdb --config ../config/base_pretrain.yaml \
 --checkpoint ../../weights/model_4_unfrozen_3m_md.pt --resample_dynamics_checkpoint ../../weights/model_3_unfrozen_1m_md.pt \
 --steps 100 --resample_steps 10 --min_steps 0 --ramp_start 0.5 --step_function_ramp --dt .001 --reverse_steps 200 \
 --resample_reverse_steps 20 --t_min .01 --t_max 1.0 --total_samples 400 --resample_total_samples 800 --temp_scaling 0.85714 \
 --resample_temp_scaling 1.0 --resample_noise_time 0.1 --scoring_time 0.1 --num_resample_rounds 4 --use_aux_score_initial \
 --batch_size 134 --resample_batch_size 10 --log_dir ../../dynamics/ --experiment_name 2chf_structure_prediction
```

After running these commands, you can analyzes the results in `notebooks/analyze_dynamics.ipynb`

## Training

To download the pretraining data form ProteinEBM, you can do

```
cd download_scripts
./download_data.sh
```

This will populate `weights/training_data` with three zipped data files. You well need about 10GB of space for the compressed data files, and about 50GB to uncompress them. After downloading and uncompressing the data files, you can run:

```
cd protein_ebm/scripts
python train.py ../config/base_pretrain.yaml
```

This will pretrain the model and save the results in `training_logs/`. Before running this, make sure to update `protein_ebm/config/base_pretrain.yaml` to correctly reflect your GPU setup.

For info on finetuning models and generating new datasets, check back soon
