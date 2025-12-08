import torch
import torch.nn as nn
import py3Dmol
import numpy as np
import math
from typing import List, Tuple, Optional


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

def residues_to_features(residues, strict=True):
    """Convert a list of residues from BioPython PDB parser to atom37"""
    # Initialize lists 
    atom_positions = []
    atom_mask = []
    aatype = []
    residue_idx = []
    # Iterate over the residues to extract backbone atom coordinates and atom37 data
    for residue in residues:
        ca_atom = residue["CA"].coord if "CA" in residue else None

        # Skip residue if any of the backbone atoms are missing
        if strict and ca_atom is None:
            assert False, "Unmodeled Residue!"
        elif ca_atom is None:
            continue


        # Get residue type
        res_shortname = restype_3to1.get(residue.get_resname(), 'X')
        restype_idx = restype_order.get(res_shortname, restype_num)
        aatype.append(restype_idx)

        # Initialize atom37 position and mask arrays
        pos = np.zeros((atom_type_num, 3))
        mask = np.zeros((atom_type_num,))

        # Fill in atom positions and mask
        for atom in residue:
            if atom.name not in atom_types:
                continue
            pos[atom_order[atom.name]] = atom.coord
            mask[atom_order[atom.name]] = 1.

        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_idx.append(residue.id[1])

    atom_positions = torch.tensor(atom_positions, dtype=torch.float32)
    atom_mask = torch.tensor(atom_mask, dtype=torch.float32)
    aatype = torch.tensor(aatype, dtype=torch.long)
    residue_idx = torch.tensor(residue_idx, dtype=torch.long)


    return atom_positions, atom_mask, aatype, residue_idx



def plot_protein_frame(atom_pos, atom_mask, cartoon=False, show_sidechains=True, show_mainchains=True):
    """Visualizes a single protein frame using py3Dmol.
    
    Args:
        atom_pos: position tensor [N, 37, 3] for a single frame
        atom_mask: [N, 37] binary mask for which atoms are present
    
    Returns:
        view: py3Dmol viewer object
    """


    pdb_str = "MODEL     1\n"
    atom_num = 1
    for res_idx, res_pos in enumerate(atom_pos):
        for atom_idx, p in enumerate(res_pos):
            if atom_mask[res_idx, atom_idx]: # Only include non-masked atoms
                x, y, z = p.detach().numpy()
                atom_name = atom_types[atom_idx]
                # PDB format: columns must be strictly adhered to
                # ATOM   2714 HD23 LEU B 185     -24.885  -4.187  -1.551  1.00 99.99           H
                pdb_str += (f"ATOM  {atom_num:5d}  {atom_name:<3s} ALA A{res_idx+1:4d}"
                           f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:1s}\n")
                atom_num += 1
    pdb_str += "ENDMDL\n"
    pdb_str += "END\n"
    
    # Create viewer with explicit size and path to 3Dmol.js
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
    
    # Add model to viewer
    view.addModel(pdb_str, "pdb")
    if cartoon:
        view.setStyle({'cartoon': {'color': 'spectrum'}})

    if show_sidechains:
        BB = ['C','O','N']
        view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                            {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                            {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                            {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
    if show_mainchains:
        BB = ['C','O','N','CA']
        view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

    view.setBackgroundColor('white')  # Make background white for better visibility
    view.zoomTo()
    
    return view

def get_animation_pdb(atom_pos, atom_mask):
    """
    Args:
        atom_pos: List of atom37 position tensors [..., N, 37, 3] for each diffusion step
        atom_mask: [N, 37] binary mask for which atoms are present
    
    Returns:
        pdb_str: Multi-model PDB string for animation
    """
    
    # Create multi-model PDB string
    pdb_str = ""
    for model_num, pos in enumerate(atom_pos, start=1):
        pdb_str += f"MODEL     {model_num}\n"
        atom_num = 1
        for res_idx, res_pos in enumerate(pos):
            for atom_idx, p in enumerate(res_pos):
                if atom_mask[res_idx, atom_idx]:
                    x, y, z = p.detach().numpy()
                    atom_name = atom_types[atom_idx]
                    pdb_str += (f"ATOM  {atom_num:5d}  {atom_name:<3s} ALA A{res_idx+1:4d}"
                               f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:1s}\n")
                    atom_num += 1
        pdb_str += "ENDMDL\n"
    pdb_str += "END\n"

    return pdb_str

def generate_random_backbone_coords(sequence: str, ramachandran_file: str = None, fixed_phi: float = None, fixed_psi: float = None, uniform_sampling: bool = False) -> Tuple[torch.Tensor, float]:
    """Generate N, CA, C coordinates for a protein sequence with correct bond lengths 
    and random backbone dihedral angles sampled from Ramachandran distributions.
    
    Args:
        sequence: Protein sequence as a string (e.g., "ACDEFGHIKLMNPQRSTVWY")
        ramachandran_file: Path to file containing Ramachandran distributions in the specified format
        fixed_phi: Optional fixed phi angle in degrees for all residues. If None, samples from Ramachandran
        fixed_psi: Optional fixed psi angle in degrees for all residues. If None, samples from Ramachandran
        uniform_sampling: If True, sample uniformly within ±15 degrees of the center. If False, use Gaussian sampling (default)
        
    Returns:
        coords: Tensor of shape [N, 3, 3] containing N, CA, C coordinates for each residue
        log_prob: Log probability density of the sampled backbone angles
    """
    import math
    import os
    
    # Standard bond lengths and angles (in Angstroms and radians)
    N_CA_LENGTH = 1.458
    CA_C_LENGTH = 1.525
    C_N_LENGTH = 1.329
    
    # Bond angles (in radians)
    N_CA_C_ANGLE = math.radians(111.0)  # N-CA-C angle
    CA_C_N_ANGLE = math.radians(116.2)  # CA-C-N angle  
    C_N_CA_ANGLE = math.radians(121.7)  # C-N-CA angle
    
    # Load Ramachandran distributions from file
    ramachandran_data = {}
    
    if ramachandran_file and os.path.exists(ramachandran_file):
        with open(ramachandran_file, 'r') as f:
            lines = f.readlines()
            
        # Parse the file
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Res'):  # Skip header and empty lines
                continue
                
            parts = line.split()
            if len(parts) >= 7:
                res = parts[0]
                direction = parts[1]  # 'left', 'right', or 'ALL'
                neighbor = parts[2]
                phi = int(parts[3])
                psi = int(parts[4])
                prob = float(parts[5])
                
                # Create key for this residue-neighbor combination
                key = f"{res}_{direction}_{neighbor}"
                
                if key not in ramachandran_data:
                    ramachandran_data[key] = {'phi': [], 'psi': [], 'prob': []}
                
                ramachandran_data[key]['phi'].append(phi)
                ramachandran_data[key]['psi'].append(psi)
                ramachandran_data[key]['prob'].append(prob)
    
    def sample_ramachandran_angles(aa, prev_aa=None, next_aa=None):
        """Sample phi, psi angles from Ramachandran distribution for given amino acid and neighbors.
        
        Returns:
            phi: phi angle in radians
            psi: psi angle in radians
            log_prob: log probability density of this sample
        """
        
        # Use fixed angles if provided
        if fixed_phi is not None and fixed_psi is not None:
            # For fixed angles, return 0 log probability (probability = 1)
            return math.radians(fixed_phi), math.radians(fixed_psi), 0.0
        
        # Fallback single letter to three letter conversion
        aa_1to3 = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        
        # Try to find the most specific distribution available
        possible_keys = []
        
        if ramachandran_data:
            # Convert single letter to three letter code
            aa_3letter = residue_constants.restype_1to3.get(aa, 'UNK')

            
            # Try with specific neighbors first
            if prev_aa:
                prev_3letter = residue_constants.restype_1to3.get(prev_aa, 'UNK')
                possible_keys.append(f"{aa_3letter}_left_{prev_3letter}")
            
            if next_aa:
                next_3letter = residue_constants.restype_1to3.get(next_aa, 'UNK')
                possible_keys.append(f"{aa_3letter}_right_{next_3letter}")
            
            # Try with ALL neighbors
            possible_keys.extend([
                f"{aa_3letter}_left_ALL",
                f"{aa_3letter}_right_ALL",
                f"{aa_3letter}_ALL_ALL"
            ])
            
            # Find the first available key
            for key in possible_keys:
                if key in ramachandran_data:
                    data = ramachandran_data[key]
                    
                    # Sample from the distribution
                    probs = np.array(data['prob'])
                    probs = probs / np.sum(probs)  # Normalize
                    
                    # Sample an index
                    idx = np.random.choice(len(probs), p=probs)
                    
                    phi_center = data['phi'][idx]
                    psi_center = data['psi'][idx]
                    
                    if uniform_sampling:
                        # Sample uniformly within ±15 degrees
                        phi_offset = np.random.uniform(-18, 18)
                        psi_offset = np.random.uniform(-18, 18)
                        phi = math.radians(phi_center + phi_offset)
                        psi = math.radians(psi_center + psi_offset)
                        
                        # Log prob = log(P(center)) + log(1/30) + log(1/30) for uniform in ±15
                        center_log_prob = np.log(probs[idx] + 1e-10)
                        uniform_log_prob = -np.log(42.0) * 2  # Two uniform distributions of width 30
                        log_prob = center_log_prob + uniform_log_prob
                    else:
                        # Original Gaussian sampling (kept for backward compatibility)
                        phi = math.radians(phi_center)
                        psi = math.radians(psi_center)
                        log_prob = np.log(probs[idx] + 1e-10)
                    
                    return phi, psi, log_prob
        
        # Fallback to hardcoded distributions if file not available or residue not found
        ramachandran_params = {
            'G': [(-60, 30), (60, 30)],  # Glycine - more flexible
            'P': [(-60, 30)],  # Proline - restricted
            'default': [(-60, 30), (-120, 120), (60, 30)]  # General case
        }
        
        if aa == 'G':
            params = ramachandran_params['G']
        elif aa == 'P':
            params = ramachandran_params['P']
        else:
            params = ramachandran_params['default']
        
        # Sample from one of the allowed regions
        region = np.random.choice(len(params))
        phi_mean, psi_mean = params[region]
        
        if uniform_sampling:
            # Sample uniformly within ±15 degrees
            phi_sample = np.random.uniform(-15, 15)
            psi_sample = np.random.uniform(-15, 15)
            phi = math.radians(phi_mean + phi_sample)
            psi = math.radians(psi_mean + psi_sample)
            
            # Compute log probability: uniform over regions + uniform noise
            # Log prob = log(1/n_regions) + log(1/30) + log(1/30)
            region_log_prob = -np.log(len(params))
            uniform_log_prob = -np.log(30.0) * 2  # Two uniform distributions of width 30
            log_prob = region_log_prob + uniform_log_prob
        else:
            # Add some noise around the mean (std = 30 degrees)
            phi_sample = np.random.normal(0, 30)
            psi_sample = np.random.normal(0, 30)
            phi = math.radians(phi_mean + phi_sample)
            psi = math.radians(psi_mean + psi_sample)
            
            # Compute log probability: uniform over regions + Gaussian noise
            # Log prob = log(1/n_regions) + log(Normal(phi_sample; 0, 30)) + log(Normal(psi_sample; 0, 30))
            region_log_prob = -np.log(len(params))
            gaussian_log_prob_phi = -0.5 * np.log(2 * np.pi * 30**2) - 0.5 * (phi_sample / 30)**2
            gaussian_log_prob_psi = -0.5 * np.log(2 * np.pi * 30**2) - 0.5 * (psi_sample / 30)**2
            log_prob = region_log_prob + gaussian_log_prob_phi + gaussian_log_prob_psi
        
        return phi, psi, log_prob
    
    def rotation_matrix(axis, angle):
        """Create rotation matrix around axis by angle using Rodrigues formula."""
        axis = axis / torch.norm(axis)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        # Rodrigues rotation formula
        K = torch.tensor([[0, -axis[2], axis[1]], 
                         [axis[2], 0, -axis[0]], 
                         [-axis[1], axis[0], 0]], dtype=torch.float32)
        
        R = torch.eye(3) + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
        return R
    
    def place_atom(p1, p2, p3, bond_length, bond_angle, dihedral_angle):
        """Place fourth atom given three previous atoms, bond length, angle, and dihedral."""
        # Vectors
        v1 = p2 - p1  # vector from p1 to p2
        v2 = p3 - p2  # vector from p2 to p3
        
        # Normalize
        v1 = v1 / torch.norm(v1)
        v2 = v2 / torch.norm(v2)
        
        # Normal to the plane containing p1, p2, p3
        n = torch.cross(v1, v2, dim=-1)
        n = n / torch.norm(n)
        
        # Vector perpendicular to v2 in the plane
        perp = torch.cross(v2, n, dim=-1)
        perp = perp / torch.norm(perp, dim=-1, keepdim=True)
        
        # Direction of new bond (from p3)
        # Start with direction opposite to v2, then apply bond angle and dihedral
        base_dir = -v2
        
        # Apply bond angle rotation around perp axis
        rotated_dir = base_dir * math.cos(bond_angle) + perp * math.sin(bond_angle)
        
        #Apply dihedral rotation around v2 axis
        final_dir = (rotated_dir * math.cos(dihedral_angle) + 
                    torch.cross(v2, rotated_dir) * math.sin(dihedral_angle) +
                    v2 * torch.dot(v2, rotated_dir) * (1 - math.cos(dihedral_angle)))
        
        # Place new atom
        return p3 + final_dir * bond_length
    
    n_residues = len(sequence)
    coords = torch.zeros(n_residues, 3, 3, dtype=torch.float32)  # [N, 3, 3] for N, CA, C
    total_log_prob = 0.0  # Track cumulative log probability
    
    # Initialize first residue in a standard configuration
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])  # N
    coords[0, 1] = torch.tensor([N_CA_LENGTH, 0.0, 0.0])  # CA  
    coords[0, 2] = torch.tensor([N_CA_LENGTH + CA_C_LENGTH * math.cos(math.pi - N_CA_C_ANGLE), 
                                CA_C_LENGTH * math.sin(math.pi - N_CA_C_ANGLE), 0.0])  # C
    
    # If only one residue, return
    if n_residues == 1:
        return coords, total_log_prob
    
    # For second residue, place it with a simple omega angle (trans peptide bond)
    omega = 0
    
    # Place N of second residue
    coords[1, 0] = place_atom(coords[0, 0], coords[0, 1], coords[0,2], 
                             C_N_LENGTH, CA_C_N_ANGLE, omega)
    
    # Sample phi and psi for second residue
    phi, psi, log_prob = sample_ramachandran_angles(sequence[1], sequence[0], 
                                                     sequence[2] if n_residues > 2 else None)
    total_log_prob += log_prob
    
    # Place CA using phi
    coords[1, 1] = place_atom(coords[0, 1], coords[0, 2], coords[1, 0], 
                             N_CA_LENGTH, C_N_CA_ANGLE, phi)
    
    # Place C using psi
    coords[1, 2] = place_atom(coords[0, 2], coords[1, 0], coords[1, 1], 
                             CA_C_LENGTH, N_CA_C_ANGLE, psi)
    
    # Build remaining residues
    for i in range(2, n_residues):
        # Get neighbor information
        prev_aa = sequence[i-1] if i > 0 else None
        next_aa = sequence[i+1] if i < n_residues - 1 else None
        
        # Sample dihedral angles for current residue
        phi, psi, log_prob = sample_ramachandran_angles(sequence[i], prev_aa, next_aa)
        total_log_prob += log_prob
        
        # Place N atom (omega is typically 180 degrees for trans peptide bonds)
        omega = np.random.normal(0, 0.1)  # Add small variation around trans
        coords[i, 0] = place_atom(coords[i-1, 0], coords[i-1, 1], coords[i-1, 2], 
                                 C_N_LENGTH, CA_C_N_ANGLE, omega)
        
        # Place CA atom using phi angle
        coords[i, 1] = place_atom(coords[i-1, 1], coords[i-1, 2], coords[i, 0], 
                                 N_CA_LENGTH, C_N_CA_ANGLE, phi)
        
        # Place C atom using psi angle  
        coords[i, 2] = place_atom(coords[i-1, 2], coords[i, 0], coords[i, 1], 
                                 CA_C_LENGTH, N_CA_C_ANGLE, psi)
    
    return coords, total_log_prob