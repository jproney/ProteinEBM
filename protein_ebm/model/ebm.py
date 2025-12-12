import torch
from torch import nn
from torch.nn import Module
from dataclasses import dataclass

from protein_ebm.model.layers import (
    PairwiseConditioning,
    SingleConditioning,
    DiffusionTransformer,
    RelativePositionEncoder,
    final_init_
)
from protein_ebm.model.boltz_utils import LinearNoBias


class ProteinEBM(Module):
    """Diffusion module"""

    def __init__(
        self,
        config, 
        diffuser
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        config : ProteinEBMConfig
            Configuration object containing model parameters.
        """
        super().__init__()

        self.config = config
        self.sequence_embedding = nn.Embedding(21, config.token_s)  # 20 amino acids + 1 for missing/masked
        self.diffuser = diffuser
        self.direct_score = getattr(config, 'direct_score', False) # whether to predict the score directly without the energy parameterization
        self.aux_score = getattr(config, 'aux_score', False)
        self.predict_sidechain = getattr(config, 'predict_sidechain', False) # whether to predict the sidechain coordinates as aux outputs
        self.diffuse_sidechain = getattr(config, 'diffuse_sidechain', False) # whether to diffuse sidechain atoms

        self.data_dimension = 3 if not self.diffuse_sidechain else 37*3

        self.noisy_coord_embedding = LinearNoBias(self.data_dimension, config.token_s) # 3 cartesian coords for N, C, CA, or for all 37 atoms

        # Add embedding for whether a residue is in contact with another residue outside the chain
        self.contact_embedding = nn.Embedding(config.num_contact_embeddings, config.token_s)
        
        # Add self-conditioning coordinate embedding if enabled
        self.use_self_conditioning = getattr(config, 'use_self_conditioning', False)
        if self.use_self_conditioning:
            self.self_conditioning_embedding = LinearNoBias(self.data_dimension, config.token_s)

        if self.diffuse_sidechain:
            self.atom_mask_embedding = LinearNoBias(37, config.token_s)

        self.single_conditioner = SingleConditioning(
            input_dim=config.token_s*(3 + int(self.use_self_conditioning) + int(self.diffuse_sidechain)),  # Add dimension for self-conditioning and sidechain mask if enabled
            token_s=config.token_s,
            dim_fourier=config.dim_fourier,
            num_transitions=config.conditioning_transition_layers,
        )

        self.rel_pos = RelativePositionEncoder(config.token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            input_dim=config.token_s*4,
            token_z=config.token_z,
            dim_token_rel_pos_feats=config.token_z,  
            num_transitions=config.conditioning_transition_layers,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * config.token_s), 
            LinearNoBias(2 * config.token_s, 2 * config.token_s)
        )
        final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * config.token_s,
            dim_single_cond=2 * config.token_s,
            dim_pairwise=config.token_z,
            depth=config.token_transformer_depth,
            heads=config.token_transformer_heads,
        )

        self.a_norm = nn.LayerNorm(2 * config.token_s)
        self.r_update_proj = LinearNoBias(2*config.token_s, self.data_dimension)

        if self.aux_score:
            self.r_update_proj_aux = LinearNoBias(2*config.token_s, self.data_dimension)

        self.sidechain_dim  = 36

        if self.predict_sidechain and not self.diffuse_sidechain:
            self.sidechain_proj = LinearNoBias(2*config.token_s,  self.sidechain_dim *3)  # 34 atoms × 3 coordinates

    def forward(
        self,
        aatype,
        r_noisy,
        residue_idx,
        residue_mask,
        times,
        chain_id=None,
        external_contacts=None,
        sc_coords=None,  #  self-conditioning coordinates parameter
        atom_mask=None,
    ):
        """Forward pass
        Parameters
        ----------
        aatype : torch.Tensor
            Amino acid types
        r_noisy : torch.Tensor
            Noisy coordinates
        residue_idx : torch.Tensor
            Residue indices
        residue_mask : torch.Tensor
            Residue mask
        times : torch.Tensor
            Time values for fourier embedding
        chain_id : torch.Tensor, optional
            Chain IDs (defaults to zeros if not provided)
        external_contacts : torch.Tensor, optional
            External contact information for single conditioning [batch, res] - values 0 or 1 (defaults to zeros if not provided)
        sc_coords : torch.Tensor, optional
            Self-conditioning coordinates [batch, length, 3 or 37]
        atom_mask : torch.Tensor, optional
            Mask for the sidechain coordinates [batch, length, 37]
        Returns
        -------
        dict
            Dictionary containing model outputs
        """
        
        # Get batch size and sequence length
        B, N = aatype.shape
        
        # Fill in chain_id with zeros if not provided
        if chain_id is None:
            chain_id = torch.zeros(B, N, dtype=torch.long, device=aatype.device)
        
        # Fill in external_contacts with zeros if not provided
        if external_contacts is None:
            external_contacts = torch.zeros(B, N, dtype=torch.long, device=aatype.device)

        r_noisy = r_noisy * self.diffuser.config.coordinate_scaling
        if sc_coords is not None:
            sc_coords = sc_coords * self.diffuser.config.coordinate_scaling

        sequence_emb = self.sequence_embedding(aatype)

        # Convert external_contacts to long for embedding lookup
        external_contacts_long = external_contacts.long() if external_contacts.dtype != torch.long else external_contacts
        
        residue_embed = torch.cat([
            sequence_emb, 
            self.noisy_coord_embedding(r_noisy),
            self.contact_embedding(external_contacts_long),
            *([self.self_conditioning_embedding(sc_coords if sc_coords is not None else torch.zeros_like(r_noisy))] if self.use_self_conditioning else []),
            *([self.atom_mask_embedding(atom_mask)] if self.diffuse_sidechain else [])
        ], dim=-1)

        s, normed_fourier = self.single_conditioner(
            s=residue_embed,
            times=times
        )

        # Create pairwise input features by tiling s
        num_batch, num_res, _ = s.shape
        s_tiled = torch.cat([
            torch.tile(s[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(s[:, None, :, :], (1, num_res, 1, 1))
        ], dim=-1)


        token_rel_pos_feats = self.rel_pos(residue_idx, chain_id)
        

        z = self.pairwise_conditioner(
            z_trunk=s_tiled, token_rel_pos_feats=token_rel_pos_feats
        )


        a = self.token_transformer(
            s,
            mask=residue_mask.float(),
            s=s,
            z=z
        )
        a = self.a_norm(a)
        r_update = self.r_update_proj(a)
        
        output = {"r_update": r_update, "token_a": a.detach()}
        
        if self.aux_score:
            r_update_aux = self.r_update_proj_aux(a)
            output["r_update_aux"] = r_update_aux
            
        if self.predict_sidechain and not self.diffuse_sidechain:
            sidechain_coords = self.sidechain_proj(a)  # [batch, res, 34*3]
            # Reshape to [batch, res, 34, 3]
            batch_size, num_res, _ = sidechain_coords.shape
            sidechain_coords = sidechain_coords.view(batch_size, num_res,  self.sidechain_dim, 3)
            output["sidechain_coords"] = sidechain_coords

        return output

    def compute_energy(self, input_feats):
        """Compute the energy of a protein structure.

        Args:
            input_feats: Dictionary containing input features including:
                - r_noisy: Noisy coordinates [batch, res, 3 or 37*3]
                - aatype: Amino acid types [batch, res]
                - residue_idx: Residue indices [batch, res]
                - mask: Residue mask [batch, res]
                - t: Time step [batch]
                - chain_encoding: Chain IDs [batch, res] (optional, defaults to zeros)
                - external_contacts: External contact information [batch, res] (optional, defaults to zeros)
                - selfcond_coords: Self-conditioning coordinates [batch, res, 3 or 37*3] (optional)
                - atom_mask: Atom mask [batch, res, 37] (optional)

        Returns:
            dict: Dictionary containing model outputs including:
                - energy: Scalar energy value for each sequence in the batch
                - r_update: Coordinate updates
                - token_a: Token embeddings
        """
        # Extract features
        r_noisy = input_feats['r_noisy']
        aatype = input_feats['aatype']
        residue_idx = input_feats['residue_idx']
        residue_mask = input_feats['mask']
        sc_coords = input_feats.get('selfcond_coords', None)
        atom_mask = input_feats.get('atom_mask', None)
        times = input_feats['t']
        
        # Get optional features (will be filled with zeros in forward if None)
        chain_id = input_feats.get('chain_encoding', None)
        external_contacts = input_feats.get('external_contacts', None)
        
        # Run main network
        model_out = self.forward(
            aatype=aatype,
            r_noisy=r_noisy,
            residue_idx=residue_idx,
            residue_mask=residue_mask,
            times=times,
            chain_id=chain_id,
            external_contacts=external_contacts,
            sc_coords=sc_coords if self.use_self_conditioning else None,
            atom_mask=atom_mask if self.diffuse_sidechain else None,
        )

        # Compute energy values
        energy_values = torch.sum(model_out['r_update']**2, dim=-1)  # [batch_size, seq_len]
        model_out['energy'] = (energy_values * residue_mask).sum(dim=1)

        if self.aux_score:
            # Predict denoised coordinates using the aux score
            pred_coords = self.diffuser.calc_trans_0(
                score_t=model_out['r_update_aux'],
                x_t=input_feats['r_noisy'],
                t=input_feats['t'],  # Use full batch of time values
                use_torch=True
            )
            # Reshape to [B, N, 3, 3] for backbone atoms
            B, N = input_feats['r_noisy'].shape[:2]
            model_out['pred_coords_aux'] = pred_coords.reshape(B, N, -1, 3)
            
            # Mask pred_coords_aux to 0 where atom_mask is 0
            if self.diffuse_sidechain and atom_mask is not None:
                model_out['pred_coords_aux'] = model_out['pred_coords_aux'] * atom_mask[..., None]

        return model_out

    @torch.enable_grad()
    def compute_score(self, input_feats):
        """Compute the score (negative gradient of energy) with respect to the coordinates.

        Args:
            input_feats: Dictionary containing input features including:
                - r_noisy: Noisy coordinates [batch, res, 3 or 37*3]
                - aatype: Amino acid types [batch, res]
                - residue_idx: Residue indices [batch, res]
                - mask: Residue mask [batch, res]
                - t: Time step [batch]
                - chain_encoding: Chain IDs [batch, res] (optional, defaults to zeros)
                - external_contacts: External contact information [batch, res] (optional, defaults to zeros)
                - selfcond_coords: Self-conditioning coordinates [batch, res, 3 or 37*3] (optional)
                - atom_mask: Atom mask [batch, res, 37] (optional)

        Returns:
            dict: Dictionary containing model outputs with score entries updated
        """
        if self.direct_score:
            # Just return the energy computation without gradient
            model_out = self.compute_energy(input_feats)
            model_out['trans_score'] = model_out['r_update']
 
            pred_coords = self.diffuser.calc_trans_0(
                score_t=model_out['trans_score'],
                x_t=input_feats['r_noisy'],
                t=input_feats['t'],  # Use full batch of time values
                use_torch=True
            )
            # Reshape to [B, N, 3, 3] for backbone atoms
            B, N = input_feats['r_noisy'].shape[:2]
            model_out['pred_coords'] = pred_coords.reshape(B, N, -1, 3)
            
            # Mask pred_coords to 0 where atom_mask is 0
            if self.diffuse_sidechain and 'atom_mask' in input_feats and input_feats['atom_mask'] is not None:
                model_out['pred_coords'] = model_out['pred_coords'] * input_feats['atom_mask'][..., None]
            
            return model_out

        # require gradients for input coords
        r_noisy = input_feats['r_noisy'].detach().requires_grad_(True)
        
        # Update input features with new coordinates
        input_feats = {**input_feats, 'r_noisy': r_noisy}

        # Run network with gradients enabled
        model_out = self.compute_energy(input_feats)

        # Get energy and compute gradients
        energy = model_out['energy']
        grad_r = torch.autograd.grad(energy.sum(), r_noisy, create_graph=True)[0]

        # Save old (non-conservative) scores
        if self.aux_score:
            model_out['trans_score_aux'] = model_out['r_update_aux']

        # Update with conservative scores
        model_out['trans_score'] = -grad_r

        # Predict denoised coordinates using calc_trans_0
        pred_coords = self.diffuser.calc_trans_0(
            score_t=model_out['trans_score'],
            x_t=r_noisy,
            t=input_feats['t'],  # Use full batch of time values
            use_torch=True
        )
            
        # Reshape to [B, N, 3, 3] for backbone atoms
        B, N = r_noisy.shape[:2]
        model_out['pred_coords'] = pred_coords.reshape(B, N, -1, 3)

        # Mask pred_coords to 0 where atom_mask is 0
        if self.diffuse_sidechain and 'atom_mask' in input_feats and input_feats['atom_mask'] is not None:
            model_out['pred_coords'] = model_out['pred_coords'] * input_feats['atom_mask'][..., None]

        return model_out
