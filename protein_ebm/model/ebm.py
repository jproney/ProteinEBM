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
        
        # Attention masking based on distance
        self.use_attention_mask = getattr(config, 'use_attention_mask', False)
        
        self.data_dimension = 3 if not self.diffuse_sidechain else 37*3

        self.noisy_coord_embedding = LinearNoBias(self.data_dimension, config.token_s) # 3 cartesian coords for N, C, CA, or for all 37 atoms

        # Add embedding for whether a residue is in contact with another residue outside the chain
        self.contact_embedding = nn.Embedding(config.num_contact_embeddings, config.token_s)

        # Optional embedding for whether a residue is present in the structure as a prediction target
        self.use_present_embedding = getattr(config, 'use_present_embedding', False)
        if self.use_present_embedding:
            self.present_embedding = nn.Embedding(2, config.token_s)  # 0 = not present, 1 = present
        
        # Add self-conditioning coordinate embedding if enabled
        self.use_self_conditioning = getattr(config, 'use_self_conditioning', False)
        if self.use_self_conditioning:
            self.self_conditioning_embedding = LinearNoBias(self.data_dimension, config.token_s)

        if self.diffuse_sidechain:
            self.atom_mask_embedding = LinearNoBias(37, config.token_s)

        # Include coordinate embedding, self-conditioning, and sidechain mask
        single_cond_input_dim = config.token_s*(3 + int(self.use_self_conditioning) + int(self.diffuse_sidechain) + int(self.use_present_embedding))
    

        self.single_conditioner = SingleConditioning(
            input_dim=single_cond_input_dim,
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

        # First transformer stack (main/trunk)
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
        r_noisy,  # [B, N, 3 or 37*3]
        residue_idx,
        residue_mask,
        times,
        chain_id=None,
        external_contacts=None,
        present=None,
        sc_coords=None,  #  self-conditioning coordinates parameter
        atom_mask=None,
        rescale_input_coords=True # don't set this to False unless you know what you're doing
    ):
        """Forward pass
        Parameters
        ----------
        aatype : torch.Tensor
            Amino acid types [batch, n_residues]
        r_noisy : torch.Tensor. Noisy coordinates. 
        residue_idx : torch.Tensor
            Residue indices [batch, n_residues]
        residue_mask : torch.Tensor
            Residue mask [batch, n_residues]
        times : torch.Tensor
            Time values for fourier embedding [batch]
        chain_id : torch.Tensor, optional
            Chain IDs [batch, n_residues] (defaults to zeros if not provided)
        external_contacts : torch.Tensor, optional
            External contact information for single conditioning [batch, n_residues] (defaults to zeros if not provided)
        present : torch.Tensor, optional
            Whether each residue is present in the structure as a prediction target [batch, n_residues].
            Values 0 (not present) or 1 (present). If None and use_present_embedding, defaults to all ones.
        sc_coords : torch.Tensor, optional
            Self-conditioning coordinates [batch, n_residues, 3 or 37*3]
        atom_mask : torch.Tensor, optional
            Mask for the sidechain coordinates [batch, n_residues, 37]

        Returns
        -------
        dict
            Dictionary containing model outputs:
            - 'r_update': Coordinate updates (format matches r_noisy)
            - 'token_a': Token-level representations [batch, n_residues, dim]
        """
        
        # Get batch size and sequence length
        B, N = aatype.shape
        
        # Fill in chain_id with zeros if not provided
        if chain_id is None:
            chain_id = torch.zeros(B, N, dtype=torch.long, device=aatype.device)
        
        # Fill in external_contacts with zeros if not provided
        if external_contacts is None:
            external_contacts = torch.ones(B, N, dtype=torch.long, device=aatype.device) if getattr(self.config, 'num_contact_embeddings', 2) == 3 else torch.zeros(B, N, dtype=torch.long, device=aatype.device)

        # Fill in present with ones (all present) if not provided and present embedding is used
        if self.use_present_embedding:
            if present is None:
                present = torch.ones(B, N, dtype=torch.long, device=aatype.device)
            else:
                present = (present > 0.5).long() if present.dtype != torch.long else present.clamp(0, 1)

        if rescale_input_coords:
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
            *([self.present_embedding(present)] if self.use_present_embedding else []),
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

            
        a = s
        a = self.token_transformer(
            a,
            mask=residue_mask.float(),
            s=s,
            z=z,
        )
        a = self.a_norm(a)
        r_update = self.r_update_proj(a)
        r_update_aux = self.r_update_proj_aux(a)
        
        output = {"r_update": r_update, "token_a": a.detach()}
        
        if self.aux_score:
            output["r_update_aux"] = r_update_aux
        
            
        if self.predict_sidechain and not self.diffuse_sidechain:
            sidechain_coords = self.sidechain_proj(a)  # [batch, res, 34*3]
            # Reshape to [batch, res, 34, 3]
            batch_size, num_res, _ = sidechain_coords.shape
            sidechain_coords = sidechain_coords.view(batch_size, num_res,  self.sidechain_dim, 3)
            output["sidechain_coords"] = sidechain_coords

        return output

    def compute_energy(self, input_feats, rescale_input_coords=True):
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
                - present: Whether each residue is present as a prediction target [batch, res], 0/1 (optional, if use_present_embedding)
                - selfcond_coords: Self-conditioning coordinates [batch, res, 3 or 37*3] (optional)
                - atom_mask: Atom mask [batch, res, 37] (optional, for sidechain diffusion)
                
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
        present = input_feats.get('present', None)
        
        
        # Run main network
        model_out = self.forward(
            aatype=aatype,
            r_noisy=r_noisy,
            residue_idx=residue_idx,
            residue_mask=residue_mask,
            times=times,
            chain_id=chain_id,
            external_contacts=external_contacts,
            present=present if self.use_present_embedding else None,
            sc_coords=sc_coords if self.use_self_conditioning else None,
            atom_mask=atom_mask if self.diffuse_sidechain else None,
            rescale_input_coords=rescale_input_coords
        )

        # Compute energy values
        energy_values = torch.sum(model_out['r_update']**2, dim=-1)  # [batch_size, seq_len or n_atoms]
        
        mask_for_energy = residue_mask
        
        model_out['energy'] = (energy_values * mask_for_energy).sum(dim=1)
        model_out['per_residue_energy'] = energy_values * mask_for_energy


        if self.aux_score:
            # Predict denoised coordinates using the aux score
            if getattr(self.config, 'use_edm_param', False):
                sigma_data = 1 / self.diffuser.config.coordinate_scaling
                beta = 1 - self.diffuser.conditional_var(input_feats['t'], use_torch=True)
                c_in = 1 / sigma_data
                c_skip = torch.sqrt(beta)
                c_out = sigma_data * torch.sqrt(1 - beta)
                model_out['mse_loss_scale'] = 1 / (sigma_data**2 * (1-beta) + 1e-3)

                pred_coords = c_skip.view(-1, 1, 1) * r_noisy + c_out.view(-1, 1, 1) * model_out['r_update_aux']
            else:

                scale = torch.sqrt(self.diffuser.conditional_var(input_feats['t'], use_torch=True)).view(-1, 1, 1) if getattr(self.config, 'precondition', False) else 1.0
                pred_coords = self.diffuser.calc_trans_0(
                    score_t=model_out['r_update_aux'] / scale,
                    x_t=input_feats['r_noisy'] / (self.diffuser.config.coordinate_scaling if getattr(self.config, 'precondition', False) else 1.0),
                    t=input_feats['t'],  # Use full batch of time values
                    use_torch=True
                )
                # Reshape to [B, N, 3, 3] for backbone atoms
            B, N = input_feats['r_noisy'].shape[:2]

            # Mask pred_coords_aux to 0 where atom_mask is 0
            if self.diffuse_sidechain and atom_mask is not None:
                model_out['pred_coords_aux'] = pred_coords.reshape(B, N, -1, 3)
                model_out['pred_coords_aux'] = model_out['pred_coords_aux'] * atom_mask[..., None]
            else:
                model_out['pred_coords_aux'] = pred_coords
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
                - present: Whether each residue is present as a prediction target [batch, res], 0/1 (optional, if use_present_embedding)
                - selfcond_coords: Self-conditioning coordinates [batch, res, 3 or 37*3] (optional)
                - atom_mask: Atom mask [batch, res, 37] (optional, for sidechain diffusion)
                

        Returns:
            dict: Dictionary containing model outputs with score entries updated
        """
        if self.direct_score:
            # Just return the energy computation without gradient
            model_out = self.compute_energy(input_feats)
            model_out['trans_score'] = model_out['r_update'] 
            r_noisy = input_feats['r_noisy']

            scale = torch.sqrt(self.diffuser.conditional_var(input_feats['t'], use_torch=True)).view(-1, 1, 1) if getattr(self.config, 'precondition', False) else 1.0
            pred_coords = self.diffuser.calc_trans_0(
                score_t=model_out['trans_score'] / scale,
                x_t=input_feats['r_noisy'],
                t=input_feats['t'],  # Use full batch of time values
                use_torch=True
            )
            # Reshape to [B, N, 3, 3] for backbone atoms
            B, N = input_feats['r_noisy'].shape[:2]
            
            model_out['pred_coords'] = pred_coords

            return model_out

        # require gradients for input coords
        if getattr(self.config, 'precondition', False):
            r_noisy = (input_feats['r_noisy'] * self.diffuser.config.coordinate_scaling).detach().requires_grad_(True) 
        else:
            r_noisy = input_feats['r_noisy'].detach().requires_grad_(True)
        

        input_feats = {**input_feats, 'r_noisy': r_noisy}
        grad_target = r_noisy  # Take gradients w.r.t. residue-level coords

        # Run network with gradients enabled
        model_out = self.compute_energy(input_feats, rescale_input_coords=not getattr(self.config, 'precondition', False))

        # Get energy and compute gradients
        energy = model_out['energy']
        grad_r = torch.autograd.grad(energy.sum(), grad_target, create_graph=True)[0]

        # Save old (non-conservative) scores
        if self.aux_score:
            model_out['trans_score_aux'] = model_out['r_update_aux']

        # Update with conservative scores
        model_out['trans_score'] = -grad_r

        # Predict denoised coordinates using calc_trans_0
        scale = torch.sqrt(self.diffuser.conditional_var(input_feats['t'], use_torch=True)).view(-1, 1, 1) if getattr(self.config, 'precondition', False) else 1.0
        pred_coords = self.diffuser.calc_trans_0(
            score_t=model_out['trans_score'] / scale,
            x_t=r_noisy / (self.diffuser.config.coordinate_scaling if getattr(self.config, 'precondition', False) else 1.0),
            t=input_feats['t'],  # Use full batch of time values
            use_torch=True
        )
            
        # Reshape to [B, N, 3, 3] for backbone atoms
        B, N = r_noisy.shape[:2]

        # Mask pred_coords to 0 where atom_mask is 0
        if self.diffuse_sidechain and atom_mask is not None:
            model_out['pred_coords'] = pred_coords.reshape(B, N, -1, 3)
            model_out['pred_coords'] = model_out['pred_coords'] * input_feats['atom_mask'][..., None]
        else:
            model_out['pred_coords'] = pred_coords

        return model_out