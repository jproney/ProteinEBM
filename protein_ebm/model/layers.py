# based off of https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro

from einops.layers.torch import Rearrange
import torch
from torch import Tensor, nn, sigmoid
import math
import numpy as np
from scipy.stats import truncnorm
from protein_ebm.model.boltz_utils import LinearNoBias, SwiGLU, default


from torch.nn import (
    LayerNorm,
    Linear,
    ModuleList,
    Sequential,
)
from torch.nn.functional import one_hot
from math import pi
from einops import rearrange

"""
utilities for initializing weights from Boltz-1
"""

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def bias_init_zero_(bias):
    with torch.no_grad():
        bias.fill_(0.0)


def bias_init_one_(bias):
    with torch.no_grad():
        bias.fill_(1.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization"""

    def __init__(self, dim, dim_single_cond):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        dim : int
            The input dimension.
        dim_single_cond : int
            The single condition dimension.

        """
        super().__init__()
        self.a_norm = LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = LayerNorm(dim_single_cond, bias=False)
        self.s_scale = Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


class ConditionedTransitionBlock(nn.Module):
    """Conditioned Transition Block"""

    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        """Initialize the conditioned transition block.

        Parameters
        ----------
        dim_single : int
            The single dimension.
        dim_single_cond : int
            The single condition dimension.
        expansion_factor : int, optional
            The expansion factor, by default 2

        """
        super().__init__()

        self.adaln = AdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = Sequential(
            LinearNoBias(dim_single, dim_inner * 2),
            SwiGLU(),
        )
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        self.b_to_a = LinearNoBias(dim_inner, dim_single)

        output_projection_linear = Linear(dim_single_cond, dim_single)
        nn.init.zeros_(output_projection_linear.weight)
        nn.init.constant_(output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(output_projection_linear, nn.Sigmoid())

    def forward(
        self,
        a,
        s,
    ):
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        a = self.output_projection(s) * self.b_to_a(b)

        return a



"""
Network layers from Boltz-1
"""

class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
            Rearrange("b ... h -> b h ..."),
        )

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        final_init_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = s.shape[0]

        # Layer norms
        if self.initial_norm:
            s = self.norm_s(s)

        k_in = s

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        z = self.proj_z(z)


        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (self.head_dim**0.5) + z.float()
            # The pairwise mask tensor (B, N) is broadcasted to (B, 1, 1, N) and (B, H, N, N)
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)

            # Compute output
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o
    

class DiffusionTransformerLayer(nn.Module):
    """Diffusion Transformer Layer"""

    def __init__(
        self,
        heads,
        dim=384,
        dim_single_cond=None,
        dim_pairwise=128,
    ):
        """Initialize the diffusion transformer layer.

        Parameters
        ----------
        heads : int
            The number of heads.
        dim : int, optional
            The dimension, by default 384
        dim_single_cond : int, optional
            The single condition dimension, by default None
        dim_pairwise : int, optional
            The pairwise dimension, by default 128

        """
        super().__init__()

        dim_single_cond = default(dim_single_cond, dim)

        self.adaln = AdaLN(dim, dim_single_cond)

        self.pair_bias_attn = AttentionPairBias(
            c_s=dim, c_z=dim_pairwise, num_heads=heads, initial_norm=False
        )

        self.output_projection_linear = Linear(dim_single_cond, dim)
        nn.init.zeros_(self.output_projection_linear.weight)
        nn.init.constant_(self.output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(
            self.output_projection_linear, nn.Sigmoid()
        )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond
        )

    def forward(
        self,
        a,
        s,
        z,
        mask=None
    ):
        b = self.adaln(a, s)
        b = self.pair_bias_attn(
            s=b,
            z=z,
            mask=mask,
        )
        b = self.output_projection(s) * b

        # NOTE: Added residual connection!
        a = a + b
        a = a + self.transition(a, s)
        return a

class DiffusionTransformer(nn.Module):
    """Diffusion Transformer"""

    def __init__(
        self,
        depth,
        heads,
        dim=384,
        dim_single_cond=None,
        dim_pairwise=128,
        activation_checkpointing=False,
    ):
        """Initialize the diffusion transformer.

        Parameters
        ----------
        depth : int
            The depth.
        heads : int
            The number of heads.
        dim : int, optional
            The dimension, by default 384
        dim_single_cond : int, optional
            The single condition dimension, by default None
        dim_pairwise : int, optional
            The pairwise dimension, by default 128
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False

        """
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        dim_single_cond = default(dim_single_cond, dim)

        self.layers = ModuleList()
        for _ in range(depth):
            if activation_checkpointing:
                self.layers.append(
                    DiffusionTransformerLayer(
                        heads,
                        dim,
                        dim_single_cond,
                        dim_pairwise,
                    )
                )
            else:
                self.layers.append(
                    DiffusionTransformerLayer(
                        heads,
                        dim,
                        dim_single_cond,
                        dim_pairwise,
                    )
                )

    def forward(
        self,
        a,
        s,
        z,
        mask=None
    ):
        for i, layer in enumerate(self.layers):
            a = layer(
                a,
                s,
                z,
                mask=mask,
            )
        return a


"""
Various conditioning encoders from Boltz-1
"""


class Transition(nn.Module):
    """Perform a two-layer MLP."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512
        out_dim: Optional[int]
            The dimension of the output, default None

        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

        bias_init_one_(self.norm.weight)
        bias_init_zero_(self.norm.bias)

        lecun_normal_init_(self.fc1.weight)
        lecun_normal_init_(self.fc2.weight)
        final_init_(self.fc3.weight)

    def forward(self, x: Tensor, chunk_size: int = None) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        x = self.norm(x)

        if chunk_size is None or self.training:
            x = self.silu(self.fc1(x)) * self.fc2(x)
            x = self.fc3(x)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.hidden, chunk_size):
                fc1_slice = self.fc1.weight[i : i + chunk_size, :]
                fc2_slice = self.fc2.weight[i : i + chunk_size, :]
                fc3_slice = self.fc3.weight[:, i : i + chunk_size]
                x_chunk = self.silu((x @ fc1_slice.T)) * (x @ fc2_slice.T)
                if i == 0:
                    x_out = x_chunk @ fc3_slice.T
                else:
                    x_out = x_out + x_chunk @ fc3_slice.T
            return x_out


class FourierEmbedding(nn.Module):
    """Fourier embedding layer."""

    def __init__(self, dim):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the embeddings.

        """

        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)


class RelativePositionEncoder(nn.Module):
    """Relative position encoder."""

    def __init__(self, token_z, r_max=32, s_max=1):
        """Initialize the relative position encoder.

        Parameters
        ----------
        token_z : int
            The pair representation dimension.
        r_max : int, optional
            The maximum index distance, by default 32.
        s_max : int, optional
            The maximum chain distance, by default 2.

        """
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear_layer = LinearNoBias(2 * (r_max + 1) + (self.s_max + 1), token_z)

    def forward(self, residue_index, chain_id):
        b_same_chain = torch.eq(
            chain_id[:, :, None], chain_id[:, None, :]
        )

        rel_pos = (
            residue_index[:, :, None] - residue_index[:, None, :]
        )

        d_residue = torch.clip(
            rel_pos + self.r_max,
            0,
            2 * self.r_max,
        )

        d_residue = torch.where(
            b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1
        )
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        d_chain = torch.clip(
            torch.abs(chain_id[:, :, None] - chain_id[:, None, :]),
            0,
            self.s_max
        ) 

        a_rel_chain = one_hot(d_chain, self.s_max + 1)

        p = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.to(dtype=self.linear_layer.weight.dtype),
                    a_rel_chain.to(dtype=self.linear_layer.weight.dtype),
                ],
                dim=-1,
            )
        )
        return p


class SingleConditioning(nn.Module):
    def __init__(
        self,
        input_dim=384,
        token_s=384,
        dim_fourier=256,
        num_transitions=2,
        transition_expansion_factor=2,
        eps=1e-20,
    ):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        token_s : int, optional
            The single representation dimension, by default 384.
        dim_fourier : int, optional
            The fourier embeddings dimension, by default 256.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.
        eps : float, optional
            The epsilon value, by default 1e-20.

        """
        super().__init__()
        self.eps = eps

        self.norm_single = nn.LayerNorm(input_dim)
        self.single_embed = nn.Linear(input_dim, 2 * token_s)
        
        # Initialize fourier embedding components
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, 2 * token_s)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=2 * token_s, hidden=transition_expansion_factor * 2 * token_s
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        *,
        s,
        times=None,
        direct_embedding=None,
    ):
        """Forward pass that accepts either times or direct_embedding.

        Parameters
        ----------
        s : torch.Tensor
            Input sequence tensor
        times : torch.Tensor, optional
            Time values for fourier embedding
        direct_embedding : torch.Tensor, optional
            Direct embedding to use instead of fourier embedding

        Returns
        -------
        tuple
            (conditioned sequence, embedding used)
        """
        if times is None and direct_embedding is None:
            raise ValueError("Either times or direct_embedding must be provided")
        if times is not None and direct_embedding is not None:
            raise ValueError("Cannot provide both times and direct_embedding")

        s = self.single_embed(self.norm_single(s))
        
        if direct_embedding is not None:
            fourier_to_single = direct_embedding
            normed_fourier = direct_embedding
        else:
            fourier_embed = self.fourier_embed(times)
            normed_fourier = self.norm_fourier(fourier_embed)
            fourier_to_single = self.fourier_to_single(normed_fourier)

        s = rearrange(fourier_to_single, "b d -> b 1 d") + s

        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier


class PairwiseConditioning(nn.Module):
    """Pairwise conditioning layer."""

    def __init__(
        self,
        input_dim,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        """Initialize the pairwise conditioning layer.

        Parameters
        ----------
        token_z : int
            The pair representation dimension.
        dim_token_rel_pos_feats : int
            The token relative position features dimension.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.

        """
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(input_dim + dim_token_rel_pos_feats),
            LinearNoBias(input_dim + dim_token_rel_pos_feats, token_z),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=token_z, hidden=transition_expansion_factor * token_z
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        z_trunk,
        token_rel_pos_feats,
    ):
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z