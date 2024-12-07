"""
IEOR 6617: This is based on 
https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py

This is a pytorch implementation of the Performer described in 'Rethinking Attention with Performers'
"""

from models_v2 import vit_models, Layer_scale_init_Block, FastAttention
from models_v2_rope import rope_vit_models, apply_rotary_emb
from functools import partial

import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops import rearrange


class PerformerAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % self.num_heads == 0, 'dimension must be divisible by number of heads'

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj_drop = nn.Dropout(proj_drop)
        
        # number of random features for performer
        self.nb_features = 256

        inner_dim = head_dim * self.num_heads
        self.fast_attention = FastAttention(head_dim, 
                                            self.nb_features, 
                                            causal = False, 
                                            generalized_attention = False, 
                                            kernel_fn = nn.ReLU(),
                                            no_projection = False)
        
        self.proj = nn.Linear(inner_dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        x = self.fast_attention(q, k, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PerformerRoPEAttention(PerformerAttention):
    """Multi-head Attention block with relative position embeddings."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # apply embeddings
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        
        out = self.fast_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out

class Performer_RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = PerformerRoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x

# Performer standard
@register_model
def performer_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, 
        use_performer=True, Attention_block=PerformerAttention,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


# Performer RoPE-Axial
@register_model
def performer_rope_axial_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, 
        Attention_block=PerformerRoPEAttention,
        rope_theta=100.0, rope_mixed=False, use_performer=True, **kwargs)
    model.default_cfg = _cfg()
    return model

# Performer RoPE-Axial + APE
@register_model
def performer_rope_axial_ape_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=100.0, rope_mixed=False, use_ape=True, use_performer=True, **kwargs)
    model.default_cfg = _cfg()
    return model

# Performer RoPE-Mixed
@register_model
def performer_rope_mixed_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_performer=True, **kwargs)
    model.default_cfg = _cfg()
    return model

# Performer RoPE-Mixed + APE
@register_model
def performer_rope_mixed_ape_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    model.default_cfg = _cfg()
    return model