import random
from collections import OrderedDict
import numpy as np
import timm
import torch
from torch import nn
# import torchvision.transforms.functional_tensor as F_t
from torchvision import transforms
from copy import deepcopy
from functools import partial
from einops import rearrange, repeat
from models.Transformer import TransformerModel
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import (
    default_cfgs,
    build_model_with_cfg,
    checkpoint_filter_fn,
)

def forward_attn(self, x):
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn.detach()
def forward_block(self, x):
    attn_x, attn = forward_attn(self.attn, self.norm1(x))
    x = x + self.drop_path(attn_x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))

    return x, attn

from models.TABS_withbn import TABS

class MaskVisionTransformer(VisionTransformer):
    # def __init__(self, mask_ratio=0, **kwargs):
    def __init__(self, mask_ratio=0, default_cfg=None,representation_size=None, pretrained_filter_fn=None, pretrained_custom_load=None,**kwargs):
        if mask_ratio > 0:
            kwargs = dict(drop_path_rate=0.1, **kwargs)
        super(MaskVisionTransformer, self).__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.patch_embed = TABS()
       
        self.transformer1 = TransformerModel(
            512,
            4,
            8,
            512*4,
            0.1,
            0.0,
            )
        
        self.transformer2 = TransformerModel(
            512,
            4,
            8,
            512*4,
            0.1,
            0.0,
        )
        self.head = nn.Sequential(nn.Linear(512*3, 2)
                                      )

        self.cls_token_t2 = nn.Parameter(torch.randn(1, 1, 512))
        self.cls_token_dce = nn.Parameter(torch.randn(1, 1, 512))
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, 512))
        self.cls_token_dwi = nn.Parameter(torch.randn(1, 1, 512))
        self.pos_embed_t2 = nn.Parameter(torch.zeros(1, 520+1, 512))
        self.pos_embed_dce =  nn.Parameter(torch.zeros(1, 1820+1 , 512))
        self.pos_embed_dwi = nn.Parameter(torch.zeros(1, 105+1, 512))
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.head_dist = None
        self.head_cls = nn.Sequential(
                                      
                                      nn.Linear(512*3, 2)
                                      )
        self.num_features = 512*3
    
    
    def forward_features(self, dce, dwi, t2,  mask=None, need_attn=False, reverse=False, img_type='t2'):
        mask = None
        x, x1, x2, ids_keep = self.patch_embed(dce, dwi, t2, mask)
        #b, t, n, _ = x.shape
        #if mask is None:
        x = x + self.pos_embed_dce[:,1:,:]
        x1 = x1 + self.pos_embed_dwi[:,1:,:]
        x2 = x2 + self.pos_embed_t2[:,1:,:]

        cls_token_dce = self.cls_token_dce + self.pos_embed_dce[:, :1, :]
        cls_token_dce = cls_token_dce.expand(
                x.shape[0], -1, -1
            )
        x = torch.cat((cls_token_dce, x), dim=1)

        cls_token_dwi = self.cls_token_dwi + self.pos_embed_dwi[:, :1, :]
        cls_token_dwi = cls_token_dwi.expand(
                x1.shape[0], -1, -1
            )
        x1 = torch.cat((cls_token_dwi, x1), dim=1)

        cls_token_t2 = self.cls_token_t2 + self.pos_embed_t2[:, :1, :]
        cls_token_t2 = cls_token_t2.expand(
                x2.shape[0], -1, -1
            )
        x2 = torch.cat((cls_token_t2, x2), dim=1)

        ids_restore = None
        x1, _ = self.transformer1(x1)
        x1 = self.norm1(x1)

        x2, _ = self.transformer2(x2)
        x2 = self.norm2(x2)
            
        x = self.blocks(x)
        x = self.norm(x)
        attn = None
            
        x = x[:,:1]
        x1 = x1[:,:1]
        x2 = x2[:,:1]
        x3 = torch.cat([x, x1, x2], dim=2)
        return x3, attn, ids_restore, mask, x, x1, x2

    def forward(self, dce, dwi, t2, mask=None, need_attn=False, reverse=False, img_type='t2', epoch=0):
        x = self.forward_features(dce, dwi, t2, mask=mask, need_attn=need_attn, reverse=reverse, img_type=img_type)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # x = self.head(x)
            x = x
            # x = self.head_cls(x[0])
        #else:
        #    x = self.head(x[0][:, 0])
        return x[0]

@register_model
def mask_vit_small_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=512, depth=4, num_heads=8, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model
'''
@register_model
def mask_vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model

@register_model
def mask_vit_large_patch16_224(pretrained=False, **kwargs):
    """ViT-Large (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model


@register_model
def mask_vit_base_patch32_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224",
        MaskVisionTransformer,
        pretrained=pretrained,
        **model_kwargs,
    )
    return model
'''
def _create_vision_transformer(
    variant,
    transformer=MaskVisionTransformer,
    pretrained=False,
    default_cfg=None,
    **kwargs,
):
    # default_cfg = default_cfg or default_cfgs[variant]
    default_cfg = default_cfgs[variant].default
    default_cfg = vars(default_cfg)
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # default_num_classes = default_cfg["num_classes"]
    # num_classes = kwargs.get("num_classes", default_num_classes)
    # repr_size = kwargs.pop("representation_size", None)
    default_num_classes = 2
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        print("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        transformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in default_cfg["url"],
        **kwargs,
    )
    return model
