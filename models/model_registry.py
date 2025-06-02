import torch
import torch.nn as nn
from functools import partial
from functions.utils import load_pretrained_weights
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import load_pretrained
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from .full_model import MultiTaskDeiT
from munch import Munch
from typing import Optional

import math
from dataclasses import dataclass


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 
        'input_size': (3, 224, 224), 
        'pool_size': None,
        'crop_pct': .9, 
        'interpolation': 
        'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 
        'classifier': 'head',
        **kwargs
    }
      

@register_model
def MultiTaskDeiT_patch16_tiny(do_jigsaw : bool, 
                       do_coloring : bool, 
                       do_classification : bool, 
                       pixel_shuffle_cfg : Munch, 
                       jigsaw_cfg : Munch,
                       n_classes : int,
                       img_size : int,
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       pretrained_model_info : Optional[Munch] = None,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle_cfg=pixel_shuffle_cfg,
                          jigsaw_cfg=jigsaw_cfg,
                          n_classes=n_classes,
                          img_size=img_size,
                          patch_size=16,
                          embed_dim=192,
                          depth=12,
                          num_heads=3,
                          mlp_ratio=4,
                          qkv_bias=True,
                          verbose=verbose,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_model_info is None:
            raise Exception('Requested Pretrained Model, but did not provide Pretrained path and info')
        load_pretrained_weights(model, pretrained_model_info, img_size=img_size, verbose=verbose)
    del model.head
    return model

@register_model
def MultiTaskDeiT_patch16_tiny_3_models(do_jigsaw : bool, 
                       do_coloring : bool, 
                       do_classification : bool, 
                       pixel_shuffle_cfg : Munch, 
                       jigsaw_cfg : Munch,
                       n_classes : int,
                       img_size : int,
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       pretrained_model_info1 : Optional[Munch] = None,
                       pretrained_model_info2 : Optional[Munch] = None,
                       pretrained_model_info3 : Optional[Munch] = None,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle_cfg=pixel_shuffle_cfg,
                          jigsaw_cfg=jigsaw_cfg,
                          n_classes=n_classes,
                          img_size=img_size,
                          patch_size=16,
                          embed_dim=192,
                          depth=12,
                          num_heads=3,
                          mlp_ratio=4,
                          qkv_bias=True,
                          verbose=verbose,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_model_info1 is None:
            raise Exception('Requested Pretrained Model, but did not provide Pretrained path and info')
        load_pretrained_weights(model, pretrained_model_info1, img_size=img_size, verbose=verbose)
        load_pretrained_weights(model, pretrained_model_info2, img_size=img_size, verbose=verbose)
        load_pretrained_weights(model, pretrained_model_info3, img_size=img_size, verbose=verbose)
    del model.head
    return model

@register_model
def MultiTaskDeiT_patch16_small(do_jigsaw : bool, 
                       do_coloring : bool, 
                       do_classification : bool, 
                       pixel_shuffle : bool, 
                       n_jigsaw_patches : int,
                       n_classes : int,
                       img_size : int,
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       pretrained_model_info : Optional[Munch] = None,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_jigsaw_patches=n_jigsaw_patches,
                          n_classes=n_classes,
                          img_size=img_size,
                          patch_size=16,
                          embed_dim=384,
                          depth=12,
                          num_heads=6,
                          mlp_ratio=4,
                          qkv_bias=True,
                          verbose=verbose,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_model_info is None:
            raise Exception('Requested Pretrained Model, but did not provide Pretrained path and info')
        load_pretrained_weights(model, pretrained_model_info, img_size=img_size, verbose=verbose)
    del model.head
    return model

@register_model
def MultiTaskDeiT_patch8_small(do_jigsaw : bool, 
                       do_coloring : bool, 
                       do_classification : bool, 
                       pixel_shuffle : bool, 
                       n_jigsaw_patches : int,
                       n_classes : int,
                       img_size : int,
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       pretrained_model_info : Optional[Munch] = None,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_jigsaw_patches=n_jigsaw_patches,
                          n_classes=n_classes,
                          img_size=img_size,
                          patch_size=8,
                          embed_dim=384,
                          depth=12,
                          num_heads=6,
                          mlp_ratio=4,
                          qkv_bias=True,
                          verbose=verbose,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_model_info is None:
            raise Exception('Requested Pretrained Model, but did not provide Pretrained path and info')
        load_pretrained_weights(model, pretrained_model_info, img_size=img_size, verbose=verbose)
    del model.head
    return model

@register_model
def MultiTaskDeiT_patch16_base(do_jigsaw : bool, 
                       do_coloring : bool, 
                       do_classification : bool, 
                       pixel_shuffle : bool, 
                       n_jigsaw_patches : int,
                       n_classes : int,
                       img_size : int,
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       pretrained_model_info : Optional[Munch] = None,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_jigsaw_patches=n_jigsaw_patches,
                          n_classes=n_classes,
                          img_size=img_size,
                          patch_size=16,
                          embed_dim=768,
                          depth=12,
                          num_heads=12,
                          mlp_ratio=4,
                          qkv_bias=True,
                          verbose=verbose,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_model_info is None:
            raise Exception('Requested Pretrained Model, but did not provide Pretrained path and info')
        load_pretrained_weights(model, pretrained_model_info, img_size=img_size, verbose=verbose)
    del model.head
    return model

@register_model
def MultiTaskDeiT_patch16_base(do_jigsaw : bool, 
                       do_coloring : bool, 
                       do_classification : bool, 
                       pixel_shuffle : bool, 
                       n_jigsaw_patches : int,
                       n_classes : int,
                       img_size : int,
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       pretrained_model_info : Optional[Munch] = None,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_jigsaw_patches=n_jigsaw_patches,
                          n_classes=n_classes,
                          img_size=img_size,
                          patch_size=16,
                          embed_dim=768,
                          depth=12,
                          num_heads=12,
                          mlp_ratio=4,
                          qkv_bias=True,
                          verbose=verbose,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_model_info is None:
            raise Exception('Requested Pretrained Model, but did not provide Pretrained path and info')
        load_pretrained_weights(model, pretrained_model_info, img_size=img_size, verbose=verbose)
    del model.head
    return model

@register_model
def MultiTaskDeiT_patch18_base(do_jigsaw : bool, 
                       do_coloring : bool, 
                       do_classification : bool, 
                       pixel_shuffle : bool, 
                       n_jigsaw_patches : int,
                       n_classes : int,
                       img_size : int,
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       pretrained_model_info : Optional[Munch] = None,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_jigsaw_patches=n_jigsaw_patches,
                          n_classes=n_classes,
                          img_size=img_size,
                          patch_size=8,
                          embed_dim=768,
                          depth=12,
                          num_heads=12,
                          mlp_ratio=4,
                          qkv_bias=True,
                          verbose=verbose,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_model_info is None:
            raise Exception('Requested Pretrained Model, but did not provide Pretrained path and info')
        load_pretrained_weights(model, pretrained_model_info, img_size=img_size, verbose=verbose)
    del model.head
    return model
