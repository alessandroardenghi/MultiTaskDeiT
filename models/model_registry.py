import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import load_pretrained
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from .full_model import MultiTaskDeiT
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
def MultiTaskDeiT_tiny(do_jigsaw, 
                       do_coloring, 
                       do_classification, 
                       pixel_shuffle, 
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_classes=20,
                          img_size=224,
                          patch_size=16,
                          embed_dim=192,
                          depth=12,
                          num_heads=3,
                          mlp_ratio=4,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
    
        # Track layers that are updated and not updated
        updated_layers = []
        not_updated_layers = []
        to_remove = ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']

        for k in to_remove:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        for k in model_dict.keys():
            if k in pretrained_dict and k not in to_remove:
                updated_layers.append(k)
            else:
                not_updated_layers.append(k)
        
        # Update the model's state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        del model.head          # MIGHT GIVE PROBLEMS
        
        # Print updated and not updated layers
        if verbose:
            print("\nUpdated layers:")
            for layer in updated_layers:
                print(f"  {layer}")
            
            print("\nNot updated layers:")
            for layer in not_updated_layers:
                print(f"  {layer}")
        
    return model

@register_model
def MultiTaskDeiT_small(do_jigsaw, do_coloring, 
                        do_classification, 
                        pixel_shuffle,
                        pretrained_cfg,
                        pretrained_cfg_overlay,
                        cache_dir,
                        verbose=False,
                        pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_classes=20,
                          img_size=224,
                          patch_size=16,
                          embed_dim=384,
                          depth=12,
                          num_heads=6,
                          mlp_ratio=4,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        # Track layers that are updated and not updated
        updated_layers = []
        not_updated_layers = []
        to_remove = ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']

        for k in to_remove:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        for k in model_dict.keys():
            if k in pretrained_dict and k not in to_remove:
                updated_layers.append(k)
            else:
                not_updated_layers.append(k)
        
        # Update the model's state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if verbose:
            # Print updated and not updated layers
            print("\nUpdated layers:")
            for layer in updated_layers:
                print(f"  {layer}")
            
            print("\nNot updated layers:")
            for layer in not_updated_layers:
                print(f"  {layer}")
                
    return model

@register_model
def MultiTaskDeiT_base(do_jigsaw, 
                       do_coloring, 
                       do_classification, 
                       pixel_shuffle, 
                       pretrained_cfg,
                       pretrained_cfg_overlay,
                       cache_dir,
                       verbose=False,
                       pretrained=False):
    
    model = MultiTaskDeiT(do_jigsaw=do_jigsaw, 
                          do_coloring=do_coloring, 
                          do_classification=do_classification,
                          pixel_shuffle=pixel_shuffle,
                          n_classes=20,
                          img_size=224,
                          patch_size=16,
                          embed_dim=768,
                          depth=12,
                          num_heads=12,
                          mlp_ratio=4,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
             url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        
        # Track layers that are updated and not updated
        updated_layers = []
        not_updated_layers = []
        to_remove = ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']
        
        for k in to_remove:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        for k in model_dict.keys():
            if k in pretrained_dict and k not in to_remove:
                updated_layers.append(k)
            else:
                not_updated_layers.append(k)
        
        # Update the model's state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if verbose:
            print("\nUpdated layers:")
            for layer in updated_layers:
                print(f"  {layer}")
            
            print("\nNot updated layers:")
            for layer in not_updated_layers:
                print(f"  {layer}")
    return model