from dataclasses import dataclass
import sys
import os
import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer, _cfg
from .coloring_decoder import ColorizationDecoder, ColorizationDecoderPixelShuffle
from .jigsaw_head import JigsawHead, JigsawPositionHead, JigsawRotationHead, JigsawMultiHead
from munch import Munch
from collections import defaultdict
import torch.nn as nn



class MultiTaskDeiT(VisionTransformer):
    def __init__(self, 
                 n_classes, 
                 do_jigsaw, 
                 do_coloring, 
                 do_classification, 
                 jigsaw_cfg, 
                 pixel_shuffle_cfg,
                 verbose = False,
                 *args, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.num_patches = self.patch_embed.num_patches
        self.do_jigsaw = do_jigsaw
        self.do_coloring = do_coloring
        self.do_classification = do_classification
        self.use_cls_embeds = jigsaw_cfg.use_cls_embeds
        
        if self.do_jigsaw:
            
            self.n_jigsaw_patches = jigsaw_cfg.n_jigsaw_patches

            if self.num_patches % (self.n_jigsaw_patches ** 2) != 0:
                raise Exception('JIGSAW PATCH SIZE NOT DIVISIBLE')
            self.per_jigsaw_patches = self.num_patches // (self.n_jigsaw_patches ** 2)
            self.M = self.n_jigsaw_patches ** 2
            
            if jigsaw_cfg.use_cls_embeds:
                self.single_jigsaw_head = JigsawMultiHead(embed_dim=self.embed_dim,
                                                          n_jigsaw_patches=jigsaw_cfg.n_jigsaw_patches)
                
                self.jigsaw_tokens =  nn.Parameter(torch.zeros(1, self.M, self.embed_dim))   # build the jigsaw tokens
                nn.init.trunc_normal_(self.jigsaw_tokens, std=.02) 
                self.jigsaw_pos_embed = nn.Parameter(torch.zeros(1, self.M, self.embed_dim))
                nn.init.trunc_normal_(self.jigsaw_pos_embed, std=.02)
                

            else:
                self.pos_head = JigsawPositionHead(embed_dim=self.embed_dim, n_jigsaw_patches=self.n_jigsaw_patches)
                self.rot_head = JigsawRotationHead(embed_dim=self.embed_dim)
            
            
        if self.do_coloring and not pixel_shuffle_cfg.do:
            self.coloring_decoder = ColorizationDecoder(embed_dim=self.embed_dim)
        if self.do_coloring and pixel_shuffle_cfg.do:
            self.coloring_decoder = ColorizationDecoderPixelShuffle(embed_dim=self.embed_dim, 
                                                                    total_upscale_factor=self.patch_embed.patch_size[0], 
                                                                    out_channels=2,
                                                                    smoothing=pixel_shuffle_cfg.smoothing,
                                                                    upscale_steps = pixel_shuffle_cfg.steps)
            
        if self.do_classification:
            self.class_head = torch.nn.Linear(self.head.in_features, n_classes)  

        if verbose:
            print('='*100)
            print('MODEL DESCRIPTION')
            print(f'Active Heads:')
            print(f'\tColoring: {self.do_coloring}')
            print(f'\tClassification: {self.do_classification}')
            print(f'\tJigsaw: {self.do_jigsaw}')
            print(f'Image Size: {self.patch_embed.img_size}')
            print(f'Number of Patches: {self.num_patches}')
            print(f'Patch Size: {self.patch_embed.patch_size}')
            if self.do_jigsaw:
                print(f'Number of Jigsaw Patches per side: {self.n_jigsaw_patches}')
                print(f'Number of ViT patches per Jigsaw Patch: {self.per_jigsaw_patches}')
                print(f'Positional Embeddings shape: {self.pos_embed.shape}')
                print(f'Positional Embeddings shape: {self.jigsaw_pos_embed.shape}')
                print(f'Using jigsaw cls tokens: {jigsaw_cfg.use_cls_embeds}')
            if self.do_coloring:
                print(f'Pixel Shuffle Active: {pixel_shuffle_cfg.do}')
                if pixel_shuffle_cfg.do:
                    print(f'Pixel Shuffle Smoothing Active: {pixel_shuffle_cfg.smoothing}')
                    print(f'Pixel Shuffle Upsampling Steps: {pixel_shuffle_cfg.steps}')
                                    
    def forward_jigsaw(self, x):
        
        x = self.patch_embed(x)
        
        if self.use_cls_embeds:
            
            x = x + self.pos_embed[:, 1:, :]       # add patch pos embeddings
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            
            jigsaw_tokens = self.jigsaw_tokens + self.jigsaw_pos_embed
            jigsaw_tokens = jigsaw_tokens.expand(x.shape[0], -1, -1)
            
            
            x = torch.cat((cls_tokens, jigsaw_tokens, x), dim=1)
            
            x = self.pos_drop(x)
            x = self.blocks(x)
            x = self.norm(x)
            pos_logits, rot_logits = self.single_jigsaw_head(x[:, 1:1+self.M, :])
            return pos_logits, rot_logits
        
        
        x = x + self.pos_embed[:, 1:, :]       # add patch pos embeddings
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        B, N, D = x.shape
        x = x[:, 1:].reshape(B, self.n_jigsaw_patches ** 2, self.per_jigsaw_patches, D).mean(dim=2)
        
        pos_logits = self.pos_head(x)
        rot_logits = self.rot_head(x)
        #print(self.device)

        return pos_logits, rot_logits

    def forward_cls(self, x): 
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.class_head(x[:, 0])
        return x
    
    def forward_denoising_coloring(self, x): 

        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 1:]  # remove cls token
        x = self.coloring_decoder(x)
        return x
    
    def forward_jigsaw_inference(self, x):
        
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_pos_embed = self.pos_embed[:, :1, :]
        cls_tokens = cls_tokens + cls_pos_embed
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw_head(x[:, 1:])
        
        # x is the vector of predictions per jigsaw patch
        return x
    
    def forward_coloring_inference(self, x):
        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 1:]  
        x = self.coloring_decoder(x)
        return x
        
    def forward_inference(self, x):
        positions = self.forward_jigsaw_inference(x)
        x = reconstruct_image(x, positions)            # TO BE DEFINED
        x = self.forward_coloring_inference(x)
        return x
        

    def forward(self, x, mode='standard'):
        # x is a Munch dict with image_classification, image_colorization and image_jigsaw
        
        self.device = next(self.parameters()).device
        if mode == 'reconstruction':
            return self.forward_inference(x)
            
        out = Munch()
        if self.do_classification:
            pred_cls = self.forward_cls(x.image_classification)
            out.pred_cls = pred_cls
        if self.do_jigsaw:
            pred_jigsaw = self.forward_jigsaw(x.image_jigsaw)
            out.pred_jigsaw_pos, out.pred_jigsaw_rot = pred_jigsaw
        if self.do_coloring:
            pred_coloring = self.forward_denoising_coloring(x.image_colorization)
            out.pred_coloring = pred_coloring
        return out
            
    

    def count_params_by_block(self):
        block_param_counts = defaultdict(int)
        total_trainable = 0
        total_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                top_block = name.split('.')[0]
                block_param_counts[top_block] += param.numel()
                total_trainable += param.numel()

        lines = []
        lines.append("Trainable parameter counts by block:\n")
        for block, count in block_param_counts.items():
            lines.append(f"{block:<20}: {count:,} parameters")

        lines.append(f"\nTotal trainable parameters: {total_trainable:,}")
        lines.append(f"Total parameters (including frozen): {total_params:,}")

        return "\n".join(lines)






# if __name__ == "__main__":
#     main()