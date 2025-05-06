from dataclasses import dataclass
import sys
import os
import numpy as np
import torch
from utils import add_gaussian_noise, grayscale_weighted_3ch, jigsaw_batch, reconstruct_image
from timm.models.vision_transformer import VisionTransformer, _cfg
from .coloring_decoder import ColorizationDecoder, ColorizationDecoderPixelShuffle
from .jigsaw_head import JigsawHead
from munch import Munch
from collections import defaultdict


class MultiTaskDeiT(VisionTransformer):
    def __init__(self, n_classes, do_jigsaw, do_coloring, do_classification, pixel_shuffle=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_patches = self.patch_embed.num_patches
        self.do_jigsaw = do_jigsaw
        self.do_coloring = do_coloring
        self.do_classification = do_classification
        
        if self.do_jigsaw:
            self.jigsaw_head = JigsawHead(embed_dim=self.embed_dim, num_patches=self.num_patches)
        
        if self.do_coloring and not pixel_shuffle:
            self.coloring_decoder = ColorizationDecoder(embed_dim=self.embed_dim)
        if self.do_coloring and pixel_shuffle:
            self.coloring_decoder = ColorizationDecoderPixelShuffle(embed_dim=self.embed_dim, upscale_factor=16, out_channels=3)
            
        if self.do_classification and n_classes != 1000:
            self.class_head = torch.nn.Linear(self.head.in_features, n_classes)  

    def forward_jigsaw(self, x):
        ## NEED TO WRITE THE FUNCTION TO JIGSAW
        x = add_gaussian_noise(x)
        x = grayscale_weighted_3ch(x)
        x, pos_vector, rot_vector = jigsaw_batch(x, n_patches=int(self.num_patches**0.5))
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
        #print(self.device)
        pos_vector = pos_vector.to(self.device)
        rot_vector = rot_vector.to(self.device)
        return x, pos_vector, rot_vector

    def forward_cls(self, x): 
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)
        x = self.norm(x)
        x = self.class_head(x[:, 0])
        return x
    
    def forward_denoising_coloring(self, x): 
        x = add_gaussian_noise(x)
        x = grayscale_weighted_3ch(x)
        x = self.patch_embed(x)
    
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

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
        self.device = next(self.parameters()).device
        if mode == 'reconstruction':
            return self.forward_inference(x)
            
        out = Munch()
        if self.do_classification:
            pred_cls = self.forward_cls(x)
            out.pred_cls = pred_cls
        if self.do_jigsaw:
            pred_jigsaw, pos_vector, rot_vector = self.forward_jigsaw(x)
            out.pred_jigsaw = pred_jigsaw
            out.pos_vector = pos_vector
            out.rot_vector = rot_vector
        if self.do_coloring:
            pred_coloring = self.forward_denoising_coloring(x)
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

        print("Trainable parameter counts by block:\n")
        for block, count in block_param_counts.items():
            print(f"{block:<20}: {count:,} parameters")

        print(f"\nTotal trainable parameters: {total_trainable:,}")
        print(f"Total parameters (including frozen): {total_params:,}")



# if __name__ == "__main__":
#     main()