from dataclasses import dataclass
import os
import numpy as np
import torch
from utils import add_gaussian_noise, grayscale_weighted_3ch, jigsaw_batch
from timm.models.vision_transformer import VisionTransformer, _cfg
from .coloring_decoder import ColorizationDecoder
from .jigsaw_head import JigsawHead
from munch import Munch


class MultiTaskDeiT(VisionTransformer):
    def __init__(self, do_jigsaw, do_coloring, do_classification, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_patches = self.patch_embed.num_patches
        self.do_jigsaw = do_jigsaw
        self.do_coloring = do_coloring
        self.do_classification = do_classification

        if self.do_jigsaw:
            self.jigsaw_head = JigsawHead(embed_dim=self.embed_dim, num_patches=self.num_patches)
        
        if self.do_coloring:
            self.coloring_decoder = ColorizationDecoder(embed_dim=self.embed_dim)

    def forward_jigsaw(self, x):
        ## NEED TO WRITE THE FUNCTION TO JIGSAW
        x = add_gaussian_noise(x)
        x = grayscale_weighted_3ch(x)
        x, self.pos_vector, self.rot_vector = jigsaw_batch(x, n_patches=self.num_patches)
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_pos_embed = self.pos_embed[:, :1, :]
        cls_tokens = cls_tokens + cls_pos_embed
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])       # TO BE DEFINED
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
        x = self.head(x[:, 0])
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

    def forward(self, x):
        out = Munch(labels=(self.pos_vector, self.rot_vector))

        if self.do_classification:
            pred_cls = self.forward_cls(x)
            out.pred_cls = pred_cls
        if self.do_jigsaw:
            pred_jigsaw, pos_vector, rot_vector = self.forward_jigsaw(x)
            out.pred_jigsaw = pred_jigsaw
        if self.do_coloring:
            pred_coloring = self.forward_denoising_coloring(x)
            out.pred_coloring = pred_coloring
        return out


def main():
    # Example usage
    model = MultiTaskDeiT(
        do_jigsaw=True,
        do_coloring=True,
        do_classification=True,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=86,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=None
    )
    print(model)

if __name__ == "__main__":
    main()