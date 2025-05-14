import torch
import torch.nn as nn

class JigsawHead(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(JigsawHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_patches + 4),  # Output dimension: num_patches * 2
        )
    
    def forward(self, x):
        return self.head(x).squeeze(-1)

class JigsawPositionHead(nn.Module):
    def __init__(self, embed_dim, n_jigsaw_patches):
        super(JigsawPositionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_jigsaw_patches**2),  
        )
    
    def forward(self, x):
        return self.head(x).squeeze(-1)
    
class JigsawRotationHead(nn.Module):
    def __init__(self, embed_dim):
        super(JigsawRotationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),  # Output dimension: num_patches * 2
        )
    
    def forward(self, x):
        return self.head(x).squeeze(-1)
    

class JigsawMultiHead(nn.Module):
    def __init__(self, embed_dim, n_jigsaw_patches, hidden_dim=None, dropout=0.1):
        super().__init__()

        h = hidden_dim or embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.stem = nn.Sequential(
            nn.Linear(embed_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, embed_dim),
            nn.GELU(),
        )
        self.pos_head = nn.Linear(embed_dim, n_jigsaw_patches**2)
        self.rot_head = nn.Linear(embed_dim, 4)

    def forward(self, x):
        # x: (B,n_jigsaw_patches,D)
        z = self.stem(self.norm(x))
        return self.pos_head(z), self.rot_head(z)
