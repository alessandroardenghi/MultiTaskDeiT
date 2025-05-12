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
    