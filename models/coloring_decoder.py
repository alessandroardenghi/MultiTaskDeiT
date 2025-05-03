import torch
import torch.nn as nn

class ColorizationDecoder(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),  # 14 → 28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # 28 → 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # 56 → 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # 112 → 224
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),                           # 224x224x3 output
            nn.Sigmoid() 
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return self.decoder(x)