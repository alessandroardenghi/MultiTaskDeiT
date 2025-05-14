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
            nn.Conv2d(32, 2, kernel_size=1),                           # 224x224x2 output
            nn.Tanh() 
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return self.decoder(x)
    
    
# x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
#         x = self.head(x)
#         x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c')
class ColorizationDecoderPixelShuffle(nn.Module):
    def __init__(self, embed_dim=384, upscale_factor=16, out_channels=2):
        super().__init__()
        # Stabilizing projection: linearly map embed_dim -> out_channels * r^2
        self.proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1,
            stride=1,
            padding_mode='reflect'
        )
        # PixelShuffle for sub-pixel upsampling
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        # # Optional smoothing conv after shuffle (can help reduce artifacts)
        # self.smooth = nn.Conv2d(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     padding=1,
        #     bias=True
        # )
        self.activation = nn.Tanh()  # or nn.Tanh() / identity, depending on range

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        # reshape to (B, C, H, W)
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # project to sub-pixel channels
        x = self.proj(x)              # (B, out_channels*r^2, H, W)
        x = self.activation(x)
        x = self.pixel_shuffle(x)      # (B, out_channels, H*r, W*r)
        #x = self.smooth(x)             # optional smoothing
        return x
