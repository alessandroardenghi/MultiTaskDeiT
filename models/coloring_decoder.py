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
    
    

class ColorizationDecoderPixelShuffle(nn.Module):
    def __init__(self, embed_dim=384, total_upscale_factor=16, upscale_steps = 1, out_channels=2, smoothing=False):
        super().__init__()

        self.smoothing = smoothing
        
        if upscale_steps > 2:
            raise Exception('Max 2 upscale steps')
        
        if upscale_steps == 2:
            if total_upscale_factor == 16:
                self.upscale_step1 = 4
                self.upscale_step2 = 4
            elif total_upscale_factor == 8:
                self.upscale_step1 = 2
                self.upscale_step2 = 4
            else:
                raise Exception('Upscale factor not supported')
        
        
        self.upscale_steps = upscale_steps
        
        if upscale_steps == 1:
            self.proj_full = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=out_channels * (total_upscale_factor ** 2),
                kernel_size=3,
                padding=1,
                stride=1,
                padding_mode='reflect'
            )
            
            self.pixel_shuffle_full = nn.PixelShuffle(upscale_factor=total_upscale_factor)
        
        elif upscale_steps == 2:
            
            self.proj1 = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=out_channels * (self.upscale_step1 ** 2),
                kernel_size=3,
                padding=1,
                stride=1,
                padding_mode='reflect'
            )
            
            self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=self.upscale_step1)
            
            self.proj2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * (self.upscale_step2 ** 2),
                kernel_size=3,
                padding=1,
                stride=1,
                padding_mode='reflect'
            )
            self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=self.upscale_step2)

        else:
            raise Exception('At most 2 upscale steps')
        
        self.smooth = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True
        )
        self.activation = nn.Tanh()
        
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # project to sub-pixel channels
        if self.upscale_steps == 1:
            x = self.proj_full(x)              
            x = self.activation(x)
            x = self.pixel_shuffle_full(x)      
            
        elif self.upscale_steps == 2:
            x = self.proj1(x)    
            x = self.activation(x)
            x = self.pixel_shuffle1(x)  
            x = self.proj2(x)   
            x = self.activation(x) 
            x = self.pixel_shuffle2(x)         
        
        if self.smoothing:
            x = self.smooth(x)            
        return x
