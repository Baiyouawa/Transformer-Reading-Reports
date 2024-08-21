import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, x):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        # Window partition
        x = x.view(B, self.input_resolution[0], self.input_resolution[1], C)
        windows = x.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size, self.window_size)
        windows = windows.contiguous().view(-1, self.window_size * self.window_size, C)  # Shape [num_windows*B, window_size*window_size, C]
        
        # Self-attention within windows
        x, _ = self.attn(windows, windows, windows)
        
        # Merge windows back to original size
        x = x.view(B, L, C)
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class SimpleSwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=10, embed_dim=96, num_heads=3):
        super(SimpleSwinTransformer, self).__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.flatten = nn.Flatten(2)
        
        # Swin Transformer Block
        self.swin_block = SwinBlock(embed_dim, (img_size // patch_size, img_size // patch_size), num_heads)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # Shape [B, embed_dim, H//patch_size, W//patch_size]
        x = self.flatten(x)  # Shape [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # Shape [B, num_patches, embed_dim]
        
        # Swin Transformer block
        x = self.swin_block(x)
        
        # Classifier head
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    # Create random input tensor with batch size 1, 3 channels, and 224x224 image size
    input_tensor = torch.randn(1, 3, 224, 224)
    model = SimpleSwinTransformer(num_classes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    output = model(input_tensor)

    print("Model output:", output)
    print("Output shape:", output.shape)
