import torch
import torch.nn as nn
import torch.nn.functional as F #含有激活函数，损失函数等模块

# 定义 Patch Embedding 层；将图像划分为patch并嵌入到更高维度的特征空间中
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#定义一个2D卷积层，将输入的图像快映射到更高维度的嵌入空间，卷积核大小等于patch_size
    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

# 定义 Transformer Encoder 层
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, hidden_dim=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-Head Attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.ln1(x)

        # MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Residual connection
        x = self.ln2(x)

        return x

# 定义 Vision Transformer 模型
class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10, embed_dim=768, num_layers=12, num_heads=8, hidden_dim=2048, dropout=0.1):
        super(SimpleViT, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Stacking Transformer Encoder layers
        self.transformer = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch Embedding
        x = self.patch_embed(x)

        # Concatenate class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer Encoder
        for layer in self.transformer:
            x = layer(x)

        # Classification token
        x = self.ln(x[:, 0])
        x = self.fc(x)

        return x

# 测试模型
if __name__ == "__main__":
    # 随机生成输入
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 resolution

    # 创建模型实例
    model = SimpleViT(num_classes=10)

    # 将模型移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 前向传播
    output = model(input_tensor)

    # 输出模型的预测
    print("Model output:", output)
    print("Output shape:", output.shape)  # 应为 (1, 10) 表示10类分类的输出
