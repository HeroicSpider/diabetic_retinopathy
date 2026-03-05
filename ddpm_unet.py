import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Maps continuous timestep t to a high-dimensional vector."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        device = t.device
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class SelfAttentionBlock(nn.Module):
    """Self-attention for spatial feature maps at lower resolutions."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attention(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return h + residual


class ConditionalResidualBlock(nn.Module):
    """ResNet block with time and class embedding injection."""
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.act = nn.SiLU()

    def forward(self, x, cond_embedding):
        residual = self.skip(x)
        h = self.conv1(self.act(self.norm1(x)))
        cond = self.cond_proj(cond_embedding).unsqueeze(-1).unsqueeze(-1)
        h = h + cond
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, use_attention=False):
        super().__init__()
        self.res1 = ConditionalResidualBlock(in_channels, out_channels, cond_dim)
        self.res2 = ConditionalResidualBlock(out_channels, out_channels, cond_dim)
        self.attn = SelfAttentionBlock(out_channels) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, cond):
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, cond_dim, use_attention=False):
        super().__init__()
        # Upsample FIRST to match skip spatial dimensions
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.res1 = ConditionalResidualBlock(in_channels + skip_channels, out_channels, cond_dim)
        self.res2 = ConditionalResidualBlock(out_channels, out_channels, cond_dim)
        self.attn = SelfAttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        x = self.attn(x)
        return x


class ClassConditionalUNet(nn.Module):
    """
    128×128 class-conditional U-Net for DDPM.
    Attention at 16×16 and 8×8 only.
    """
    def __init__(self, c_in=3, c_out=3, base_channels=64, num_classes=5):
        super().__init__()
        self.cond_dim = base_channels * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        # Class embedding (5 classes + 1 null token for CFG)
        self.class_emb = nn.Embedding(num_classes + 1, self.cond_dim)

        # Initial projection
        self.init_conv = nn.Conv2d(c_in, base_channels, kernel_size=3, padding=1)

        # Downsampling: 128 → 64 → 32 → 16 → 8
        self.down1 = DownBlock(base_channels,     base_channels * 2, self.cond_dim, use_attention=False)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, self.cond_dim, use_attention=False)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, self.cond_dim, use_attention=False)  # 32→16, no attn
        self.down4 = DownBlock(base_channels * 8, base_channels * 8, self.cond_dim, use_attention=True)   # 16→8, ATTN at 16×16

        # Bottleneck at 8×8
        self.bot1 = ConditionalResidualBlock(base_channels * 8, base_channels * 8, self.cond_dim)
        self.bot_attn = SelfAttentionBlock(base_channels * 8)  # ATTN at 8×8
        self.bot2 = ConditionalResidualBlock(base_channels * 8, base_channels * 8, self.cond_dim)

        # Upsampling: 8 → 16 → 32 → 64 → 128
        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 8, self.cond_dim, use_attention=True)   # ATTN at 16×16
        self.up2 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, self.cond_dim, use_attention=False)  # 32×32, no attn
        self.up3 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, self.cond_dim, use_attention=False)
        self.up4 = UpBlock(base_channels * 2, base_channels * 2, base_channels,     self.cond_dim, use_attention=False)

        # Output
        self.out_norm = nn.GroupNorm(min(32, base_channels), base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, c_out, kernel_size=3, padding=1)

    def forward(self, x, t, class_labels):
        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(class_labels)
        cond = t_emb + c_emb

        x = self.init_conv(x)

        x, skip1 = self.down1(x, cond)
        x, skip2 = self.down2(x, cond)
        x, skip3 = self.down3(x, cond)
        x, skip4 = self.down4(x, cond)

        x = self.bot1(x, cond)
        x = self.bot_attn(x)
        x = self.bot2(x, cond)

        x = self.up1(x, skip4, cond)
        x = self.up2(x, skip3, cond)
        x = self.up3(x, skip2, cond)
        x = self.up4(x, skip1, cond)

        x = self.out_act(self.out_norm(x))
        return self.out_conv(x)
