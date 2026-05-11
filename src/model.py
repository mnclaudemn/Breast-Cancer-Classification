import torch
import torch.nn as nn
import torchvision.models as models
# ----------------------------
# 1. ResNet Backbone
# ----------------------------
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # remove avgpool + fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # (B, 3, H, W) -> (B, 2048, H/32, W/32)
        return self.backbone(x)


# ----------------------------
# 2. Patchify (Flatten spatial dims)
# ----------------------------
class Patchify(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)      # (B, C, N)
        x = x.transpose(1, 2) # (B, N, C)
        return x
# ----------------------------
# 3. Positional Encoding (learnable)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]


# ----------------------------
# 4. Full Model
# ----------------------------
class ResNetTransformer(nn.Module):
    def __init__(self, num_classes=10, dim=512, num_layers=4, num_heads=8):
        super().__init__()

        # CNN backbone
        self.backbone = ResNetBackbone()
        self.patchify = Patchify()

        # projection
        self.proj = nn.Linear(2048, dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # positional encoding
        self.pos_enc = PositionalEncoding(dim)

        # Transformer Encoder (BEST PRACTICE)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # normalization before head
        self.norm = nn.LayerNorm(dim)

        # classifier head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # CNN features
        x = self.backbone(x)      # (B, 2048, H, W)
        x = self.patchify(x)      # (B, N, 2048)

        # projection
        x = self.proj(x)          # (B, N, dim)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)          # (B, 1+N, dim)

        # positional encoding
        x = self.pos_enc(x)

        # transformer
        x = self.transformer(x)  # (B, 1+N, dim)

        # CLS token only
        x = x[:, 0]              # (B, dim)

        # head
        x = self.norm(x)
        x = self.dropout(x)
        out = self.classifier(x)

        return out
