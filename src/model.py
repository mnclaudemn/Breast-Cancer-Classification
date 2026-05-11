import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------
# 1. CNN Backbone (ResNet50)
# ----------------------------
class ResNetBackbone(nn.Module):
    """
    CNN feature extractor using pretrained ResNet50.
    Removes classification head and global pooling.
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove avgpool and fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # (B, 3, H, W) → (B, 2048, H/32, W/32)
        return self.backbone(x)


# ----------------------------
# 2. Feature Tokenization
# ----------------------------
class FeatureTokenizer(nn.Module):
    """
    Converts CNN feature maps into sequence tokens.
    """
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)       # (B, C, N)
        x = x.transpose(1, 2)  # (B, N, C)
        return x


# ----------------------------
# 3. Positional Encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings for feature tokens.
    """
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
  
    def forward(self, x):
        # safety: prevent overflow if tokens > 1000
        seq_len = x.size(1)
        if seq_len > self.pos_embed.size(1):
            raise ValueError("Sequence length exceeds positional encoding limit")

        return x + self.pos_embed[:, :seq_len, :]

# ----------------------------
# 4. Hybrid CNN-Transformer Model
# ----------------------------
class ResNetTransformer(nn.Module):
    """
    Hybrid architecture:
    - ResNet50 extracts spatial features
    - Features are converted into tokens
    - Transformer learns global dependencies
    - CLS token used for classification
    """

    def __init__(self, num_classes=2, dim=512, num_layers=4, num_heads=8):
        super().__init__()

        # CNN backbone
        self.backbone = ResNetBackbone()
        self.tokenizer = FeatureTokenizer()

        # Projection from CNN feature space → transformer space
        self.proj = nn.Linear(2048, dim)

        # CLS token (global representation)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Positional encoding
        self.pos_enc = PositionalEncoding(dim)

        # Transformer encoder
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

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # 1. CNN feature extraction
        x = self.backbone(x)          # (B, 2048, H, W)

        # 2. Tokenization
        x = self.tokenizer(x)         # (B, N, 2048)

        # 3. Projection
        x = self.proj(x)              # (B, N, dim)

        # 4. CLS token addition
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)

        # 5. Positional encoding
        x = self.pos_enc(x)

        # 6. Transformer encoding
        x = self.transformer(x)

        # 7. CLS token extraction
        x = x[:, 0]

        # 8. Classification head
        x = self.norm(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
