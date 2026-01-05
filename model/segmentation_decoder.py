import torch
import torch.nn as nn
from utils.misc import dice_loss
from utils.film import FiLM

class SegmentationDecoder(nn.Module):
    def __init__(
        self,
        vis_dim: int,        # visual feature dim (e.g. 768)
        text_dim: int,       # text embedding dim (e.g. 512)
        hidden_dims=(256, 128, 64),
        output_dim=1,
    ):
        super().__init__()

        # Project visual features
        self.vis_proj = nn.Sequential(
            nn.Conv2d(vis_dim, hidden_dims[0], kernel_size=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )

        # FiLM layers (multi-scale conditioning)
        self.film1 = FiLM(text_dim, hidden_dims[0])
        self.film2 = FiLM(text_dim, hidden_dims[1])
        self.film3 = FiLM(text_dim, hidden_dims[2])

        # Decoder (upsampling)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[1],
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[2],
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(inplace=True),
        )

        # Final mask head
        self.mask_head = nn.Conv2d(hidden_dims[2], output_dim, kernel_size=1)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, image_feats: torch.Tensor, text_emb: torch.Tensor):
        """
        image_feats: [B, C, H, W]  ( ViT patch features)
        text_emb:    [B, D_text]   (pooled CLIP text embedding)
        """

        x = self.vis_proj(image_feats)
        x = self.film1(x, text_emb)

        x = self.decoder1(x)
        x = self.film2(x, text_emb)

        x = self.decoder2(x)
        x = self.film3(x, text_emb)

        mask = self.mask_head(x)
        return mask

    def get_loss(self, pred_mask, gt_mask):
        return self.bce(pred_mask, gt_mask) + dice_loss(pred_mask, gt_mask)