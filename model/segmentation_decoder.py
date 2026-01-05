import torch
import torch.nn as nn
from utils.misc import dice_loss

class SegmentationDecoder(nn.Module):
    def __init__(self, vis_token_dim, text_token_dim, input_dim, output_dim=1, hidden_dim=[256, 128, 64, 32]):
        super().__init__()
        self.vis_proj_layer = nn.Linear(vis_token_dim, input_dim)
        self.text_proj_layer = nn.Linear(text_token_dim, input_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=8,
            batch_first=True
        )
        # Pre-norm layers and MLP for post-attention processing
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        mlp_hidden = input_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, input_dim),
            nn.Dropout(0.1),
        )
        
        self.init_conv = nn.Conv2d(input_dim, hidden_dim[0], kernel_size=3, padding=1)
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim[i+1]),
                nn.ReLU()
            ) for i in range(len(hidden_dim)-1)
        ])
        
        self.output_deconv = nn.ConvTranspose2d(
            hidden_dim[-1], output_dim, kernel_size=4, stride=2, padding=1
        )
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def get_loss(self, predicted_mask, ground_truth_mask):
        bce_loss = self.bce_loss(predicted_mask, ground_truth_mask)
        dice = dice_loss(predicted_mask, ground_truth_mask)
        
        return bce_loss + dice
    
    def forward(self, vis_tokens, text_tokens, return_attn: bool = False):
        """Forward with pre-norm Attention -> MLP Transformer block.

        Args:
            vis_tokens: [B, N, D_vis]
            text_tokens: [B, L, D_text] (token-level features)
            return_attn: if True, also return attention weights (B, N, L)
        Returns:
            pred_mask [B, 1, H, W] (and attn weights if requested)
        """
        # Project visual and text tokens into a common dimension
        vis_tokens = self.vis_proj_layer(vis_tokens)  # [B, N, D]
        text_tokens = self.text_proj_layer(text_tokens)  # [B, L, D]

        q = self.norm1(vis_tokens)
        attn_out, attn_weights = self.cross_attention(q, text_tokens, text_tokens, need_weights=True)
        # Residual for attention sublayer
        x = vis_tokens + attn_out  # [B, N, D]

        x2 = self.mlp(self.norm2(x))
        # Residual for MLP sublayer
        x = x + x2  # [B, N, D]

        # Reshape to spatial feature map
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(B, D, H, W)

        # Initialize with the first convolution and decode
        x = self.init_conv(x)
        for layer in self.decoder:
            x = layer(x)

        pred_mask = self.output_deconv(x)
        if return_attn:
            return pred_mask, attn_weights
        return pred_mask