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
    
    def forward(self, vis_tokens, text_tokens):
        # broadcast text tokens
        text_tokens = text_tokens.unsqueeze(1)
        
        # Project visual and text tokens
        vis_tokens = self.vis_proj_layer(vis_tokens)
        text_tokens = self.text_proj_layer(text_tokens)
        
        # Apply cross attention
        attended_tokens, _ = self.cross_attention(vis_tokens, text_tokens, text_tokens)
        
        # Reshape attended tokens to feature map
        B, N, D = attended_tokens.shape
        H = W = int(N ** 0.5)
        attended_tokens = attended_tokens.permute(0, 2, 1).contiguous().view(B, D, H, W)
        
        # Initialize with the first convolution
        x = self.init_conv(attended_tokens)
        
        # Apply decoder layers
        for layer in self.decoder:
            x = layer(x)
            
        # Final deconvolution and activation
        return self.output_deconv(x)