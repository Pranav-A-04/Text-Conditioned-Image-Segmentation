import torch
import torch.nn as nn
from utils.misc import dice_loss

class SegmentationDecoder(nn.Module):
    def __init__(self, vis_token_dim, text_token_dim, input_dim, output_dim=1, hidden_dim=[256, 128, 64, 32]):
        super().__init__()
        self.vis_proj_layer = nn.Linear(vis_token_dim, input_dim)
        self.text_proj_layer = nn.Linear(text_token_dim, input_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)
        self.init_conv = nn.Conv2d(input_dim, hidden_dim[0], kernel_size=3, padding=1)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ) for i in range(len(hidden_dim))
        ])
        
        self.output_deconv = nn.ConvTranspose2d(
            hidden_dim[-1], output_dim, kernel_size=4, stride=2, padding=1
        )
        
        self.activation = nn.Sigmoid()
    
    def get_loss(self, predicted_mask, ground_truth_mask):
        BCELoss = nn.BCELoss()
        bce_loss = BCELoss(predicted_mask, ground_truth_mask)
        dice_loss = dice_loss(predicted_mask, ground_truth_mask)
        
        return bce_loss + dice_loss
    
    def forward(self, vis_tokens, text_tokens):
        # broadcast text tokens
        text_tokens = text_tokens.unsqueeze(1).repeat(1, vis_tokens.size(1), 1)
        
        # Project visual and text tokens
        vis_tokens = self.vis_proj_layer(vis_tokens)
        text_tokens = self.text_proj_layer(text_tokens)
        
        # Apply cross attention
        attended_tokens, _ = self.cross_attention(vis_tokens, text_tokens, text_tokens)
        
        # Initialize with the first convolution
        x = self.init_conv(attended_tokens)
        
        # Apply decoder layers
        for layer in self.decoder:
            x = layer(x)
            
        # Final deconvolution and activation
        x = self.output_deconv(x)
        return self.activation(x)