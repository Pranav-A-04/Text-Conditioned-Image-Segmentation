import os
import numpy as np
import torch
import timm
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataloader import CracksAndDrywallDataloader
from model.segmentation_decoder import SegmentationDecoder
from utils.prompts import PROMPTS
import argparse

parser = argparse.ArgumentParser(description="Train Text-Conditioned Image Segmentation Model")

parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--save_interval', type=int, default=10, help='Interval (in epochs) to save model checkpoints')
parser.add_argument('--cracks_path', type=str, help='Path to cracks dataset')
parser.add_argument('--drywall_path', type=str, help='Path to drywall dataset')
args = parser.parse_args()

torch.manual_seed(args.seed)

train_dataset = CracksAndDrywallDataloader(
    cracks_path=os.path.join(args.cracks_path, "train"),
    drywall_path=os.path.join(args.drywall_path, "train"),
    prompts=PROMPTS
)

val_dataset = CracksAndDrywallDataloader(
    cracks_path=os.path.join(args.cracks_path, "valid"),
    drywall_path=os.path.join(args.drywall_path, "valid"),
    prompts=PROMPTS
)

train_loader = DataLoader(
    train_dataset,
    args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

val_loader = DataLoader(
    val_dataset,
    args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

# Fetch pre-trained ViT and CLIP models
vit = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
)
# Remove classification head
vit.reset_classifier(0)
vit = vit.to(args.device)
vit.eval()

# Load CLIP model
clip_model, preprocess = clip.load('ViT-B/32', device=args.device)
clip_model.eval()

# Freeze pre-trained model parameters
for p in vit.parameters():
    p.requires_grad = False

for p in clip_model.parameters():
    p.requires_grad = False

# Initialize segmentation decoder
decoder = SegmentationDecoder(
    vis_token_dim=768,
    text_token_dim=512,
    input_dim=512,
    output_dim=1
).to(args.device)

decoder = torch.nn.DataParallel(decoder)

# Define optimizer
optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4)

def train(model, dataloader, optimizer):
    losses = []
    for batch in tqdm(dataloader):
        images = batch['image'].to(args.device)
        masks = batch['mask'].to(args.device)
        prompts = batch['prompt']

        # Obtain visual and text embeddings from pre-trained models
        with torch.no_grad():
            vis_emb = vit.forward_features(images)
            vis_emb = vis_emb[:, 1:, :]  # Remove CLS token
            assert vis_emb.ndim == 3, f"Expected [B, N, D], got {vis_emb.shape}"

            text_tokens = clip.tokenize(prompts).to(args.device)
            # Use CLIP token-level transformer outputs so attention has multiple keys
            token_emb = clip_model.token_embedding(text_tokens).type(clip_model.dtype)  # [B, L, D]
            x = token_emb + clip_model.positional_embedding.type(clip_model.dtype)
            x = x.permute(1, 0, 2)  # [L, B, D]
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # [B, L, D]
            text_emb = clip_model.ln_final(x).float()

        optimizer.zero_grad()
        pred_masks = model(vis_emb, text_emb)

        # If the sampled prompt doesn't match the image class, supervise with an empty (zero) mask
        is_positive = batch['is_positive'].to(device=args.device, dtype=torch.bool)
        target_masks = masks.clone()
        if not is_positive.all():
            target_masks[~is_positive] = 0.0

        loss = model.module.get_loss(pred_masks, target_masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_loss = np.mean(losses)
    return avg_loss
    

def validate(model, dataloader):
    val_losses = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(args.device)
            masks = batch['mask'].to(args.device)
            prompts = batch['prompt']

            vis_emb = vit.forward_features(images)
            vis_emb = vis_emb[:, 1:, :] # remove cls token
            assert vis_emb.ndim == 3, f"Expected [B, N, D], got {vis_emb.shape}"
            text_tokens = clip.tokenize(prompts).to(args.device)
            token_emb = clip_model.token_embedding(text_tokens).type(clip_model.dtype)  # [B, L, D]
            x = token_emb + clip_model.positional_embedding.type(clip_model.dtype)
            x = x.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            text_emb = clip_model.ln_final(x).float()

            pred_masks = model(vis_emb, text_emb)

            # negative-prompt supervision: if prompt doesn't match, target mask is empty
            is_positive = torch.tensor(batch['is_positive'], dtype=torch.bool, device=args.device)
            target_masks = masks.clone()
            if not is_positive.all():
                target_masks[~is_positive] = 0.0

            loss = model.module.get_loss(pred_masks, target_masks)
            val_losses.append(loss.item())
    avg_loss = np.mean(val_losses)
    return avg_loss


if __name__ == "__main__":
    
    # train model
    for epoch in range(args.num_epochs):
        decoder.train()
        loss = train(decoder, train_loader, optimizer)
        print(f"Epoch {epoch+1}/{args.num_epochs} | Training Loss: {loss}")
        
        # validate model and save ckpt
        if epoch % int(args.save_interval) == 0:
            decoder.eval()
            val_loss = validate(decoder, val_loader)
            print(f"Epoch {epoch+1}/{args.num_epochs} | Validation Loss: {val_loss}")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(decoder.state_dict(), os.path.join("checkpoints", f"segmentation_decoder_epoch_{epoch+1}.pth"))    