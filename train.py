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

# CLIP image encoder (ViT-B/16)
clip_img_model, preprocess = clip.load("ViT-B/16", device=args.device)
clip_img_model.eval()

# CLIP text encoder (ViT-B/32)
clip_txt_model, _ = clip.load("ViT-B/32", device=args.device)
clip_txt_model.eval()

for p in clip_img_model.parameters():
    p.requires_grad = False

for p in clip_txt_model.parameters():
    p.requires_grad = False

decoder = SegmentationDecoder(
    vis_dim=768,
    text_dim=512,
    output_dim=1,
).to(args.device)

decoder = torch.nn.DataParallel(decoder)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate)

def extract_clip_image_features(images):
    with torch.no_grad():
        images = images.to(dtype=clip_img_model.dtype)

        x = clip_img_model.visual.conv1(images)          # [B, C, H/16, W/16]
        x = x.reshape(x.shape[0], x.shape[1], -1)        # [B, C, N]
        x = x.permute(0, 2, 1)                            # [B, N, C]

        cls_token = clip_img_model.visual.class_embedding.to(x.dtype)
        cls_token = cls_token.to(x.dtype)
        cls_token = cls_token.expand(x.shape[0], 1, -1)

        x = torch.cat([cls_token, x], dim=1)
        x = x + clip_img_model.visual.positional_embedding.to(x.dtype)
        x = clip_img_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = clip_img_model.visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = clip_img_model.visual.ln_post(x)

        patch_tokens = x[:, 1:, :]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)

        patch_tokens = patch_tokens.permute(0, 2, 1).contiguous()
        patch_tokens = patch_tokens.view(B, C, H, W)

    return patch_tokens

def extract_clip_text_embedding(prompts):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(args.device)
        text_emb = clip_txt_model.encode_text(text_tokens)
        text_emb = text_emb.float()   # important for FiLM
    return text_emb

def train(model, dataloader, optimizer):
    losses = []

    for batch in tqdm(dataloader):
        images = batch["image"].to(args.device)
        masks = batch["mask"].to(args.device)
        prompts = batch["prompt"]

        image_feats = extract_clip_image_features(images)
        text_emb = extract_clip_text_embedding(prompts)

        optimizer.zero_grad()
        pred_masks = model(image_feats, text_emb)

        loss = model.module.get_loss(pred_masks, masks)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)
    

def validate(model, dataloader):
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            prompts = batch["prompt"]

            image_feats = extract_clip_image_features(images)
            text_emb = extract_clip_text_embedding(prompts)

            pred_masks = model(image_feats, text_emb)
            loss = model.module.get_loss(pred_masks, masks)
            losses.append(loss.item())

    return np.mean(losses)


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