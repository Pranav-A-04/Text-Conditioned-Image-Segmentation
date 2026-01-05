import argparse
import torch
import timm
import clip
import numpy as np
from PIL import Image
from torchvision import transforms

from model.segmentation_decoder import SegmentationDecoder

parser = argparse.ArgumentParser("Text-conditioned segmentation inference")

parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt, e.g. 'segment crack'")
parser.add_argument("--ckpt", type=str, required=True, help="Path to trained decoder checkpoint")
parser.add_argument("--output", type=str, default="output_mask.png", help="Output mask filename")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

args = parser.parse_args()
device = args.device

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# CLIP image encoder (ViT-B/16)
clip_img_model, preprocess = clip.load("ViT-B/16", device=args.device)
clip_img_model = clip_img_model.float()
clip_img_model.eval()

# CLIP text encoder (ViT-B/32)
clip_txt_model, _ = clip.load("ViT-B/32", device=args.device)
clip_txt_model = clip_txt_model.float()
clip_txt_model.eval()

decoder = SegmentationDecoder(
    vis_dim=768,
    text_dim=512,
    output_dim=1,
).to(args.device)
state_dict = torch.load(args.ckpt, map_location=device)

def extract_clip_image_features(images):
    with torch.no_grad():
        x = clip_img_model.visual.conv1(images)          # [B, C, H/16, W/16]
        x = x.reshape(x.shape[0], x.shape[1], -1)        # [B, C, N]
        x = x.permute(0, 2, 1)                            # [B, N, C]

        cls_token = clip_img_model.visual.class_embedding
        cls_token = cls_token.to(x.dtype)
        cls_token = cls_token.expand(x.shape[0], 1, -1)

        x = torch.cat([cls_token, x], dim=1)              # [B, N+1, C]
        x = x + clip_img_model.visual.positional_embedding
        x = clip_img_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)                             # [N+1, B, C]
        x = clip_img_model.visual.transformer(x)
        x = x.permute(1, 0, 2)                             # [B, N+1, C]

        x = clip_img_model.visual.ln_post(x)

        patch_tokens = x[:, 1:, :]                         # remove CLS
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).contiguous()
        patch_tokens = patch_tokens.view(B, C, H, W)      # [B, C, H, W]

    return patch_tokens

def extract_clip_text_embedding(prompts):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(args.device)
        text_emb = clip_txt_model.encode_text(text_tokens)
        text_emb = text_emb.float()   # important for FiLM
    return text_emb

# Strip "module." prefix if present
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[len("module."):]] = v
    else:
        new_state_dict[k] = v

decoder.load_state_dict(new_state_dict)
decoder.eval().to(device)

orig_image = Image.open(args.image_path).convert("RGB")
orig_w, orig_h = orig_image.size

image = image_transform(orig_image).unsqueeze(0).to(device)  # [1,3,224,224]

image_feats = extract_clip_image_features(image)  # [1,768,H,W]
text_emb = extract_clip_text_embedding([args.prompt])  # [1,512]

with torch.no_grad():
    logits = decoder(image_feats, text_emb)      # [1,1,H,W]
    probs = torch.sigmoid(logits)
    print("Prob min/max:", probs.min().item(), probs.max().item())
    mask = (probs > 0.5).float()                # binary

mask_np = mask[0, 0].cpu().numpy().astype("uint8") * 255
print("Mask unique values:", np.unique(mask_np))
print("Foreground pixel count:", (mask_np > 0).sum())
mask_img = Image.fromarray(mask_np)

# Resize mask back to original image size
mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

# Convert original image to RGBA
overlay_img = orig_image.convert("RGBA")

# Create a colored mask (red)
mask_rgba = Image.new("RGBA", (orig_w, orig_h), (255, 0, 0, 0))

# Put alpha where mask is present
mask_pixels = mask_img.load()
overlay_pixels = mask_rgba.load()

alpha = 120  # transparency: 0 (transparent) → 255 (opaque)

for y in range(orig_h):
    for x in range(orig_w):
        if mask_pixels[x, y] > 0:
            overlay_pixels[x, y] = (255, 0, 0, alpha)

# Alpha-composite mask onto image
overlay_img = Image.alpha_composite(overlay_img, mask_rgba)

# Save overlay
overlay_path = args.output.replace(".png", "_overlay.png")
overlay_img.save(overlay_path)

# Show overlay
overlay_img.show()

print(f"Saved overlay to: {overlay_path}")
