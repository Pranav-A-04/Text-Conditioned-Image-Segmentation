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

print("Loading ViT...")
vit = timm.create_model("vit_base_patch16_224", pretrained=True)
vit.reset_classifier(0)
vit.eval().to(device)

print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

print("Loading decoder checkpoint...")
decoder = SegmentationDecoder(
    vis_token_dim=768,
    text_token_dim=512,
    input_dim=512,
    output_dim=1
)
decoder.load_state_dict(torch.load(args.ckpt, map_location=device))
decoder.eval().to(device)

orig_image = Image.open(args.image_path).convert("RGB")
orig_w, orig_h = orig_image.size

image = image_transform(orig_image).unsqueeze(0).to(device)  # [1,3,224,224]


with torch.no_grad():
    text_tokens = clip.tokenize([args.prompt]).to(device)
    text_emb = clip_model.encode_text(text_tokens).float()  # [1,512]

with torch.no_grad():
    vis_tokens = vit.forward_features(image)   # [1,197,768]
    vis_tokens = vis_tokens[:, 1:, :]           # drop CLS → [1,196,768]

with torch.no_grad():
    logits = decoder(vis_tokens, text_emb)      # [1,1,H,W]
    probs = torch.sigmoid(logits)
    print("Prob min/max:", probs.min().item(), probs.max().item())
    mask = (probs > 0.2).float()                # binary

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
