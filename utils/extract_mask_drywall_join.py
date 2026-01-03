import cv2
import os
import numpy as np


# Extract masks from yaml config file
IMAGE_DIR = "/content/Drywall-Join-Detect-3/train/images"
LABEL_DIR = "/content/Drywall-Join-Detect-3/train/labels"
MASK_DIR  = "/content/Drywall-Join-Detect-3/train/masks"

os.makedirs(MASK_DIR, exist_ok=True)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(
        LABEL_DIR, img_name.rsplit(".", 1)[0] + ".txt"
    )

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    mask = np.zeros((h, w), dtype=np.uint8)

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                mask[y1:y2, x1:x2] = 255

    out_path = os.path.join(
        MASK_DIR, img_name.rsplit(".", 1)[0] + ".png"
    )
    cv2.imwrite(out_path, mask)
