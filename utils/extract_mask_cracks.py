import cv2
import os
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument('--image_dir', type=str, help='Directory containing input images')
args.add_argument('--label_dir', type=str, help='Directory containing label files')
args.add_argument('--mask_dir', type=str, help='Directory to save output masks')
opts = args.parse_args()

IMAGE_DIR = opts.image_dir
LABEL_DIR = opts.label_dir
MASK_DIR  = opts.mask_dir

os.makedirs(MASK_DIR, exist_ok=True)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(
        LABEL_DIR, img_name.rsplit(".", 1)[0] + ".txt"
    )

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                values = list(map(float, line.split()))
                cls = int(values[0])
                coords = values[1:]

                polygon = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    polygon.append([x, y])

                polygon = np.array([polygon], dtype=np.int32)

                # Fill polygon
                cv2.fillPoly(mask, polygon, 255)

    out_path = os.path.join(
        MASK_DIR, img_name.rsplit(".", 1)[0] + ".png"
    )
    cv2.imwrite(out_path, mask)