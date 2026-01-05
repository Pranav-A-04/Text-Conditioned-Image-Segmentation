import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CracksAndDrywallDataloader(Dataset):
    def __init__(self, cracks_path, drywall_path, prompts, img_size=224):
        super().__init__()
        self.cracks = cracks_path
        self.drywall = drywall_path
        self.prompts = prompts
        self.img_size = img_size
        self.dataset = self.load_data()

        # ImageNet normalization for timm ViT
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # converts to [3,H,W] in [0,1]
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()  # [1,H,W] in [0,1]
        ])

    def load_data(self):
        dataset = []

        def load_folder(root):
            samples = []
            for img_name in os.listdir(os.path.join(root, "images")):
                if not img_name.lower().endswith((".jpg", ".png")):
                    continue

                img_path = os.path.join(root, "images", img_name)
                mask_path = os.path.join(
                    root, "masks", img_name.rsplit(".", 1)[0] + ".png"
                )

                if not os.path.exists(mask_path):
                    continue

                samples.append((img_path, mask_path))
            return samples

        for img_path, mask_path in load_folder(self.cracks):
            dataset.append((img_path, mask_path, "crack"))

        for img_path, mask_path in load_folder(self.drywall):
            dataset.append((img_path, mask_path, "taping"))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, mask_path, gt_class = self.dataset[idx]

        # load image
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)   # [3,224,224]

        # load mask
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)      # [1,224,224]
        mask = (mask > 0.5).float()            # binarize to {0,1}

        # sample prompt
        if gt_class == "crack":
            if random.random() < 0.65:
                prompt = random.choice(self.prompts["crack"])
            else:
                prompt = random.choice(self.prompts["taping"])
        else:
            if random.random() < 0.65:
                prompt = random.choice(self.prompts["taping"])
            else:
                prompt = random.choice(self.prompts["crack"])

        is_positive = (prompt in self.prompts[gt_class])
        return {
            "image": image,   # [3,224,224]
            "mask": mask,     # [1,224,224]
            "prompt": prompt,
            "is_positive": is_positive,
            "gt_class": gt_class
        }