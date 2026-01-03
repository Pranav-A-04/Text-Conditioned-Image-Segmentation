import os
import random
import cv2
import torch
from torch.utils.data import Dataset
class CracksAndDrywallDataloader(Dataset):
    def __init__(self, cracks_path, drywall_path, prompts):
        super().__init__()
        self.cracks = cracks_path
        self.drywall = drywall_path
        self.prompts = prompts
        self.dataset = self.load_data()

    def load_data(self):
        dataset = []
        
        # Load cracks data
        for img_name in os.listdir(os.path.join(self.cracks, "images")):
            if not img_name.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(self.cracks, "images", img_name)
            mask_path = os.path.join(
                self.cracks, "masks", img_name.rsplit(".", 1)[0] + ".png"
            )

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if random.random() < 0.5:
                prompt = random.choice(self.prompts["crack"])
            else:
                prompt = random.choice(self.prompts["taping"])
            dataset.append({
                'image': image,
                'mask': mask,
                'prompt': prompt
            })
        # Load drywall data
        for img_name in os.listdir(os.path.join(self.drywall, "images")):
            if not img_name.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(self.drywall, "images", img_name)
            mask_path = os.path.join(
                self.drywall, "masks", img_name.rsplit(".", 1)[0] + ".png"
            )

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if random.random() < 0.5:
                prompt = random.choice(self.prompts["crack"])
            else:
                prompt = random.choice(self.prompts["taping"])
            dataset.append({
                'image': image,
                'mask': mask,
                'prompt': prompt
            })
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data['image']
        mask = data['mask']
        prompt = data['prompt']
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'prompt': prompt
        }