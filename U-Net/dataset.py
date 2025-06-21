from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

torch.set_printoptions(profile="full")


class RoadSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # 기본 변환 (Tensor + Resize)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = transform(image)
        mask = transform(mask)

        # 마스크를 0과 1로 정규화 (255 → 1)
        mask = mask / 255.0
        mask = (mask > 0).float()

        return image, mask
