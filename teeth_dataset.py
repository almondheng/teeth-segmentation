from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import torch
import os
import json


class_to_rgb = {
    "13": (248, 80, 42),
    "14": (80, 111, 190),
    "15": (138, 42, 162),
    "11": (63, 42, 122),
    "12": (4, 174, 154),
    "19": (28, 227, 190),
    "20": (126, 129, 200),
    "21": (135, 213, 222),
    "22": (89, 3, 226),
    "23": (18, 105, 211),
    "24": (200, 30, 58),
    "25": (4, 195, 134),
    "27": (58, 203, 65),
    "32": (112, 219, 207),
    "16": (255, 102, 242),
    "26": (178, 222, 59),
    "17": (119, 17, 192),
    "1": (51, 153, 104),
    "2": (30, 170, 235),
    "3": (33, 91, 166),
    "4": (245, 97, 107),
    "5": (42, 126, 100),
    "6": (240, 86, 227),
    "7": (126, 102, 225),
    "8": (146, 106, 210),
    "9": (229, 156, 31),
    "10": (39, 139, 138),
    "18": (181, 76, 230),
    "28": (82, 203, 201),
    "29": (240, 189, 112),
    "30": (48, 126, 108),
    "31": (61, 29, 191),
    "13(polygon)": (243, 63, 117),
}


def create_segmentation_mask(ann_data, target_size=(256, 256)):
    height, width = ann_data["size"]["height"], ann_data["size"]["width"]
    mask = np.zeros((height, width), dtype=np.int32)

    for obj in ann_data["objects"]:
        class_id = obj["classTitle"]
        points = np.array(obj["points"]["exterior"], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=class_to_rgb.get(class_id, (0, 0, 0)))

    # Resize the mask to match the target size (same as the image)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask_resized).long()


class TeethSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_dir, image_size=(256, 256), transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.image_size = image_size

        # Match images to their annotations
        self.image_files = [
            f
            for f in os.listdir(img_dir)
            if f.endswith(".jpg") and os.path.exists(os.path.join(ann_dir, f + ".json"))
        ]

        self.transform = transform or self._get_default_transform()

    def _get_default_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name + ".json")

        # Open the image in PIL format
        image = Image.open(img_path).convert("L")  # Grayscale

        # Apply image transformation (resize, convert to tensor, normalize)
        if self.transform:
            image = self.transform(image)

        # Load annotation data and generate segmentation mask
        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        # Resize and create mask
        mask = create_segmentation_mask(ann_data, target_size=self.image_size)

        return image, mask
