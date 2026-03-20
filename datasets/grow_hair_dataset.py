import traceback
import json
import os
import random
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class GrowHairDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 json_file_path: str,
                 resolution: int = 512,
                 ):

        with open(json_file_path, 'r') as file:
            self.data = [json.loads(line) for line in file]
        self.data_root = data_root
        self.resolution = resolution

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor
        
    def to_mask_tensor(self, mask: np.ndarray) -> torch.Tensor:
        """Converts a uint8 mask (0-255) to a float tensor in [0, 1]."""
        mask = mask.astype(np.float32) / 255.0
        if mask.ndim == 2:
            mask = mask[None, ...]
        tensor = torch.from_numpy(mask).float()
        return tensor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        try:
            data = self.data[item]

            mask_path = os.path.join(self.data_root, data['mask_path'])
            img_path = os.path.join(self.data_root, data['image_path'])
            
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
            mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)

            img = np.array(img)
            mask = np.array(mask)

            img_tensor = self.to_tensor(img)
            dilate_kernel = (3, 3)
            mask = cv2.dilate(mask, np.ones(dilate_kernel, np.uint8), iterations=1)
            mask_tensor = self.to_mask_tensor(mask)

            prompt = data.get('prompt', '')

            return {
                'image': img_tensor,
                'mask': mask_tensor,
                'prompt': prompt,
            }
        
        except Exception as e:
            raise Exception(f"Error in .__getitem__: {traceback.format_exc()}")


if __name__ == "__main__":
    dataset = GrowHairDataset(
        data_root="/home/toxu/work/dataset/grow_hair",
        json_file_path="/home/toxu/work/dataset/grow_hair/train.jsonl",
        resolution=512,
    )
    
    item = dataset[0]
    print(f"image: {item['image'].shape}, mask: {item['mask'].shape}, prompt: {item['prompt']!r}")