import os
from PIL import Image
import torch
from torchvision import transforms

import torch.nn as nn

class TrainDataset(torch.utils.data.Dataset):
    """
    Train dataset class for training mlp layer between alpha-clip and clip
    path: path to the directory containing images
    size: size of the images
    """
    def __init__(self, path, size=512):
        super().__init__()
        self.validate_args(path, size)
        self.image_root_path = path
        self.img_paths = [path for path in os.listdir(path) if path.endswith("jpg") or path.endswith("png") or path.endswith("jpeg")]
        self.size = size
        
    def validate_args(self, path, size):
        assert os.path.isdir(path), "Path should be a directory"
        assert size > 0, "Size should be greater than 0"
        
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_root_path, self.img_paths[idx])).convert("RGB").resize((self.size, self.size))
        width, height = image.size
        mask = torch.ones(height, width).unsqueeze(0)
        return {
            "image": transforms.ToTensor()(image),
            "mask": mask,
        }

    def __len__(self):
        return len(self.img_paths)
    
class ValidationDataset(torch.utils.data.Dataset):
    """
    Validation dataset class for mlp layer training between alpha-clip and clip
    path: path to the directory containing images
    size: size of the images
    """
    def __init__(self, path, size=512):
        super().__init__()
        self.validate_args(path, size)
        self.image_root_path = path
        self.img_paths = [path for path in os.listdir(path) if path.endswith("jpg") or path.endswith("png") or path.endswith("jpeg")]
        self.size = size
        
    def validate_args(self, path, size):
        assert os.path.isdir(path), "Path should be a directory"
        assert size > 0, "Size should be greater than 0"
        
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_root_path, self.img_paths[idx])).convert("RGB").resize((self.size, self.size))
        width, height = image.size
        mask = torch.ones(height, width).unsqueeze(0)
        return {
            "image": transforms.ToTensor()(image),
            "mask": mask,
        }

    def __len__(self):
        return len(self.img_paths)
        
class TestDataset(torch.utils.data.Dataset):
    """
    Test dataset class for inference of the clipaway model for a dataset
    root_path: path to the directory containing images and masks. expected file structure is:
    root_path
    ├── images
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── masks
        ├── image1.png
        ├── image2.png
        └── ...
    size: size of the images
    """
    def __init__(self, root_path, size=512):
        super().__init__()
        self.validate_args(root_path, size)
        self.image_root_path = root_path
        self.image_path = os.path.join(root_path, "images")
        self.mask_path = os.path.join(root_path, "masks")
        self.images = [path for path in sorted(os.listdir(self.image_path)) if path.endswith("jpg")]
        self.masks = [path for path in sorted(os.listdir(self.mask_path)) if path.endswith("png")]
        self.size = size
    
    def validate_args(self, root_path, size):
        assert os.path.isdir(root_path), "Root path should be a directory"
        assert os.path.isdir(os.path.join(root_path, "images")), "Images directory should exist"
        assert os.path.isdir(os.path.join(root_path, "masks")), "Masks directory should exist"
        assert len(os.listdir(os.path.join(root_path, "images"))) == len(os.listdir(os.path.join(root_path, "masks"))), "Number of images and masks should be the same"
        assert size > 0, "Size should be greater than 0"
        
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, self.images[idx])).convert("RGB").resize((self.size, self.size))
        mask = Image.open(os.path.join(self.mask_path, self.masks[idx])).convert("L").resize((self.size, self.size), Image.NEAREST)
        return {
            "image": transforms.ToTensor()(image),
            "mask": transforms.ToTensor()(mask).round(),
            "image_path": self.images[idx],
            "mask_path": self.masks[idx],
        }

    def __len__(self):
        return len(self.images)
