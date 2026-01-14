"""
Inference script for PADL.

This script performs two main tasks:
1. Encode protective perturbations into images
2. Decode and detect manipulations in protected images
"""

import argparse
from json import decoder
import os
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from Models.DecodingModule import DecodingModule
from Models.EncodingModule import EncodingModule


# Supported image extensions
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, img_size: int = 128):
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_list = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = os.path.join(self.data_dir, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


def load_models(model_dir: str, depth: int, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    encoding_module = EncodingModule(depth=depth).to(device)
    encoding_module.load_state_dict(torch.load(os.path.join(model_dir, 'encoding_module.pth'), map_location=device))
    encoding_module.eval()

    decoding_module = DecodingModule(S_depth=depth, M_depth=depth).to(device)
    decoding_module.load_state_dict(torch.load(os.path.join(model_dir, 'decoding_module.pth'), map_location=device))
    decoding_module.eval()

    return encoding_module, decoding_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ImSP Inference - Encode/Decode protective perturbations')
    parser.add_argument('--model_dir', type=str, required=True,help='Directory containing model weights')
    parser.add_argument('--data_dir', type=str, required=True,help='Directory containing input images')
    parser.add_argument('--depth', type=int, default=3,help='Network depth parameter (default: 3)')
    parser.add_argument('--resolution', type=int, default=128,help='Image resolution for processing (default: 128)')
    parser.add_argument('--batch_size', type=int, default=16,help='Batch size for inference (default: 16)')
    parser.add_argument('--alpha', type=float, default=0.03,help='Perturbation strength coefficient (default: 0.03)')
    return parser.parse_args()


def main():
    """Main inference pipeline."""
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoding_module, decoding_module = load_models(args.model_dir, args.depth, device)

    dataset = ImageDataset(data_dir=args.data_dir, img_size=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} images from: {args.data_dir}")

    # Run inference
    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)

            # Encoding: Apply protective perturbation
            perturbation = encoding_module(images)
            protected_images = images + perturbation * args.alpha
            protected_images = torch.clamp(protected_images, -1, 1)

            # Placeholder for manipulation
            manipulated_images = protected_images

            # Decoding: Detect manipulation
            manipulation_map, recovered_perturbation, detection = decoding_module(manipulated_images)
            
            detection = torch.sigmoid(detection).squeeze()
            detection = (detection > 0.5).long()

            # Detection scores: 0 = protected, 1 = unprotected
            print(f"Batch {batch_idx + 1}: Detection scores = {detection.cpu().numpy()}")


if __name__ == '__main__':
    main()






