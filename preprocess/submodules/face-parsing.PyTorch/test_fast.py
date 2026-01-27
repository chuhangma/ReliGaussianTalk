#!/usr/bin/env python3
"""
Fast face parsing with batch processing support
Optimized version: Uses DataLoader for efficient GPU batch processing
"""
import os
import os.path as osp
import sys
import argparse
import re

import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


def natural_sort_key(s):
    """Sort key for natural sorting (1, 2, 10 instead of 1, 10, 2)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


class FaceDataset(Dataset):
    """Dataset for batch face parsing"""
    
    def __init__(self, image_dir, transform=None, target_size=512):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort(key=natural_sort_key)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = osp.join(self.image_dir, filename)
        
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        img_resized = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        
        if self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)
        
        return {
            'image': img_tensor,
            'filename': filename,
            'original_size': original_size
        }


def vis_parsing_maps(parsing_anno, save_path):
    """Save parsing mask as grayscale image"""
    vis_parsing_anno = parsing_anno.astype(np.uint8)
    cv2.imwrite(save_path, vis_parsing_anno)


def vis_parsing_maps_color(im, parsing_anno, save_path):
    """Save colored parsing visualization"""
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    num_of_class = np.max(vis_parsing_anno)
    
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        if pi < len(part_colors):
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    
    cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def evaluate_batch(respth, dspth, cp_path, batch_size=8, num_workers=4, save_color=True):
    """
    Batch face parsing evaluation
    
    Args:
        respth: Path to save parsing results
        dspth: Path to input images
        cp_path: Path to model checkpoint
        batch_size: Batch size for GPU processing
        num_workers: Number of data loading workers
        save_color: Whether to save colored visualization
    """
    # Import model (assumes we're in the face-parsing directory)
    from model import BiSeNet
    
    # Create output directories
    os.makedirs(respth, exist_ok=True)
    if save_color:
        os.makedirs(respth + '_color', exist_ok=True)
    
    # Initialize model
    print("Loading BiSeNet model...")
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp_path))
    net.eval()
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Create dataset and dataloader
    dataset = FaceDataset(dspth, transform=transform, target_size=512)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Processing {len(dataset)} images with batch_size={batch_size}...")
    
    # Process in batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Face Parsing"):
            images = batch['image'].cuda()
            filenames = batch['filename']
            
            # Forward pass
            outputs = net(images)[0]
            parsings = outputs.cpu().numpy().argmax(1)  # [B, H, W]
            
            # Save results
            for i, (parsing, filename) in enumerate(zip(parsings, filenames)):
                # Save grayscale parsing mask
                save_path = osp.join(respth, filename)
                vis_parsing_maps(parsing, save_path)
                
                # Optionally save colored visualization
                if save_color:
                    # Need to load original image for color overlay
                    img = Image.open(osp.join(dspth, filename)).convert('RGB')
                    img_resized = img.resize((512, 512), Image.BILINEAR)
                    save_path_color = osp.join(respth + '_color', filename)
                    vis_parsing_maps_color(img_resized, parsing, save_path_color)
    
    print(f"\nSaved parsing results to: {respth}")
    if save_color:
        print(f"Saved colored visualizations to: {respth}_color")


def main():
    parser = argparse.ArgumentParser(description='Fast face parsing with batch processing')
    parser.add_argument('--dspth', type=str, required=True,
                        help='Path to input images')
    parser.add_argument('--respth', type=str, required=True,
                        help='Path to save results')
    parser.add_argument('--cp', type=str, default='79999_iter.pth',
                        help='Checkpoint filename')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--no_color', action='store_true',
                        help='Disable colored visualization output')
    
    args = parser.parse_args()
    
    # Find checkpoint
    cp_path = osp.join('res/cp', args.cp)
    if not osp.exists(cp_path):
        cp_path = args.cp
    if not osp.exists(cp_path):
        raise RuntimeError(f"Checkpoint not found at {cp_path}")
    
    evaluate_batch(
        respth=args.respth,
        dspth=args.dspth,
        cp_path=cp_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_color=not args.no_color
    )


if __name__ == "__main__":
    main()
