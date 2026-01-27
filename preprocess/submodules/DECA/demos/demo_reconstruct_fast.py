#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast DECA Reconstruction Script

Optimizations over demo_reconstruct.py:
1. Batch processing (multiple images at once)
2. Skip face detection for pre-cropped images (--iscrop False)
3. Multi-worker data loading
4. Optional GPU memory optimization

Usage:
    # Fast mode for pre-cropped faces (recommended for ReliTalk pipeline)
    python demo_reconstruct_fast.py -i /path/to/images --savefolder /path/to/output \
        --iscrop False --batch_size 8 --num_workers 4

    # With face detection (slower)
    python demo_reconstruct_fast.py -i /path/to/images --savefolder /path/to/output \
        --iscrop True --batch_size 4
"""

import os
import sys
import cv2
import numpy as np
from time import time
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


class FastTestData(torch.utils.data.Dataset):
    """Optimized dataset for batch processing without face detection"""
    
    def __init__(self, testpath, crop_size=224, sample_step=1):
        from glob import glob
        from skimage.io import imread
        
        if os.path.isdir(testpath):
            self.imagepath_list = sorted(
                glob(testpath + '/*.jpg') + 
                glob(testpath + '/*.png') + 
                glob(testpath + '/*.bmp')
            )
        else:
            self.imagepath_list = [testpath]
        
        # Apply sample step
        if sample_step > 1:
            self.imagepath_list = self.imagepath_list[::sample_step]
        
        self.crop_size = crop_size
        print(f"FastTestData: {len(self.imagepath_list)} images to process")
    
    def __len__(self):
        return len(self.imagepath_list)
    
    def __getitem__(self, index):
        from skimage.io import imread
        from skimage.transform import resize
        
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        
        # Read image
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        
        # Resize to crop_size (skip face detection)
        h, w, _ = image.shape
        if h != self.crop_size or w != self.crop_size:
            image = resize(image, (self.crop_size, self.crop_size), anti_aliasing=True)
        else:
            image = image / 255.0
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.tensor(image.transpose(2, 0, 1)).float()
        
        return {
            'image': image_tensor,
            'imagename': imagename,
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch], dim=0)
    imagenames = [item['imagename'] for item in batch]
    return {'images': images, 'imagenames': imagenames}


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    
    print("="*60)
    print("Fast DECA Reconstruction")
    print("="*60)
    print(f"Input: {args.inputpath}")
    print(f"Output: {savefolder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Skip face detection: {not args.iscrop}")
    print(f"Sample step: {args.sample_step}")
    print("="*60)
    
    # Choose dataset based on whether we need face detection
    if args.iscrop:
        # Use original dataset with face detection (slower)
        print("\n⚠️  Face detection enabled - this will be slower")
        print("   For pre-cropped faces, use --iscrop False")
        testdata = datasets.TestData(
            args.inputpath, 
            iscrop=True, 
            face_detector=args.detector,
            sample_step=args.sample_step
        )
        # Cannot batch with face detection due to variable preprocessing
        use_batch = False
    else:
        # Use fast dataset without face detection
        print("\n✓ Face detection disabled - using fast mode")
        testdata = FastTestData(
            args.inputpath,
            crop_size=224,
            sample_step=args.sample_step
        )
        use_batch = True
    
    # Initialize DECA
    print("\nLoading DECA model...")
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)
    deca.eval()
    print("DECA model loaded")
    
    # Storage for all results
    all_codes = {}
    
    start_time = time()
    
    if use_batch and args.batch_size > 1:
        # Batch processing mode
        dataloader = DataLoader(
            testdata,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if device == 'cuda' else False
        )
        
        total_batches = len(dataloader)
        processed = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            images = batch['images'].to(device)
            imagenames = batch['imagenames']
            
            with torch.no_grad():
                # Encode batch
                codedict = deca.encode(images, use_detail=False)
            
            # Save results for each image in batch
            for i, name in enumerate(imagenames):
                code_dict = {}
                for k in ['pose', 'cam', 'exp', 'shape']:
                    if k in codedict:
                        code_dict[k] = codedict[k][i:i+1].detach().cpu().numpy().tolist()
                all_codes[name] = code_dict
                
                # Create individual folder if needed
                if args.saveVis or args.saveObj:
                    os.makedirs(os.path.join(savefolder, name), exist_ok=True)
            
            processed += len(imagenames)
            
            # Memory cleanup every N batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache() if device == 'cuda' else None
        
    else:
        # Single image processing mode (with face detection)
        for i in tqdm(range(len(testdata)), desc="Processing images"):
            data = testdata[i]
            name = data['imagename']
            images = data['image'].to(device)[None, ...]
            
            with torch.no_grad():
                codedict = deca.encode(images, use_detail=False)
            
            # Save code
            code_dict = {}
            for k in ['pose', 'cam', 'exp', 'shape']:
                if k in codedict:
                    code_dict[k] = codedict[k].detach().cpu().numpy().tolist()
            all_codes[name] = code_dict
            
            if args.saveVis or args.saveObj:
                os.makedirs(os.path.join(savefolder, name), exist_ok=True)
    
    elapsed = time() - start_time
    fps = len(testdata) / elapsed if elapsed > 0 else 0
    
    # Save all codes to JSON
    if args.saveCode:
        # Save to deca folder (same location as savefolder)
        json_path_deca = os.path.join(savefolder, 'code.json')
        with open(json_path_deca, 'w') as f:
            json.dump(all_codes, f, indent=2)
        print(f"\n✓ Saved FLAME codes to: {json_path_deca}")
        
        # Also save to parent folder for compatibility with ReliTalk
        parent_folder = os.path.dirname(savefolder.rstrip('/'))
        if parent_folder and parent_folder != savefolder:
            json_path_parent = os.path.join(parent_folder, 'code.json')
            with open(json_path_parent, 'w') as f:
                json.dump(all_codes, f, indent=2)
            print(f"✓ Also saved to: {json_path_parent}")
        
        # Save individual .npy files for each image (compatible with original DECA format)
        print(f"\nSaving individual parameter files...")
        for name, code_dict in tqdm(all_codes.items(), desc="Saving .npy files"):
            # Create subfolder for each image
            img_folder = os.path.join(savefolder, name)
            os.makedirs(img_folder, exist_ok=True)
            
            # Save each parameter as .npy
            for param_name, param_value in code_dict.items():
                npy_path = os.path.join(img_folder, f'{name}_{param_name}.npy')
                np.save(npy_path, np.array(param_value))
            
            # Also save a combined .npy file
            combined_path = os.path.join(img_folder, f'{name}_code.npy')
            np.save(combined_path, code_dict, allow_pickle=True)
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Total images: {len(testdata)}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Speed: {fps:.2f} FPS")
    print(f"Results saved to: {savefolder}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast DECA Reconstruction')
    
    parser.add_argument('-i', '--inputpath', required=True, type=str,
                        help='Path to input images folder')
    parser.add_argument('-s', '--savefolder', required=True, type=str,
                        help='Path to output directory')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device: cuda or cpu')
    
    # Speed optimization options
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for processing (default: 8)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--sample_step', default=1, type=int,
                        help='Process every Nth image (default: 1, process all)')
    
    # Processing options
    parser.add_argument('--iscrop', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to run face detection (default: False for speed)')
    parser.add_argument('--detector', default='fan', type=str,
                        help='Face detector type if iscrop=True')
    
    # Output options
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='Rasterizer type: pytorch3d or standard')
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to use texture model')
    parser.add_argument('--saveCode', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save FLAME parameters')
    parser.add_argument('--saveVis', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save visualizations')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save OBJ meshes')
    
    args = parser.parse_args()
    
    # Auto-detect GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
        args.batch_size = min(args.batch_size, 4)  # Reduce batch size for CPU
    
    main(args)
