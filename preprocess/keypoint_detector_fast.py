#!/usr/bin/env python3
"""
Fast face landmark detection with better error handling and progress tracking
"""
import face_alignment
from skimage import io
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import re


def natural_sort_key(s):
    """Sort key for natural sorting (1, 2, 10 instead of 1, 10, 2)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def run_landmark_detection(path, batch_size=16):
    """
    Run face landmark detection with batch processing
    
    Args:
        path: Base path containing 'image' folder
        batch_size: Number of images to process in each batch (affects memory usage)
    """
    image_path = os.path.join(path, 'image')
    
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image directory not found: {image_path}")
    
    # Get all image files
    image_files = [f for f in os.listdir(image_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise RuntimeError(f"No images found in {image_path}")
    
    # Sort naturally (1, 2, 3, ..., 10, 11, ... instead of 1, 10, 11, ...)
    image_files.sort(key=natural_sort_key)
    
    print(f"Found {len(image_files)} images in {image_path}")
    
    # Initialize face alignment detector (loads model to GPU)
    print("Initializing Face Alignment detector...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False,
        device='cuda'
    )
    
    # Process images and collect landmarks
    results = {}
    failed_count = 0
    
    print("Processing images...")
    
    # Method 1: Use built-in batch processing from directory
    # This is already optimized internally
    try:
        preds = fa.get_landmarks_from_directory(image_path)
        
        # Convert predictions to our format
        for full_path, landmarks in preds.items():
            filename = os.path.basename(full_path)
            if landmarks is not None and len(landmarks) > 0:
                # Take the first detected face
                results[filename] = landmarks[0].tolist()
            else:
                results[filename] = None
                failed_count += 1
                print(f"Warning: No face detected in {filename}")
                
    except Exception as e:
        print(f"Batch processing failed: {e}")
        print("Falling back to individual image processing...")
        
        # Method 2: Fallback to individual processing with progress bar
        for filename in tqdm(image_files, desc="Detecting landmarks"):
            img_path = os.path.join(image_path, filename)
            try:
                img = io.imread(img_path)
                landmarks = fa.get_landmarks(img)
                
                if landmarks is not None and len(landmarks) > 0:
                    results[filename] = landmarks[0].tolist()
                else:
                    results[filename] = None
                    failed_count += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results[filename] = None
                failed_count += 1
    
    # Handle missing landmarks by interpolation
    if failed_count > 0:
        print(f"\nInterpolating {failed_count} missing landmarks...")
        results = interpolate_missing_landmarks(results, image_files)
    
    # Save results
    output_path = os.path.join(path, 'keypoint.json')
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nSaved landmarks to: {output_path}")
    print(f"Successfully processed: {len(image_files) - failed_count}/{len(image_files)} images")
    
    if failed_count > 0:
        print(f"Warning: {failed_count} images had no face detection (interpolated from neighbors)")
    
    return results


def interpolate_missing_landmarks(results, sorted_files):
    """
    Interpolate missing landmarks from neighboring frames
    """
    # Convert to numpy for easier manipulation
    all_landmarks = []
    for f in sorted_files:
        lm = results.get(f)
        if lm is not None:
            all_landmarks.append(np.array(lm))
        else:
            all_landmarks.append(None)
    
    # Find first valid landmark
    first_valid = None
    for i, lm in enumerate(all_landmarks):
        if lm is not None:
            first_valid = i
            break
    
    if first_valid is None:
        raise RuntimeError("No valid landmarks detected in any frame!")
    
    # Forward fill
    for i in range(len(all_landmarks)):
        if all_landmarks[i] is None:
            # Find previous valid
            prev_idx = i - 1
            while prev_idx >= 0 and all_landmarks[prev_idx] is None:
                prev_idx -= 1
            
            # Find next valid
            next_idx = i + 1
            while next_idx < len(all_landmarks) and all_landmarks[next_idx] is None:
                next_idx += 1
            
            # Interpolate or copy
            if prev_idx >= 0 and next_idx < len(all_landmarks):
                # Linear interpolation
                t = (i - prev_idx) / (next_idx - prev_idx)
                all_landmarks[i] = (1 - t) * all_landmarks[prev_idx] + t * all_landmarks[next_idx]
            elif prev_idx >= 0:
                all_landmarks[i] = all_landmarks[prev_idx].copy()
            elif next_idx < len(all_landmarks):
                all_landmarks[i] = all_landmarks[next_idx].copy()
    
    # Convert back to results dict
    for i, f in enumerate(sorted_files):
        if all_landmarks[i] is not None:
            results[f] = all_landmarks[i].tolist()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast face landmark detection')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to data directory (containing image/ folder)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing (default: 16)')
    
    args = parser.parse_args()
    run_landmark_detection(args.path, args.batch_size)
