#!/usr/bin/env python3
"""
Fast iris landmark detection with batch-style processing
Optimized version: detectors are initialized once and reused
"""
# Code borrowed and adapted from Neural Head Avatar
# https://github.com/philgras/neural-head-avatars/blob/473457eef83c9ee26f316451e31c2aa01a74603c/python_scripts/video_to_dataset.py#L26
from fdlite import (
    FaceDetection,
    FaceLandmark,
    face_detection_to_roi,
    IrisLandmark,
    iris_roi_from_face_landmarks,
)
from PIL import Image
import numpy as np
import os
import argparse
import re
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading


def natural_sort_key(s):
    """Sort key for natural sorting (1, 2, 10 instead of 1, 10, 2)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


class IrisDetector:
    """Reusable iris detector with pre-initialized models"""
    
    def __init__(self):
        print("Initializing iris detection models...")
        self.detect_faces = FaceDetection()
        self.detect_face_landmarks = FaceLandmark()
        self.detect_iris_landmarks = IrisLandmark()
        
    def detect(self, img):
        """
        Detect iris landmarks in a single image
        
        Args:
            img: PIL Image
            
        Returns:
            list: [left_eye_x, left_eye_y, right_eye_x, right_eye_y] or None
        """
        width, height = img.size
        img_size = (width, height)
        
        try:
            face_detections = self.detect_faces(img)
            
            if len(face_detections) != 1:
                return None
            
            for face_detection in face_detections:
                try:
                    face_roi = face_detection_to_roi(face_detection, img_size)
                except ValueError:
                    return None
                
                face_landmarks = self.detect_face_landmarks(img, face_roi)
                if len(face_landmarks) == 0:
                    return None
                
                iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)
                
                if len(iris_rois) != 2:
                    return None
                
                lmks = []
                for iris_roi in iris_rois[::-1]:
                    try:
                        iris_landmarks = self.detect_iris_landmarks(img, iris_roi).iris[0:1]
                    except np.linalg.LinAlgError:
                        return None
                    
                    for landmark in iris_landmarks:
                        lmks.append(landmark.x * width)
                        lmks.append(landmark.y * height)
                
                return lmks
                
        except Exception as e:
            print(f"Iris detection error: {e}")
            return None
        
        return None


def annotate_iris_landmarks_fast(image_path, savefolder, num_workers=4):
    """
    Annotates each frame with 2 iris landmarks using optimized processing
    
    Args:
        image_path: Path to directory containing images
        savefolder: Path to save output JSON
        num_workers: Number of parallel workers for image loading
    """
    
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image directory not found: {image_path}")
    
    # Get and sort frames
    frames = [f for f in os.listdir(image_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not frames:
        raise RuntimeError(f"No images found in {image_path}")
    
    frames.sort(key=natural_sort_key)
    print(f"Found {len(frames)} images in {image_path}")
    
    # Initialize detector once (main optimization)
    detector = IrisDetector()
    
    # Process images
    landmarks = {}
    failed_count = 0
    
    print("Detecting iris landmarks...")
    for frame in tqdm(frames, desc="Processing"):
        img_path = os.path.join(image_path, frame)
        try:
            img = Image.open(img_path)
            lmks = detector.detect(img)
            landmarks[frame] = lmks
            
            if lmks is None:
                failed_count += 1
                
        except Exception as e:
            print(f"Error processing {frame}: {e}")
            landmarks[frame] = None
            failed_count += 1
    
    # Interpolate missing iris landmarks from neighboring frames
    if failed_count > 0:
        print(f"\nInterpolating {failed_count} missing iris landmarks...")
        landmarks = interpolate_missing_iris(landmarks, frames)
    
    # Save results
    output_path = os.path.join(savefolder, 'iris.json')
    with open(output_path, 'w') as f:
        json.dump(landmarks, f)
    
    print(f"\nSaved iris landmarks to: {output_path}")
    print(f"Successfully processed: {len(frames) - failed_count}/{len(frames)} images")
    
    if failed_count > 0:
        print(f"Warning: {failed_count} images had no iris detection (interpolated from neighbors)")
    
    return landmarks


def interpolate_missing_iris(landmarks, sorted_frames):
    """
    Interpolate missing iris landmarks from neighboring frames
    """
    # Convert to list for easier indexing
    all_lmks = []
    for f in sorted_frames:
        lm = landmarks.get(f)
        if lm is not None and len(lm) == 4:
            all_lmks.append(np.array(lm))
        else:
            all_lmks.append(None)
    
    # Find first valid landmark
    first_valid = None
    for i, lm in enumerate(all_lmks):
        if lm is not None:
            first_valid = i
            break
    
    if first_valid is None:
        print("Warning: No valid iris landmarks detected in any frame!")
        return landmarks
    
    # Interpolate missing values
    for i in range(len(all_lmks)):
        if all_lmks[i] is None:
            # Find previous valid
            prev_idx = i - 1
            while prev_idx >= 0 and all_lmks[prev_idx] is None:
                prev_idx -= 1
            
            # Find next valid
            next_idx = i + 1
            while next_idx < len(all_lmks) and all_lmks[next_idx] is None:
                next_idx += 1
            
            # Interpolate or copy
            if prev_idx >= 0 and next_idx < len(all_lmks):
                # Linear interpolation
                t = (i - prev_idx) / (next_idx - prev_idx)
                all_lmks[i] = (1 - t) * all_lmks[prev_idx] + t * all_lmks[next_idx]
            elif prev_idx >= 0:
                all_lmks[i] = all_lmks[prev_idx].copy()
            elif next_idx < len(all_lmks):
                all_lmks[i] = all_lmks[next_idx].copy()
    
    # Convert back to dict
    for i, f in enumerate(sorted_frames):
        if all_lmks[i] is not None:
            landmarks[f] = all_lmks[i].tolist()
    
    return landmarks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast iris landmark detection')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to data directory (containing image/ folder)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel loading')
    
    args = parser.parse_args()
    image_path = os.path.join(args.path, 'image')
    annotate_iris_landmarks_fast(image_path=image_path, savefolder=args.path, 
                                  num_workers=args.num_workers)
