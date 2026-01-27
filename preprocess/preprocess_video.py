#!/usr/bin/env python3
"""
Unified Video Preprocessing Script for Relightable Gaussian Talking Head

This script automates the complete preprocessing pipeline:
1. Video cropping and resizing
2. Background/foreground segmentation (MODNet)
3. FLAME parameter estimation (DECA)
4. Face landmark detection
5. Iris segmentation
6. FLAME parameter optimization
7. Semantic segmentation (face-parsing)

Usage:
    python preprocess_video.py --video path/to/video.mp4 --output path/to/output/
    
    # Or with configuration file:
    python preprocess_video.py --config preprocess_config.yaml
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import json


class VideoPreprocessor:
    """Unified video preprocessing pipeline"""
    
    def __init__(self, 
                 video_path: str,
                 output_dir: str,
                 fps: int = 25,
                 crop: str = None,  # "w:h:x:y" format
                 resize: int = 512,
                 fx: float = None,
                 fy: float = None,
                 cx: float = None,
                 cy: float = None):
        
        self.video_path = Path(video_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.fps = fps
        self.crop = crop
        self.resize = resize
        
        # Camera intrinsics (will be estimated if not provided)
        self.fx = fx if fx else resize * 2.0
        self.fy = fy if fy else resize * 2.0
        self.cx = cx if cx else resize / 2.0
        self.cy = cy if cy else resize / 2.0
        
        # Get preprocess directory
        self.preprocess_dir = Path(__file__).parent.resolve()
        self.modnet_dir = self.preprocess_dir / "submodules" / "MODNet"
        self.deca_dir = self.preprocess_dir / "submodules" / "DECA"
        self.parser_dir = self.preprocess_dir / "submodules" / "face-parsing.PyTorch"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "image").mkdir(exist_ok=True)
        (self.output_dir / "mask").mkdir(exist_ok=True)
        (self.output_dir / "deca").mkdir(exist_ok=True)
        (self.output_dir / "semantic").mkdir(exist_ok=True)
        (self.output_dir / "flame_params").mkdir(exist_ok=True)
        
        self.cropped_video = self.output_dir / "video_cropped.mp4"
        self.matte_video = self.output_dir / "video_matte.mp4"
        
    def run_all(self, start_step: int = 1, end_step: int = 8):
        """Run preprocessing pipeline from start_step to end_step
        
        Args:
            start_step: First step to run (1-8)
            end_step: Last step to run (1-8)
        """
        # Define all steps
        steps = [
            (1, "Cropping and resizing video", self.crop_and_resize_video),
            (2, "Background segmentation with MODNet", self.run_modnet),
            (3, "Extracting frames", self.extract_frames),
            (4, "DECA FLAME parameter estimation", self.run_deca),
            (5, "Face landmark detection", self.run_landmark_detection),
            (6, "Iris segmentation", self.run_iris_segmentation),
            (7, "Semantic segmentation", self.run_face_parsing),
            (8, "Generating camera parameters", self.generate_camera_json),
        ]
        
        print("="*60)
        print("Relightable Gaussian Talking Head - Video Preprocessing")
        print("="*60)
        print(f"Input video: {self.video_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Resolution: {self.resize}x{self.resize}")
        print(f"FPS: {self.fps}")
        print(f"Running steps: {start_step} to {end_step}")
        print("="*60)
        
        # Show step plan
        print("\nStep plan:")
        for step_num, step_name, _ in steps:
            if start_step <= step_num <= end_step:
                print(f"  [Step {step_num}] {step_name} - WILL RUN")
            else:
                print(f"  [Step {step_num}] {step_name} - SKIP")
        print()
        
        # Execute steps
        for step_num, step_name, step_func in steps:
            if step_num < start_step:
                print(f"\n[Step {step_num}/8] {step_name}... SKIPPED (before start_step)")
                continue
            if step_num > end_step:
                print(f"\n[Step {step_num}/8] {step_name}... SKIPPED (after end_step)")
                continue
                
            print(f"\n[Step {step_num}/8] {step_name}...")
            print("-" * 50)
            
            try:
                step_func()
                print(f"✓ Step {step_num} completed successfully")
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"✗ ERROR at Step {step_num}: {step_name}")
                print(f"{'='*60}")
                print(f"Error message: {e}")
                print(f"\nTo resume from this step, run with: --start-step {step_num}")
                raise
        
        print("\n" + "="*60)
        print("Preprocessing complete!")
        print(f"Output saved to: {self.output_dir}")
        print("="*60)
        
    def crop_and_resize_video(self):
        """Crop and resize input video using ffmpeg"""
        if self.crop:
            # crop format: "w:h:x:y"
            filter_str = f"fps={self.fps}, crop={self.crop}, scale={self.resize}:{self.resize}"
        else:
            filter_str = f"fps={self.fps}, scale={self.resize}:{self.resize}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-vf", filter_str,
            "-c:v", "libx264",
            str(self.cropped_video)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: ffmpeg returned non-zero exit code")
            print(f"stderr: {result.stderr}")
        
        if not self.cropped_video.exists():
            raise RuntimeError(f"Failed to create cropped video: {self.cropped_video}")
            
        print(f"Created: {self.cropped_video}")
        
    def run_modnet(self):
        """Run MODNet for background segmentation"""
        if not self.modnet_dir.exists():
            print(f"Warning: MODNet not found at {self.modnet_dir}")
            print("Skipping background segmentation...")
            return
        
        # Check if MODNet pretrained model exists
        modnet_ckpt = self.modnet_dir / "pretrained" / "modnet_photographic_portrait_matting.ckpt"
        if not modnet_ckpt.exists():
            print(f"Warning: MODNet checkpoint not found at {modnet_ckpt}")
            print("Please download from: https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR")
            print("Skipping background segmentation...")
            return
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.modnet_dir)
            
            cmd = [
                sys.executable, "-m", "demo.video_matting.custom.run",
                "--video", str(self.cropped_video),
                "--result-type", "matte",
                "--fps", str(self.fps)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: MODNet returned non-zero exit code")
                print(f"stderr: {result.stderr}")
                
            # MODNet outputs to video_cropped_matte.mp4 in the same directory
            expected_matte = self.output_dir / "video_cropped_matte.mp4"
            if expected_matte.exists():
                shutil.move(str(expected_matte), str(self.matte_video))
                print(f"Created: {self.matte_video}")
            else:
                print(f"Warning: Matte video not found at {expected_matte}")
                
        finally:
            os.chdir(original_dir)
            
    def extract_frames(self):
        """Extract frames from video using ffmpeg"""
        image_dir = self.output_dir / "image"
        mask_dir = self.output_dir / "mask"
        
        # Extract RGB frames
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.cropped_video),
            "-q:v", "2",
            str(image_dir / "%d.png")
        ]
        
        print(f"Extracting RGB frames...")
        subprocess.run(cmd, capture_output=True)
        
        # Extract mask frames if matte video exists
        if self.matte_video.exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.matte_video),
                "-q:v", "2",
                str(mask_dir / "%d.png")
            ]
            
            print(f"Extracting mask frames...")
            subprocess.run(cmd, capture_output=True)
        else:
            print("Warning: Matte video not found, creating white masks...")
            # Create white masks
            import cv2
            import numpy as np
            for img_file in sorted(image_dir.glob("*.png")):
                img = cv2.imread(str(img_file))
                if img is not None:
                    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                    cv2.imwrite(str(mask_dir / img_file.name), mask)
        
        num_frames = len(list(image_dir.glob("*.png")))
        print(f"Extracted {num_frames} frames")
        
    def run_deca(self, use_fast_mode: bool = True, batch_size: int = 8, num_workers: int = 4):
        """Run DECA for FLAME parameter estimation
        
        Args:
            use_fast_mode: Use optimized fast DECA script (skips face detection)
            batch_size: Batch size for GPU processing (only for fast mode)
            num_workers: Number of data loading workers
        """
        if not self.deca_dir.exists():
            raise RuntimeError(f"DECA not found at {self.deca_dir}")
            
        # Check DECA dependencies
        deca_data = self.deca_dir / "data"
        required_files = ["deca_model.tar", "generic_model.pkl"]
        missing_files = [f for f in required_files if not (deca_data / f).exists()]
        
        if missing_files:
            raise RuntimeError(f"Missing DECA data files: {missing_files}. Please download from DECA repository")
        
        image_dir = self.output_dir / "image"
        deca_out_dir = self.output_dir / "deca"
        
        # Check which script to use
        fast_script = self.deca_dir / "demos" / "demo_reconstruct_fast.py"
        original_script = self.deca_dir / "demos" / "demo_reconstruct.py"
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.deca_dir)
            
            if use_fast_mode and fast_script.exists():
                # Use optimized fast script
                print(f"Using FAST mode: batch_size={batch_size}, num_workers={num_workers}")
                print("Note: Fast mode skips face detection (assumes pre-cropped faces)")
                
                cmd = [
                    sys.executable, "demos/demo_reconstruct_fast.py",
                    "-i", str(image_dir),
                    "--savefolder", str(deca_out_dir),
                    "--batch_size", str(batch_size),
                    "--num_workers", str(num_workers),
                    "--iscrop", "False",  # Skip face detection for speed
                    "--saveCode", "True",
                    "--saveVis", "False"
                ]
            else:
                # Fall back to original script
                print("Using ORIGINAL mode (with face detection - slower)")
                cmd = [
                    sys.executable, "demos/demo_reconstruct.py",
                    "-i", str(image_dir),
                    "--savefolder", str(deca_out_dir),
                    "--saveCode", "True",
                    "--saveVis", "False",
                    "--sample_step", "1",
                    "--render_orig", "False",
                    "--iscrop", "False"  # Skip face detection since images are already cropped
                ]
            
            print(f"Running: {' '.join(cmd)}")
            # 不捕获输出，直接打印到终端，方便看到完整错误
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                raise RuntimeError(f"DECA failed with exit code {result.returncode}")
                
        finally:
            os.chdir(original_dir)
        
        # Convert DECA output to ReliTalk format
        self._convert_deca_to_flame_params()
    
    def _convert_deca_to_flame_params(self):
        """Convert DECA code.json to ReliTalk flame_params.json format"""
        import numpy as np
        
        # Look for code.json in multiple locations
        code_json_paths = [
            self.output_dir / "code.json",
            self.output_dir / "deca" / "code.json",
        ]
        
        code_json_path = None
        for p in code_json_paths:
            if p.exists():
                code_json_path = p
                break
        
        if not code_json_path:
            print("Warning: code.json not found, skipping flame_params conversion")
            return
        
        print(f"\nConverting DECA output to ReliTalk format...")
        print(f"Reading: {code_json_path}")
        
        with open(code_json_path, 'r') as f:
            deca_codes = json.load(f)
        
        # Sort by frame number
        def get_frame_num(name):
            try:
                return int(name)
            except:
                import re
                nums = re.findall(r'\d+', name)
                return int(nums[-1]) if nums else 0
        
        sorted_names = sorted(deca_codes.keys(), key=get_frame_num)
        
        # Convert to ReliTalk format
        flame_params = {
            "frames": []
        }
        
        # Extract shared shape parameters (average of all frames)
        all_shapes = []
        for name in sorted_names:
            code = deca_codes[name]
            if 'shape' in code:
                shape = np.array(code['shape']).flatten()
                all_shapes.append(shape)
        
        if all_shapes:
            avg_shape = np.mean(all_shapes, axis=0)
            # Save shape parameters separately
            shape_path = self.output_dir / "shape_params.npy"
            np.save(shape_path, avg_shape)
            print(f"Saved shape parameters to: {shape_path}")
        
        # Build frames list
        for name in sorted_names:
            code = deca_codes[name]
            
            # DECA pose: [global_rot(3), jaw_pose(3)] = 6 params
            # We need to expand to 15 params for FLAME (global + neck + jaw + eyes)
            pose_raw = np.array(code.get('pose', [[0]*6])).flatten()
            
            # Pad pose to 15 dimensions if needed
            if len(pose_raw) < 15:
                pose_full = np.zeros(15)
                pose_full[:len(pose_raw)] = pose_raw
            else:
                pose_full = pose_raw[:15]
            
            # Expression coefficients
            exp_raw = np.array(code.get('exp', [[0]*50])).flatten()
            if len(exp_raw) < 50:
                exp_full = np.zeros(50)
                exp_full[:len(exp_raw)] = exp_raw
            else:
                exp_full = exp_raw[:50]
            
            # Camera parameters (DECA format: [scale, tx, ty])
            cam_raw = np.array(code.get('cam', [[1, 0, 0]])).flatten()
            
            # Create a simple world matrix from camera params
            # This is a simplified conversion - may need adjustment
            scale = cam_raw[0] if len(cam_raw) > 0 else 1.0
            tx = cam_raw[1] if len(cam_raw) > 1 else 0.0
            ty = cam_raw[2] if len(cam_raw) > 2 else 0.0
            
            world_mat = np.eye(4)
            world_mat[0, 3] = tx
            world_mat[1, 3] = ty
            world_mat[2, 3] = 3.0 / scale  # Approximate depth
            
            # Intrinsics (default)
            intrinsics = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
            
            frame_data = {
                "expression": exp_full.tolist(),
                "pose": pose_full.tolist(),
                "world_mat": world_mat.tolist(),
                "intrinsics": intrinsics.tolist(),
                "cam": cam_raw.tolist()  # Keep original cam for reference
            }
            flame_params["frames"].append(frame_data)
        
        # Save flame_params.json
        flame_params_path = self.output_dir / "flame_params.json"
        with open(flame_params_path, 'w') as f:
            json.dump(flame_params, f, indent=2)
        
        print(f"Converted {len(flame_params['frames'])} frames")
        print(f"Saved: {flame_params_path}")
            
    def run_landmark_detection(self, use_fast_mode: bool = True, batch_size: int = 16):
        """Run face landmark detection
        
        Args:
            use_fast_mode: Use optimized fast script with better error handling
            batch_size: Batch size for processing
        """
        fast_script = self.preprocess_dir / "keypoint_detector_fast.py"
        original_script = self.preprocess_dir / "keypoint_detector.py"
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.preprocess_dir)
            
            if use_fast_mode and fast_script.exists():
                print(f"Using FAST mode with improved error handling")
                cmd = [
                    sys.executable, "keypoint_detector_fast.py",
                    "--path", str(self.output_dir),
                    "--batch_size", str(batch_size)
                ]
            else:
                if not original_script.exists():
                    raise RuntimeError(f"keypoint_detector.py not found at {original_script}")
                cmd = [
                    sys.executable, "keypoint_detector.py",
                    "--path", str(self.output_dir)
                ]
            
            print(f"Running: {' '.join(cmd)}")
            # 不捕获输出，直接打印到终端，方便看到完整错误
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                raise RuntimeError(f"Landmark detection failed with exit code {result.returncode}")
                
        finally:
            os.chdir(original_dir)
            
    def run_iris_segmentation(self, use_fast_mode: bool = True):
        """Run iris segmentation
        
        Args:
            use_fast_mode: Use optimized fast script with progress bar and interpolation
        """
        fast_script = self.preprocess_dir / "iris_fast.py"
        original_script = self.preprocess_dir / "iris.py"
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.preprocess_dir)
            
            if use_fast_mode and fast_script.exists():
                print(f"Using FAST mode with progress tracking and interpolation")
                cmd = [
                    sys.executable, "iris_fast.py",
                    "--path", str(self.output_dir)
                ]
            else:
                if not original_script.exists():
                    raise RuntimeError(f"iris.py not found at {original_script}")
                cmd = [
                    sys.executable, "iris.py",
                    "--path", str(self.output_dir)
                ]
            
            print(f"Running: {' '.join(cmd)}")
            # 不捕获输出，直接打印到终端，方便看到完整错误
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                raise RuntimeError(f"Iris segmentation failed with exit code {result.returncode}")
                
        finally:
            os.chdir(original_dir)
            
    def run_face_parsing(self, use_fast_mode: bool = True, batch_size: int = 8, num_workers: int = 4):
        """Run semantic face parsing
        
        Args:
            use_fast_mode: Use optimized batch processing script
            batch_size: Batch size for GPU processing
            num_workers: Number of data loading workers
        """
        if not self.parser_dir.exists():
            raise RuntimeError(f"face-parsing not found at {self.parser_dir}")
            
        # Check for pretrained model
        parser_ckpt = self.parser_dir / "res" / "cp" / "79999_iter.pth"
        if not parser_ckpt.exists():
            # Try alternate location
            parser_ckpt = self.parser_dir / "79999_iter.pth"
            
        if not parser_ckpt.exists():
            raise RuntimeError(
                f"Face parsing checkpoint not found. "
                f"Please download from: https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812 "
                f"and place it at {self.parser_dir / 'res' / 'cp' / '79999_iter.pth'}"
            )
        
        image_dir = self.output_dir / "image"
        semantic_dir = self.output_dir / "semantic"
        
        fast_script = self.parser_dir / "test_fast.py"
        original_script = self.parser_dir / "test.py"
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.parser_dir)
            
            if use_fast_mode and fast_script.exists():
                print(f"Using FAST mode: batch_size={batch_size}, num_workers={num_workers}")
                cmd = [
                    sys.executable, "test_fast.py",
                    "--dspth", str(image_dir),
                    "--respth", str(semantic_dir),
                    "--batch_size", str(batch_size),
                    "--num_workers", str(num_workers),
                    "--no_color"  # Skip color visualization to save time
                ]
            else:
                cmd = [
                    sys.executable, "test.py",
                    "--dspth", str(image_dir),
                    "--respth", str(semantic_dir)
                ]
            
            print(f"Running: {' '.join(cmd)}")
            # 不捕获输出，直接打印到终端，方便看到完整错误
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                raise RuntimeError(f"Face parsing failed with exit code {result.returncode}")
                
        finally:
            os.chdir(original_dir)
            
    def generate_camera_json(self):
        """Generate camera parameters JSON file"""
        image_dir = self.output_dir / "image"
        num_frames = len(list(image_dir.glob("*.png")))
        
        # Create transforms JSON (similar to NeRF/3DGS format)
        transforms = {
            "camera_angle_x": 2.0 * 3.14159 * self.fx / self.resize,
            "camera_angle_y": 2.0 * 3.14159 * self.fy / self.resize,
            "fl_x": self.fx,
            "fl_y": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.resize,
            "h": self.resize,
            "frames": []
        }
        
        for i in range(1, num_frames + 1):
            frame_data = {
                "file_path": f"image/{i}.png",
                "mask_path": f"mask/{i}.png",
                "frame_id": i,
                # Default camera pose (identity - frontal view)
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 3],  # Z = 3 distance
                    [0, 0, 0, 1]
                ]
            }
            transforms["frames"].append(frame_data)
        
        # Save transforms
        transforms_path = self.output_dir / "transforms_train.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        print(f"Created: {transforms_path}")
        
        # Also create a simple metadata file
        metadata = {
            "subject_name": self.output_dir.name,
            "num_frames": num_frames,
            "resolution": self.resize,
            "fps": self.fps,
            "camera": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy
            }
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Created: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Video preprocessing for Relightable Gaussian Talking Head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps
  python preprocess_video.py --video video.mp4 --output ./output/
  
  # Resume from step 4 (skip steps 1-3)
  python preprocess_video.py --video video.mp4 --output ./output/ --start-step 4
  
  # Run only steps 4 to 6
  python preprocess_video.py --video video.mp4 --output ./output/ --start-step 4 --end-step 6

Steps:
  1. Video cropping and resizing
  2. Background segmentation (MODNet)
  3. Extract frames
  4. DECA FLAME estimation
  5. Face landmark detection
  6. Iris segmentation
  7. Semantic segmentation (face-parsing)
  8. Generate camera JSON
        """
    )
    
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory")
    parser.add_argument("--fps", type=int, default=25,
                       help="Target FPS (default: 25)")
    parser.add_argument("--crop", type=str, default=None,
                       help="Crop region in format 'w:h:x:y' (optional)")
    parser.add_argument("--resize", type=int, default=512,
                       help="Output resolution (default: 512)")
    parser.add_argument("--fx", type=float, default=None,
                       help="Camera focal length x (optional)")
    parser.add_argument("--fy", type=float, default=None,
                       help="Camera focal length y (optional)")
    parser.add_argument("--cx", type=float, default=None,
                       help="Camera principal point x (optional)")
    parser.add_argument("--cy", type=float, default=None,
                       help="Camera principal point y (optional)")
    
    # Step control
    parser.add_argument("--start-step", type=int, default=1,
                       help="Start from this step (1-8, default: 1)")
    parser.add_argument("--end-step", type=int, default=8,
                       help="End at this step (1-8, default: 8)")
    
    args = parser.parse_args()
    
    # Validate step range
    if args.start_step < 1 or args.start_step > 8:
        print(f"Error: --start-step must be between 1 and 8, got {args.start_step}")
        sys.exit(1)
    if args.end_step < 1 or args.end_step > 8:
        print(f"Error: --end-step must be between 1 and 8, got {args.end_step}")
        sys.exit(1)
    if args.start_step > args.end_step:
        print(f"Error: --start-step ({args.start_step}) cannot be greater than --end-step ({args.end_step})")
        sys.exit(1)
    
    # Validate input (only required for step 1)
    if args.start_step == 1 and not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        crop=args.crop,
        resize=args.resize,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy
    )
    
    # Run preprocessing with step range
    preprocessor.run_all(start_step=args.start_step, end_step=args.end_step)


if __name__ == "__main__":
    main()
