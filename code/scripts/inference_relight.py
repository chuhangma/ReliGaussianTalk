"""
Inference script for Relightable Gaussian Talking Head

This script allows rendering of a trained model with:
1. Novel lighting conditions
2. Audio-driven animation
3. Expression control
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene.relightable_gaussian_model import RelightableGaussianModel
from gaussian_renderer import (
    RelightableGaussianRenderer,
    Camera,
    create_camera_from_params,
    RenderOutput
)


def load_model(checkpoint_path: str, device: torch.device) -> RelightableGaussianModel:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    model = RelightableGaussianModel(
        sh_degree=config.get('sh_degree', 3),
        with_motion_net=config.get('with_motion_net', True)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def load_audio_features(audio_path: str) -> torch.Tensor:
    """Load audio features from file"""
    if audio_path.endswith('.npy'):
        features = np.load(audio_path)
    elif audio_path.endswith('.wav') or audio_path.endswith('.mp3'):
        # Extract features using a pretrained model
        # This is a placeholder - you'd typically use DeepSpeech or similar
        print(f"Warning: Audio extraction not implemented. Using zeros.")
        features = np.zeros((100, 29))  # Dummy features
    else:
        raise ValueError(f"Unsupported audio format: {audio_path}")
    
    return torch.from_numpy(features).float()


def create_lighting_presets() -> dict:
    """Create predefined lighting presets"""
    presets = {
        'frontal': torch.tensor([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'left': torch.tensor([0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0]),
        'right': torch.tensor([0.4, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, -0.2, 0.0]),
        'top': torch.tensor([0.4, 0.3, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0]),
        'bottom': torch.tensor([0.4, -0.3, 0.0, 0.0, 0.0, -0.2, -0.1, 0.0, 0.0]),
        'ambient': torch.tensor([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'dramatic': torch.tensor([0.3, 0.0, 0.2, 0.4, 0.1, 0.1, 0.0, 0.2, 0.1]),
        'soft': torch.tensor([0.6, 0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]),
    }
    return presets


def interpolate_sh_coeffs(sh1: torch.Tensor, sh2: torch.Tensor, t: float) -> torch.Tensor:
    """Interpolate between two SH lighting conditions"""
    return sh1 * (1 - t) + sh2 * t


def render_single_frame(
    model: RelightableGaussianModel,
    renderer: RelightableGaussianRenderer,
    camera: Camera,
    sh_coeffs: torch.Tensor,
    audio_features: Optional[torch.Tensor] = None,
    expression_params: Optional[torch.Tensor] = None,
) -> RenderOutput:
    """Render a single frame"""
    with torch.no_grad():
        output = renderer(
            gaussians=model,
            camera=camera,
            audio_features=audio_features,
            expression_params=expression_params,
            sh_coeffs=sh_coeffs,
        )
    return output


def render_video(
    model: RelightableGaussianModel,
    renderer: RelightableGaussianRenderer,
    camera: Camera,
    num_frames: int,
    output_path: str,
    audio_features: Optional[torch.Tensor] = None,
    sh_coeffs_start: Optional[torch.Tensor] = None,
    sh_coeffs_end: Optional[torch.Tensor] = None,
    fps: int = 25,
    device: torch.device = torch.device('cuda'),
):
    """Render a video with optional lighting animation"""
    
    # Default lighting
    presets = create_lighting_presets()
    if sh_coeffs_start is None:
        sh_coeffs_start = presets['frontal'].to(device)
    if sh_coeffs_end is None:
        sh_coeffs_end = sh_coeffs_start
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (camera.image_width, camera.image_height)
    )
    
    # Render frames
    frames = []
    for i in tqdm(range(num_frames), desc="Rendering"):
        # Interpolate lighting
        t = i / max(num_frames - 1, 1)
        sh_coeffs = interpolate_sh_coeffs(sh_coeffs_start, sh_coeffs_end, t)
        
        # Get audio features for this frame
        if audio_features is not None:
            frame_audio = audio_features[i:i+1] if i < len(audio_features) else None
        else:
            frame_audio = None
        
        # Render
        output = render_single_frame(
            model, renderer, camera, sh_coeffs,
            audio_features=frame_audio
        )
        
        # Convert to numpy
        image = output.rendered_image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = image.transpose(1, 2, 0)  # (H, W, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        video_writer.write(image)
        frames.append(image)
    
    video_writer.release()
    print(f"Video saved to {output_path}")
    
    return frames


def render_relit_comparison(
    model: RelightableGaussianModel,
    renderer: RelightableGaussianRenderer,
    camera: Camera,
    output_dir: str,
    device: torch.device = torch.device('cuda'),
):
    """Render comparison images with different lighting conditions"""
    
    presets = create_lighting_presets()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for name, sh_coeffs in presets.items():
        output = render_single_frame(
            model, renderer, camera, 
            sh_coeffs.to(device)
        )
        
        # Save rendered image
        image = output.rendered_image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = image.transpose(1, 2, 0)
        
        save_path = output_path / f"relit_{name}.png"
        Image.fromarray(image).save(save_path)
        
        results[name] = image
        print(f"Saved {name} lighting to {save_path}")
    
    # Also save albedo and normal maps
    output = render_single_frame(
        model, renderer, camera,
        presets['ambient'].to(device)
    )
    
    # Albedo
    albedo = output.albedo_map.cpu().numpy()
    albedo = (albedo * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(albedo).save(output_path / "albedo.png")
    
    # Normal
    normal = output.normal_map.cpu().numpy()
    normal = ((normal + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(normal).save(output_path / "normal.png")
    
    # Depth
    if output.depth_map is not None:
        depth = output.depth_map.cpu().numpy()[0]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)
        Image.fromarray(depth).save(output_path / "depth.png")
    
    # Create comparison grid
    grid_images = []
    for name in ['frontal', 'left', 'right', 'top', 'soft', 'dramatic']:
        if name in results:
            grid_images.append(results[name])
    
    if len(grid_images) >= 6:
        # Create 2x3 grid
        row1 = np.concatenate(grid_images[:3], axis=1)
        row2 = np.concatenate(grid_images[3:6], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
        Image.fromarray(grid).save(output_path / "comparison_grid.png")
        print(f"Saved comparison grid to {output_path / 'comparison_grid.png'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Relightable Gaussian Talking Head Inference")
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path (directory or video file)')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'comparison'], 
                        default='comparison', help='Rendering mode')
    
    # Audio/animation
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file or features')
    
    # Lighting
    parser.add_argument('--lighting', type=str, default='frontal',
                        help='Lighting preset name or path to SH coefficients')
    parser.add_argument('--lighting_end', type=str, default=None,
                        help='End lighting for animation (video mode)')
    
    # Camera
    parser.add_argument('--image_width', type=int, default=512, help='Output image width')
    parser.add_argument('--image_height', type=int, default=512, help='Output image height')
    parser.add_argument('--fov', type=float, default=0.8, help='Field of view')
    
    # Video
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames for video')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Create renderer
    renderer = RelightableGaussianRenderer(
        img_height=args.image_height,
        img_width=args.image_width,
        sh_degree=model.max_sh_degree if hasattr(model, 'max_sh_degree') else 2,
        bg_color=(1.0, 1.0, 1.0)
    ).to(device)
    
    # Create camera
    R = torch.eye(3, device=device)
    T = torch.tensor([0., 0., 3.], device=device)
    camera = create_camera_from_params(
        R=R, T=T,
        fov_x=args.fov, fov_y=args.fov,
        image_width=args.image_width,
        image_height=args.image_height
    )
    
    # Get lighting
    presets = create_lighting_presets()
    if args.lighting in presets:
        sh_coeffs = presets[args.lighting].to(device)
    elif os.path.exists(args.lighting):
        sh_coeffs = torch.from_numpy(np.load(args.lighting)).float().to(device)
    else:
        print(f"Unknown lighting preset: {args.lighting}. Using frontal.")
        sh_coeffs = presets['frontal'].to(device)
    
    # Load audio if provided
    audio_features = None
    if args.audio:
        audio_features = load_audio_features(args.audio).to(device)
    
    # Run inference based on mode
    if args.mode == 'image':
        # Single image
        output = render_single_frame(
            model, renderer, camera, sh_coeffs,
            audio_features=audio_features[0:1] if audio_features is not None else None
        )
        
        image = output.rendered_image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = image.transpose(1, 2, 0)
        Image.fromarray(image).save(args.output)
        print(f"Saved image to {args.output}")
        
    elif args.mode == 'video':
        # Video rendering
        sh_coeffs_end = None
        if args.lighting_end:
            if args.lighting_end in presets:
                sh_coeffs_end = presets[args.lighting_end].to(device)
            elif os.path.exists(args.lighting_end):
                sh_coeffs_end = torch.from_numpy(np.load(args.lighting_end)).float().to(device)
        
        render_video(
            model, renderer, camera,
            num_frames=args.num_frames,
            output_path=args.output,
            audio_features=audio_features,
            sh_coeffs_start=sh_coeffs,
            sh_coeffs_end=sh_coeffs_end,
            fps=args.fps,
            device=device
        )
        
    elif args.mode == 'comparison':
        # Comparison of different lightings
        render_relit_comparison(
            model, renderer, camera,
            output_dir=args.output,
            device=device
        )


if __name__ == "__main__":
    main()
