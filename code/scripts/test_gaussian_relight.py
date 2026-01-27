"""
Test/Inference Script for Relightable Gaussian Talking Head
Supports:
1. Single frame rendering with custom lighting
2. Video generation with rotating light
3. Talking head animation with audio input
"""

import os
import sys
sys.path.append('../code')

import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
import cv2
from PIL import Image

from pyhocon import ConfigFactory

# Gaussian modules
from scene.relightable_gaussian_model import RelightableGaussianModel
from gaussian_renderer import RelightableGaussianRenderer
from datasets.gaussian_dataset import GaussianRelightDataset

# Relighting networks
import model.resnet_network as ResNet
import model.unet_network as UNet
from utils.light_util import add_SHlight, normal_shading_sh, normalize

import utils.general as utils


class GaussianRelightTester:
    """Test/inference for relightable Gaussian talking head"""
    
    def __init__(self, **kwargs):
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.checkpoint = kwargs.get('checkpoint', 'latest')
        self.output_dir = kwargs.get('output_dir', './output')
        self.mode = kwargs.get('mode', 'video')  # 'single', 'video', 'comparison'
        
        # Setup directories
        self.exps_folder = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname', 'GaussianRelight')
        
        train_split = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))
        self.expdir = os.path.join(self.exps_folder, self.subject, self.methodname)
        
        # Stage 1 checkpoint path
        self.stage1_dir = os.path.join(self.expdir, train_split, 'train')
        # Stage 2 checkpoint path
        self.stage2_dir = os.path.join(self.expdir, train_split, 'train_relight')
        
        utils.mkdir_ifnotexists(self.output_dir)
        
        print("="*60)
        print(f"Relightable Gaussian Tester")
        print(f"Subject: {self.subject}")
        print(f"Output: {self.output_dir}")
        print("="*60)
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = GaussianRelightDataset(
            data_folder=self.conf.get_string('dataset.data_folder'),
            subject_name=self.subject,
            json_name=self.conf.get_string('dataset.json_name'),
            use_semantics=self.conf.get_bool('loss.gt_w_seg', True),
            use_normals=True,
            **self.conf.get_config('dataset.test')
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.dataset.collate_fn
        )
        
        self.img_res = self.dataset.img_res
        
        # Load models
        print("Loading models...")
        self._load_models()
        
        # SH constant factor
        self.constant_factor = self._get_sh_constant_factor().cuda()
        
        # Predefined lighting conditions
        self.lighting_presets = {
            'frontal': [0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'left': [0.3, 0.0, 0.3, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            'right': [0.3, 0.0, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            'top': [0.3, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'bottom': [0.3, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'ambient': [0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'dramatic': [0.2, 0.0, 0.3, 0.9, 0.0, 0.0, 0.2, 0.0, 0.0],
        }
        
    def _load_models(self):
        """Load Gaussian model and relighting networks"""
        # Load Gaussian model
        self.gaussian_model = RelightableGaussianModel(
            sh_degree=2,
            num_gaussians=self.conf.get_int('model.num_gaussians', 50000),
            with_motion_net=self.conf.get_bool('model.with_motion_net', True)
        ).cuda()
        
        stage1_ckpt = os.path.join(self.stage1_dir, 'checkpoints', f'gaussian_{self.checkpoint}.pth')
        if os.path.exists(stage1_ckpt):
            print(f"Loading Gaussian model from {stage1_ckpt}")
            checkpoint = torch.load(stage1_ckpt)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print(f"Warning: Stage 1 checkpoint not found at {stage1_ckpt}")
        
        # Initialize renderer
        self.renderer = RelightableGaussianRenderer(
            img_height=self.img_res[0],
            img_width=self.img_res[1],
            sh_degree=2,
            bg_color=[1.0, 1.0, 1.0]
        ).cuda()
        
        # Load relighting networks
        self.albedo_net = ResNet.Albedo_ResnetGenerator().cuda()
        self.normal_net = ResNet.ResnetGenerator(input_nc=6).cuda()
        self.spec_net = UNet.UnetGenerator(output_nc=1, input_nc=6).cuda()
        
        stage2_ckpt = os.path.join(self.stage2_dir, 'checkpoints', f'relight_{self.checkpoint}.pth')
        if os.path.exists(stage2_ckpt):
            print(f"Loading relighting networks from {stage2_ckpt}")
            checkpoint = torch.load(stage2_ckpt)
            self.albedo_net.load_state_dict(checkpoint['albedo_net'])
            self.normal_net.load_state_dict(checkpoint['normal_net'])
            self.spec_net.load_state_dict(checkpoint['spec_net'])
            self.learned_light = checkpoint.get('light_var', torch.zeros(9))
        else:
            print(f"Warning: Stage 2 checkpoint not found at {stage2_ckpt}")
            self.learned_light = torch.zeros(9)
        
        # Set to eval mode
        self.gaussian_model.eval()
        self.albedo_net.eval()
        self.normal_net.eval()
        self.spec_net.eval()
        
    def _get_sh_constant_factor(self):
        """Get constant factors for SH computation"""
        pi = np.pi
        constant_factor = torch.tensor([
            1/np.sqrt(4*pi),
            ((2*pi)/3) * (np.sqrt(3/(4*pi))),
            ((2*pi)/3) * (np.sqrt(3/(4*pi))),
            ((2*pi)/3) * (np.sqrt(3/(4*pi))),
            (pi/4) * (3) * (np.sqrt(5/(12*pi))),
            (pi/4) * (3) * (np.sqrt(5/(12*pi))),
            (pi/4) * (3) * (np.sqrt(5/(12*pi))),
            (pi/4) * (3/2) * (np.sqrt(5/(12*pi))),
            (pi/4) * (1/2) * (np.sqrt(5/(4*pi)))
        ]).float()
        return constant_factor
    
    def run(self, mode='video', lighting=None, num_frames=125):
        """Run inference
        
        Args:
            mode: 'single', 'video', 'comparison'
            lighting: Lighting preset name or custom SH coefficients
            num_frames: Number of frames for video mode
        """
        if mode == 'single':
            self._render_single(lighting)
        elif mode == 'video':
            self._render_video(num_frames)
        elif mode == 'comparison':
            self._render_comparison()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @torch.no_grad()
    def _render_single(self, lighting='frontal'):
        """Render a single frame with specified lighting"""
        # Get lighting coefficients
        if isinstance(lighting, str):
            if lighting in self.lighting_presets:
                light_coeffs = torch.tensor(self.lighting_presets[lighting]).float().cuda()
            else:
                raise ValueError(f"Unknown lighting preset: {lighting}")
        elif isinstance(lighting, (list, np.ndarray)):
            light_coeffs = torch.tensor(lighting).float().cuda()
        else:
            light_coeffs = torch.tensor(self.lighting_presets['frontal']).float().cuda()
        
        # Get first sample
        indices, model_input, ground_truth = next(iter(self.dataloader))
        
        for k, v in model_input.items():
            if isinstance(v, torch.Tensor):
                model_input[k] = v.cuda()
        for k, v in ground_truth.items():
            if isinstance(v, torch.Tensor):
                ground_truth[k] = v.cuda()
        
        # Render
        result = self._render_frame(model_input, ground_truth, light_coeffs)
        
        # Save
        frame_id = model_input['img_name'][0, 0].item()
        
        for name, img in result.items():
            save_path = os.path.join(self.output_dir, f'{frame_id:05d}_{name}.png')
            Image.fromarray(img).save(save_path)
            print(f"Saved {save_path}")
    
    @torch.no_grad()
    def _render_video(self, num_frames=125):
        """Render video with rotating light"""
        video_dir = os.path.join(self.output_dir, 'video_frames')
        utils.mkdir_ifnotexists(video_dir)
        
        print(f"Rendering {num_frames} frames with rotating light...")
        
        frames = []
        
        for frame_idx, (indices, model_input, ground_truth) in enumerate(tqdm(self.dataloader)):
            if frame_idx >= num_frames:
                break
                
            # Move to GPU
            for k, v in model_input.items():
                if isinstance(v, torch.Tensor):
                    model_input[k] = v.cuda()
            for k, v in ground_truth.items():
                if isinstance(v, torch.Tensor):
                    ground_truth[k] = v.cuda()
            
            # Rotating light
            theta = math.pi * (frame_idx % num_frames) / num_frames
            a = -math.cos(theta)  # X direction
            b = -math.sin(theta)  # Y direction
            
            light_coeffs = torch.cuda.FloatTensor([
                0.3,   # Ambient
                b,     # Y
                0.3,   # Z  
                a,     # X
                0.0, 0.0, 0.0, 0.0, 0.0
            ])
            
            # Render
            result = self._render_frame(model_input, ground_truth, light_coeffs)
            
            # Save frame
            relit_img = result['relit']
            frame_path = os.path.join(video_dir, f'{frame_idx:05d}.png')
            Image.fromarray(relit_img).save(frame_path)
            
            frames.append(relit_img)
        
        # Create video
        self._create_video(frames, os.path.join(self.output_dir, 'relit_video.mp4'))
        print(f"Video saved to {self.output_dir}/relit_video.mp4")
    
    @torch.no_grad()
    def _render_comparison(self):
        """Render comparison with multiple lighting conditions"""
        comparison_dir = os.path.join(self.output_dir, 'comparison')
        utils.mkdir_ifnotexists(comparison_dir)
        
        print("Rendering comparison with different lighting conditions...")
        
        # Get first few samples
        for frame_idx, (indices, model_input, ground_truth) in enumerate(tqdm(self.dataloader)):
            if frame_idx >= 5:  # Only first 5 frames
                break
                
            # Move to GPU
            for k, v in model_input.items():
                if isinstance(v, torch.Tensor):
                    model_input[k] = v.cuda()
            for k, v in ground_truth.items():
                if isinstance(v, torch.Tensor):
                    ground_truth[k] = v.cuda()
            
            frame_id = model_input['img_name'][0, 0].item()
            
            # Render with each lighting preset
            comparison_images = []
            
            # Original image
            H, W = self.img_res
            orig_img = ground_truth['rgb'].view(H, W, 3).cpu().numpy()
            orig_img = ((orig_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            comparison_images.append(('original', orig_img))
            
            for light_name, light_values in self.lighting_presets.items():
                light_coeffs = torch.tensor(light_values).float().cuda()
                result = self._render_frame(model_input, ground_truth, light_coeffs)
                comparison_images.append((light_name, result['relit']))
            
            # Create comparison grid
            grid = self._create_comparison_grid(comparison_images)
            grid_path = os.path.join(comparison_dir, f'{frame_id:05d}_comparison.png')
            Image.fromarray(grid).save(grid_path)
            
            # Also save individual images
            for name, img in comparison_images:
                save_path = os.path.join(comparison_dir, f'{frame_id:05d}_{name}.png')
                Image.fromarray(img).save(save_path)
        
        print(f"Comparison saved to {comparison_dir}")
    
    def _render_frame(self, model_input, ground_truth, light_coeffs):
        """Render a single frame with given lighting
        
        Returns:
            Dictionary with 'relit', 'albedo', 'normal', 'shading' images
        """
        H, W = self.img_res
        batch_size = ground_truth['rgb'].shape[0]
        
        # Prepare inputs
        rgb_input = ground_truth['rgb'].view(batch_size, H, W, 3).permute(0, 3, 1, 2)
        rgb_input = (rgb_input + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Get normal from dataset or render from Gaussians
        if 'normal' in ground_truth:
            normal = ground_truth['normal'].view(batch_size, H, W, 3).permute(0, 3, 1, 2)
        else:
            # Render normal from Gaussian model
            gaussian_out = self.gaussian_model(
                viewpoint_camera=None,
                audio_features=model_input.get('audio_features', None),
                expression_params=model_input.get('expression', None),
                sh_coeffs=None,
            )
            rendered = self.renderer(
                xyz=gaussian_out['xyz'],
                scales=gaussian_out['scales'],
                rotations=gaussian_out['rotations'],
                opacity=gaussian_out['opacity'],
                colors=gaussian_out['albedo'],
                normals=gaussian_out['normals'],
                camera_pose=model_input['cam_pose'][0],
                intrinsics=model_input['intrinsics'][0],
            )
            normal = rendered['normal'].unsqueeze(0)
        
        # Get mask
        face_mask = model_input['object_mask'].view(-1, 1, H, W)
        
        semantics = ground_truth.get('semantics', None)
        if semantics is not None:
            face_mask = torch.sum(semantics[:, :, :-2], dim=2).view(-1, 1, H, W) * face_mask
            shoulder_mask = semantics[:, :, -2].view(-1, 1, H, W)
        else:
            shoulder_mask = torch.zeros_like(face_mask)
        
        # Predict albedo
        albedo = self.albedo_net(rgb_input * 2 - 1)
        
        # Refine normal
        fine_normal = (self.normal_net(torch.cat([rgb_input * 2 - 1, normal], dim=1)) + 1) / 2
        fine_normal = fine_normal * face_mask * 2 - 1
        
        # Transform normal for SH computation (matching ReliTalk convention)
        fine_normal_sh = fine_normal.clone()
        fine_normal_sh[:, 1, ...] = -((fine_normal[:, 2, ...] + 1) / 2)
        fine_normal_sh[:, 2, ...] = fine_normal[:, 1, ...]
        
        # Compute shading
        shading = add_SHlight(self.constant_factor, fine_normal_sh, 
                             light_coeffs.view(1, -1, 1))
        
        # Normalize shading
        masked_shading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading) + 1e-7) * face_mask
        masked_shading_nonzero = masked_shading[masked_shading.nonzero(as_tuple=True)]
        if masked_shading_nonzero.numel() > 0:
            shading_min = torch.min(masked_shading_nonzero)
            shading_max = torch.max(masked_shading_nonzero)
            masked_shading = torch.clamp((masked_shading - shading_min) / (shading_max - shading_min + 1e-7), 0, 1.2)
        
        # Compute specular
        specmap = self.spec_net(torch.cat([rgb_input * 2 - 1, fine_normal], dim=1))
        scaled_specmap = (specmap + 1) / 2 * 0.2
        
        # Simple specular approximation
        spec = torch.clamp(masked_shading * 1.5, 0, 1)
        
        # Final relit image
        relit_image = albedo * (masked_shading + spec * scaled_specmap)
        relit_image = torch.clamp(relit_image, 0, 1)
        
        # Handle shoulder region
        relit_image = relit_image * (1 - shoulder_mask) + rgb_input * shoulder_mask
        
        # Handle background
        bg_mask = 1 - model_input['object_mask'].view(-1, 1, H, W).float()
        relit_image = relit_image * (1 - bg_mask) + bg_mask  # White background
        
        # Convert to numpy images
        def to_numpy_img(tensor):
            img = tensor[0].permute(1, 2, 0).cpu().numpy()
            return (img * 255).clip(0, 255).astype(np.uint8)
        
        return {
            'relit': to_numpy_img(relit_image),
            'albedo': to_numpy_img(albedo),
            'normal': to_numpy_img((fine_normal + 1) / 2),
            'shading': to_numpy_img(masked_shading.repeat(1, 3, 1, 1)),
        }
    
    def _create_comparison_grid(self, images, cols=4):
        """Create a grid of comparison images"""
        # images: list of (name, img) tuples
        H, W = self.img_res
        
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        grid = np.ones((rows * (H + 30), cols * W, 3), dtype=np.uint8) * 255
        
        for idx, (name, img) in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            y_start = row * (H + 30) + 25
            x_start = col * W
            
            grid[y_start:y_start + H, x_start:x_start + W] = img
            
            # Add label
            cv2.putText(grid, name, (x_start + 5, row * (H + 30) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return grid
    
    def _create_video(self, frames, output_path, fps=25):
        """Create video from frames"""
        H, W = self.img_res
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Relightable Gaussian Talking Head')
    parser.add_argument('--conf', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, default='latest', help='Checkpoint to load')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--mode', type=str, default='video', 
                       choices=['single', 'video', 'comparison'],
                       help='Rendering mode')
    parser.add_argument('--lighting', type=str, default='frontal',
                       help='Lighting preset for single mode')
    parser.add_argument('--num_frames', type=int, default=125,
                       help='Number of frames for video mode')
    
    args = parser.parse_args()
    
    tester = GaussianRelightTester(
        conf=args.conf,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )
    
    tester.run(
        mode=args.mode,
        lighting=args.lighting,
        num_frames=args.num_frames
    )
