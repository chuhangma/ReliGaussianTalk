"""
Stage 1: Gaussian Geometry Training
Train the 3D Gaussian point cloud with basic appearance (no relighting yet).
This stage focuses on learning accurate geometry (positions, normals, scales).
"""

import os
import sys
sys.path.append('../code')

import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

from pyhocon import ConfigFactory

# Gaussian modules
from scene.relightable_gaussian_model import RelightableGaussianModel
from gaussian_renderer import RelightableGaussianRenderer
from datasets.gaussian_dataset import GaussianRelightDataset, GaussianInitDataset

# Utils
import utils.general as utils

# Losses
from pytorch3d.loss import chamfer_distance
from torchvision import transforms
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.229],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, x, y):
        # x, y: (B, C, H, W) in [0, 1]
        x = self.normalize(x)
        y = self.normalize(y)
        return nn.functional.l1_loss(self.vgg(x), self.vgg(y))


class SSIMLoss(nn.Module):
    """SSIM Loss"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
        
    def _create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.exp(-torch.arange(window_size).float().sub(window_size//2).pow(2) / (2*sigma**2))
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        mu1 = nn.functional.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = nn.functional.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = nn.functional.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class Stage1TrainRunner:
    """Stage 1: Train Gaussian geometry with basic appearance"""
    
    def __init__(self, **kwargs):
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.nepochs = kwargs.get('nepochs', 100)
        self.checkpoint = kwargs.get('checkpoint', 'latest')
        self.is_continue = kwargs.get('is_continue', False)
        self.load_path = kwargs.get('load_path', '')
        
        # Setup directories
        self.exps_folder = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname', 'GaussianRelight')
        
        train_split = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))
        self.expdir = os.path.join(self.exps_folder, self.subject, self.methodname)
        self.train_dir = os.path.join(self.expdir, train_split, 'train')
        self.eval_dir = os.path.join(self.expdir, train_split, 'eval')
        
        utils.mkdir_ifnotexists(self.train_dir)
        utils.mkdir_ifnotexists(self.eval_dir)
        
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        
        # Save config
        os.system(f'cp -r {kwargs["conf"]} "{os.path.join(self.train_dir, "runconf.conf")}"')
        
        print("="*60)
        print(f"Stage 1: Gaussian Geometry Training")
        print(f"Subject: {self.subject}")
        print(f"Output: {self.train_dir}")
        print("="*60)
        
        # Load dataset
        print("Loading dataset...")
        self.train_dataset = GaussianRelightDataset(
            data_folder=self.conf.get_string('dataset.data_folder'),
            subject_name=self.subject,
            json_name=self.conf.get_string('dataset.json_name'),
            use_semantics=self.conf.get_bool('loss.gt_w_seg', True),
            use_normals=False,  # Stage 1 doesn't need normals
            **self.conf.get_config('dataset.train')
        )
        
        self.test_dataset = GaussianRelightDataset(
            data_folder=self.conf.get_string('dataset.data_folder'),
            subject_name=self.subject,
            json_name=self.conf.get_string('dataset.json_name'),
            use_semantics=self.conf.get_bool('loss.gt_w_seg', True),
            use_normals=False,
            **self.conf.get_config('dataset.test')
        )
        
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.conf.get_int('train.batch_size', 4),
            shuffle=True,
            num_workers=4,
            collate_fn=self.train_dataset.collate_fn
        )
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn
        )
        
        self.img_res = self.train_dataset.img_res
        
        # Initialize Gaussian model
        print("Initializing Gaussian model...")
        self.gaussian_model = RelightableGaussianModel(
            sh_degree=0,  # No SH in Stage 1
            num_gaussians=self.conf.get_int('model.num_gaussians', 50000),
            with_motion_net=self.conf.get_bool('model.with_motion_net', True)
        ).cuda()
        
        # Initialize from FLAME or point cloud
        self._initialize_gaussians()
        
        # Initialize renderer
        self.renderer = RelightableGaussianRenderer(
            img_height=self.img_res[0],
            img_width=self.img_res[1],
            sh_degree=0,
            bg_color=[1.0, 1.0, 1.0]
        ).cuda()
        
        # Losses
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss().cuda()
        self.vgg_loss = VGGPerceptualLoss().cuda()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.gaussian_model.get_training_params(),
            lr=self.conf.get_float('train.learning_rate', 1e-3)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        # Training state
        self.start_epoch = 0
        self.iteration = 0
        
        # Load checkpoint if continuing
        if self.is_continue:
            self._load_checkpoint()
            
    def _initialize_gaussians(self):
        """Initialize Gaussians from FLAME mesh or point cloud"""
        flame_path = self.conf.get_string('model.flame_model_path', '')
        
        init_dataset = GaussianInitDataset(
            flame_model_path=flame_path,
            shape_params=self.train_dataset.shape_params,
            num_samples=self.conf.get_int('model.num_gaussians', 50000)
        )
        
        points, colors, normals = init_dataset.generate_init_points()
        
        self.gaussian_model.initialize_from_point_cloud(
            points=points.cuda(),
            colors=colors.cuda(),
            normals=normals.cuda()
        )
        
        print(f"Initialized {self.gaussian_model.num_gaussians} Gaussians")
        
    def _load_checkpoint(self):
        """Load checkpoint for continuing training"""
        ckpt_path = os.path.join(self.checkpoints_path, f'gaussian_{self.checkpoint}.pth')
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            self.gaussian_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.iteration = checkpoint.get('iteration', 0)
        else:
            print(f"Checkpoint {ckpt_path} not found, starting from scratch")
            
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'iteration': self.iteration,
            'model_state_dict': self.gaussian_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.checkpoints_path, 'gaussian_latest.pth'))
        
        # Save periodic
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(self.checkpoints_path, f'gaussian_{epoch}.pth'))
            
        # Save Gaussian PLY
        ply_path = os.path.join(self.checkpoints_path, f'point_cloud_{epoch}.ply')
        self.gaussian_model.save_ply(ply_path)
        
    def run(self):
        """Main training loop"""
        print(f"\nStarting training from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.nepochs):
            self.gaussian_model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            
            for batch_idx, (indices, model_input, ground_truth) in enumerate(pbar):
                self.iteration += 1
                
                # Move to GPU
                for k, v in model_input.items():
                    if isinstance(v, torch.Tensor):
                        model_input[k] = v.cuda()
                        
                for k, v in ground_truth.items():
                    if isinstance(v, torch.Tensor):
                        ground_truth[k] = v.cuda()
                
                # Forward pass
                loss, loss_dict = self._train_step(model_input, ground_truth)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'rgb': f'{loss_dict["rgb"]:.4f}',
                    'ssim': f'{loss_dict.get("ssim", 0):.4f}',
                })
                
                # Densification (adaptive control)
                if self.iteration % 500 == 0 and epoch < self.nepochs // 2:
                    self._densification_step()
                    
            # End of epoch
            self.scheduler.step()
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
            # Evaluation
            if epoch % 10 == 0:
                self._evaluate(epoch)
                
        print("Training completed!")
        
    def _train_step(self, model_input, ground_truth):
        """Single training step"""
        batch_size = ground_truth['rgb'].shape[0]
        H, W = self.img_res
        
        # Get Gaussian properties with optional deformation
        audio_features = model_input.get('audio_features', None)
        expression = model_input.get('expression', None)
        
        gaussian_output = self.gaussian_model(
            viewpoint_camera=None,  # Will use cam_pose from model_input
            audio_features=audio_features,
            expression_params=expression,
            sh_coeffs=None,  # No lighting in Stage 1
        )
        
        # Render
        rendered = self.renderer(
            xyz=gaussian_output['xyz'],
            scales=gaussian_output['scales'],
            rotations=gaussian_output['rotations'],
            opacity=gaussian_output['opacity'],
            colors=gaussian_output['albedo'],  # Use albedo as color in Stage 1
            normals=gaussian_output['normals'],
            camera_pose=model_input['cam_pose'][0],
            intrinsics=model_input['intrinsics'][0],
        )
        
        rendered_rgb = rendered['rgb']  # (C, H, W)
        rendered_normal = rendered.get('normal', None)
        
        # Prepare ground truth
        gt_rgb = ground_truth['rgb'].view(batch_size, H, W, 3)[0].permute(2, 0, 1)  # (C, H, W)
        gt_rgb = (gt_rgb + 1) / 2  # [-1,1] -> [0,1]
        
        mask = model_input['object_mask'].view(batch_size, H, W)[0]  # (H, W)
        
        # Losses
        loss_dict = {}
        
        # RGB loss
        rgb_loss = self.l1_loss(rendered_rgb * mask, gt_rgb * mask)
        loss_dict['rgb'] = rgb_loss.item()
        
        # SSIM loss
        ssim_loss = self.ssim_loss(
            rendered_rgb.unsqueeze(0), 
            gt_rgb.unsqueeze(0)
        )
        loss_dict['ssim'] = ssim_loss.item()
        
        # VGG perceptual loss
        vgg_loss = self.vgg_loss(
            rendered_rgb.unsqueeze(0),
            gt_rgb.unsqueeze(0)
        )
        loss_dict['vgg'] = vgg_loss.item()
        
        # Normal smoothness loss
        if rendered_normal is not None:
            normal_smooth = self._normal_smoothness_loss(rendered_normal)
            loss_dict['normal_smooth'] = normal_smooth.item()
        else:
            normal_smooth = 0.0
            
        # Opacity regularization (encourage sparse representation)
        opacity_reg = self.gaussian_model.get_opacity.mean()
        loss_dict['opacity'] = opacity_reg.item()
        
        # Total loss
        total_loss = (
            1.0 * rgb_loss +
            0.2 * ssim_loss +
            0.1 * vgg_loss +
            0.01 * normal_smooth +
            0.01 * opacity_reg
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _normal_smoothness_loss(self, normal_map):
        """Compute normal smoothness loss"""
        # normal_map: (3, H, W)
        dx = normal_map[:, :, 1:] - normal_map[:, :, :-1]
        dy = normal_map[:, 1:, :] - normal_map[:, :-1, :]
        return (dx.abs().mean() + dy.abs().mean()) * 0.5
    
    def _densification_step(self):
        """Adaptive density control for Gaussians"""
        # Get gradient statistics
        if self.gaussian_model.xyz_gradient_accum is not None:
            grads = self.gaussian_model.xyz_gradient_accum / (self.gaussian_model.denom + 1e-7)
            grads = grads.squeeze()
            
            grad_threshold = 0.0002
            scene_extent = 2.0
            
            # Clone small Gaussians with large gradients
            self.gaussian_model.densify_and_clone(grads, grad_threshold, scene_extent)
            
            # Prune low-opacity Gaussians
            opacity_threshold = 0.005
            prune_mask = self.gaussian_model.get_opacity.squeeze() < opacity_threshold
            if prune_mask.sum() > 0:
                self.gaussian_model.prune_points(prune_mask)
                
            print(f"  Densification: {self.gaussian_model.num_gaussians} Gaussians")
            
    def _evaluate(self, epoch):
        """Evaluate on test set and save normal maps"""
        self.gaussian_model.eval()
        
        eval_epoch_dir = os.path.join(self.eval_dir, f'epoch_{epoch}')
        normal_dir = os.path.join(eval_epoch_dir, 'normal')
        rgb_dir = os.path.join(eval_epoch_dir, 'rgb')
        
        utils.mkdir_ifnotexists(eval_epoch_dir)
        utils.mkdir_ifnotexists(normal_dir)
        utils.mkdir_ifnotexists(rgb_dir)
        
        print(f"\nEvaluating epoch {epoch}...")
        
        with torch.no_grad():
            for batch_idx, (indices, model_input, ground_truth) in enumerate(tqdm(self.test_dataloader)):
                # Move to GPU
                for k, v in model_input.items():
                    if isinstance(v, torch.Tensor):
                        model_input[k] = v.cuda()
                
                # Forward
                gaussian_output = self.gaussian_model(
                    viewpoint_camera=None,
                    audio_features=model_input.get('audio_features', None),
                    expression_params=model_input.get('expression', None),
                    sh_coeffs=None,
                )
                
                rendered = self.renderer(
                    xyz=gaussian_output['xyz'],
                    scales=gaussian_output['scales'],
                    rotations=gaussian_output['rotations'],
                    opacity=gaussian_output['opacity'],
                    colors=gaussian_output['albedo'],
                    normals=gaussian_output['normals'],
                    camera_pose=model_input['cam_pose'][0],
                    intrinsics=model_input['intrinsics'][0],
                )
                
                # Save normal map (for Stage 2)
                if 'normal' in rendered:
                    normal_img = rendered['normal'].permute(1, 2, 0).cpu().numpy()
                    normal_img = ((normal_img + 1) / 2 * 255).astype(np.uint8)
                    
                    frame_id = model_input['img_name'][0, 0].item()
                    normal_path = os.path.join(normal_dir, f'{frame_id:05d}.png')
                    
                    from PIL import Image
                    Image.fromarray(normal_img).save(normal_path)
                
                # Save rendered RGB
                rgb_img = rendered['rgb'].permute(1, 2, 0).cpu().numpy()
                rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
                
                frame_id = model_input['img_name'][0, 0].item()
                rgb_path = os.path.join(rgb_dir, f'{frame_id:05d}.png')
                
                from PIL import Image
                Image.fromarray(rgb_img).save(rgb_path)
        
        print(f"Saved evaluation results to {eval_epoch_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Gaussian Geometry Training')
    parser.add_argument('--conf', type=str, required=True, help='Config file path')
    parser.add_argument('--nepoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--is_continue', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='latest', help='Checkpoint to load')
    parser.add_argument('--load_path', type=str, default='', help='Path to load checkpoint from')
    
    args = parser.parse_args()
    
    runner = Stage1TrainRunner(
        conf=args.conf,
        nepochs=args.nepoch,
        is_continue=args.is_continue,
        checkpoint=args.checkpoint,
        load_path=args.load_path
    )
    
    runner.run()
