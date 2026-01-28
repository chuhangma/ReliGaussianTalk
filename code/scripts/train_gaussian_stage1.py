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
import random

from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter

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


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        # ImageNet normalization (correct values)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, x, y):
        # x, y: (B, C, H, W) in [0, 1]
        # Clamp to valid range to prevent NaN from normalization
        x = torch.clamp(x, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)
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
        self.skip_first_eval = kwargs.get('skip_first_eval', True)
        
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
        
        # Initialize TensorBoard writer
        # TensorBoard: 从配置文件读取路径，如果未设置则使用默认路径
        tensorboard_base = self.conf.get_string('train.tensorboard_dir', default='')
        if tensorboard_base and tensorboard_base.strip():
            # 使用配置文件中的路径，添加实验名称作为子目录
            subject_name = self.conf.get_string('dataset.subject_name', default='unknown')
            tensorboard_dir = os.path.join(tensorboard_base, subject_name, 'stage1')
        else:
            # 默认路径：在训练输出目录下
            tensorboard_dir = os.path.join(self.train_dir, 'tensorboard')
        utils.mkdir_ifnotexists(tensorboard_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard logs: {tensorboard_dir}")
        
        # Load checkpoint if continuing
        if self.is_continue:
            self._load_checkpoint()
            
    def _initialize_gaussians(self):
        """Initialize Gaussians from FLAME mesh or point cloud
        
        Following GaussianTalker's approach, we use direct mesh vertices
        for initialization by default (use_mesh_vertices=True).
        This typically gives ~5023 Gaussians for FLAME or ~34650 for BFM.
        """
        flame_path = self.conf.get_string('model.flame_model_path', '')
        
        # Config options:
        # model.use_mesh_vertices: True = use exact mesh vertices (recommended)
        # model.num_gaussians: only used if use_mesh_vertices=False
        use_mesh_vertices = self.conf.get_bool('model.use_mesh_vertices', True)
        # Only read num_gaussians if we're NOT using mesh vertices
        if use_mesh_vertices:
            num_samples = None  # Will use exact mesh vertex count
        else:
            num_samples = self.conf.get_int('model.num_gaussians', 5023)  # Default to FLAME vertex count
        
        print(f"[Stage1] Gaussian initialization mode: {'mesh vertices' if use_mesh_vertices else f'{num_samples} samples'}")
        
        init_dataset = GaussianInitDataset(
            flame_model_path=flame_path,
            shape_params=self.train_dataset.shape_params,
            num_samples=num_samples,
            use_mesh_vertices=use_mesh_vertices
        )
        
        points, colors, normals = init_dataset.generate_init_points()
        
        self.gaussian_model.initialize_from_point_cloud(
            points=points.cuda(),
            colors=colors.cuda(),
            normals=normals.cuda()
        )
        
        print(f"Initialized {self.gaussian_model.num_gaussians} Gaussians from {'FLAME mesh' if use_mesh_vertices else 'sampled points'}")
        
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
        
        # Epoch-level loss accumulators for TensorBoard
        epoch_loss_accum = {}
        
        for epoch in range(self.start_epoch, self.nepochs):
            self.gaussian_model.train()
            epoch_loss = 0.0
            epoch_loss_accum = {k: 0.0 for k in ['rgb', 'ssim', 'vgg', 'normal_smooth', 'opacity', 'total']}
            
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
                
                # IMPORTANT: zero_grad BEFORE forward pass
                self.optimizer.zero_grad()
                
                # Forward pass
                loss, loss_dict, rendered_images = self._train_step(model_input, ground_truth)
                
                # Backward
                loss.backward()
                
                # Accumulate gradients for densification (AFTER backward, BEFORE step)
                self._accumulate_gradients()
                
                # Gradient clipping (optional but helps stability)
                torch.nn.utils.clip_grad_norm_(self.gaussian_model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Accumulate losses for epoch average
                for k, v in loss_dict.items():
                    if k in epoch_loss_accum:
                        epoch_loss_accum[k] += v
                
                # TensorBoard: Log iteration-level losses (every 50 iterations)
                if self.iteration % 50 == 0:
                    for loss_name, loss_value in loss_dict.items():
                        self.writer.add_scalar(f'train_iter/{loss_name}', loss_value, self.iteration)
                
                # TensorBoard: Log images (every 200 iterations)
                if self.iteration % 200 == 0 and rendered_images is not None:
                    self._log_images_to_tensorboard(rendered_images, self.iteration)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'rgb': f'{loss_dict["rgb"]:.4f}',
                    'ssim': f'{loss_dict.get("ssim", 0):.4f}',
                })
                
                # Densification (adaptive control)
                if self.iteration % 500 == 0 and epoch < self.nepochs // 2:
                    self._densification_step()
                    # Log Gaussian count to TensorBoard
                    self.writer.add_scalar('model/num_gaussians', self.gaussian_model.num_gaussians, self.iteration)
                    
            # End of epoch
            self.scheduler.step()
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")
            
            # TensorBoard: Log epoch-level average losses
            num_batches = len(self.train_dataloader)
            for loss_name, loss_sum in epoch_loss_accum.items():
                self.writer.add_scalar(f'train_epoch/{loss_name}', loss_sum / num_batches, epoch)
            
            # TensorBoard: Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # TensorBoard: Log Gaussian model statistics
            self._log_gaussian_stats_to_tensorboard(epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
            # Evaluation (skip epoch 0 if skip_first_eval is True)
            if epoch % 10 == 0 and not (epoch == 0 and self.skip_first_eval):
                eval_metrics = self._evaluate(epoch)
                # Log evaluation metrics to TensorBoard
                if eval_metrics:
                    for metric_name, metric_value in eval_metrics.items():
                        self.writer.add_scalar(f'eval/{metric_name}', metric_value, epoch)
                
        # Close TensorBoard writer
        self.writer.close()
        print("Training completed!")
    
    def _log_images_to_tensorboard(self, rendered_images, step):
        """Log rendered images to TensorBoard"""
        # rendered_images: dict with 'rendered', 'gt', 'normal', 'depth', etc.
        if 'rendered' in rendered_images:
            self.writer.add_image('train/rendered', rendered_images['rendered'], step)
        if 'gt' in rendered_images:
            self.writer.add_image('train/ground_truth', rendered_images['gt'], step)
        if 'normal' in rendered_images:
            # Normalize normal map for visualization
            normal_viz = (rendered_images['normal'] + 1) / 2
            self.writer.add_image('train/normal', normal_viz, step)
        if 'depth' in rendered_images:
            # Normalize depth for visualization
            depth = rendered_images['depth']
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            self.writer.add_image('train/depth', depth_norm, step)
        if 'albedo' in rendered_images:
            self.writer.add_image('train/albedo', rendered_images['albedo'], step)
    
    def _log_gaussian_stats_to_tensorboard(self, epoch):
        """Log Gaussian model statistics to TensorBoard"""
        with torch.no_grad():
            # Check if model has valid Gaussians
            if self.gaussian_model.num_gaussians == 0:
                print("Warning: No Gaussians to log stats for")
                return
                
            # Number of Gaussians
            self.writer.add_scalar('model/num_gaussians', self.gaussian_model.num_gaussians, epoch)
            
            # Opacity statistics
            opacity = self.gaussian_model.get_opacity
            if opacity.numel() > 0 and not torch.isnan(opacity).any():
                self.writer.add_scalar('model/opacity_mean', opacity.mean().item(), epoch)
                self.writer.add_scalar('model/opacity_std', opacity.std().item(), epoch)
                # Only add histogram if data is valid
                try:
                    self.writer.add_histogram('model/opacity_dist', opacity.cpu(), epoch)
                except ValueError:
                    pass  # Skip if histogram is empty
            
            # Scale statistics
            scales = self.gaussian_model.get_scaling
            if scales.numel() > 0 and not torch.isnan(scales).any():
                self.writer.add_scalar('model/scale_mean', scales.mean().item(), epoch)
                self.writer.add_scalar('model/scale_max', scales.max().item(), epoch)
                try:
                    self.writer.add_histogram('model/scale_dist', scales.cpu().flatten(), epoch)
                except ValueError:
                    pass
            
            # Position statistics
            xyz = self.gaussian_model.get_xyz
            if xyz.numel() > 0 and not torch.isnan(xyz).any():
                self.writer.add_scalar('model/xyz_range', (xyz.max() - xyz.min()).item(), epoch)
            
            # Color/Albedo statistics
            if hasattr(self.gaussian_model, '_albedo') and self.gaussian_model._albedo is not None:
                albedo = self.gaussian_model.get_albedo
                if albedo.numel() > 0 and not torch.isnan(albedo).any():
                    self.writer.add_scalar('model/albedo_mean', albedo.mean().item(), epoch)
        
    def _train_step(self, model_input, ground_truth):
        """Single training step
        
        Returns:
            total_loss: The total loss tensor
            loss_dict: Dictionary of individual loss values
            rendered_images: Dictionary of rendered images for TensorBoard (detached)
        """
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
        rendered_depth = rendered.get('depth', None)
        rendered_albedo = rendered.get('albedo', None)
        
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
        
        # Prepare images for TensorBoard (detach to avoid memory issues)
        rendered_images = {
            'rendered': rendered_rgb.detach().clamp(0, 1),
            'gt': gt_rgb.detach().clamp(0, 1),
        }
        if rendered_normal is not None:
            rendered_images['normal'] = rendered_normal.detach()
        if rendered_depth is not None:
            rendered_images['depth'] = rendered_depth.detach()
        if rendered_albedo is not None:
            rendered_images['albedo'] = rendered_albedo.detach().clamp(0, 1)
        
        return total_loss, loss_dict, rendered_images
    
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
            cloned = self.gaussian_model.densify_and_clone(grads, grad_threshold, scene_extent)
            
            # Prune low-opacity Gaussians
            opacity_threshold = 0.005
            prune_mask = self.gaussian_model.get_opacity.squeeze() < opacity_threshold
            pruned = False
            if prune_mask.sum() > 0:
                pruned = self.gaussian_model.prune_points(prune_mask)
            
            # Refresh optimizer if parameters changed
            if self.gaussian_model.needs_optimizer_refresh():
                self._refresh_optimizer()
                self.gaussian_model.clear_optimizer_refresh_flag()
                
            # print(f"  Densification: {self.gaussian_model.num_gaussians} Gaussians")
    
    def _accumulate_gradients(self):
        """Accumulate gradients for adaptive density control"""
        if self.gaussian_model._xyz.grad is not None:
            # Accumulate gradient magnitude for each Gaussian
            grad_norm = self.gaussian_model._xyz.grad.norm(dim=-1, keepdim=True)
            
            # Ensure gradient_accum has correct shape (may change after densification)
            if self.gaussian_model.xyz_gradient_accum is None or \
               self.gaussian_model.xyz_gradient_accum.shape[0] != grad_norm.shape[0]:
                device = self.gaussian_model._xyz.device
                self.gaussian_model.xyz_gradient_accum = torch.zeros(
                    (self.gaussian_model.num_gaussians, 1), device=device
                )
                self.gaussian_model.denom = torch.zeros(
                    (self.gaussian_model.num_gaussians, 1), device=device
                )
            
            self.gaussian_model.xyz_gradient_accum += grad_norm.detach()
            self.gaussian_model.denom += 1
    
    def _refresh_optimizer(self):
        """Recreate optimizer after densification/pruning changes parameters"""
        # Get current learning rates from existing optimizer
        current_lrs = {}
        for group in self.optimizer.param_groups:
            if 'name' in group:
                current_lrs[group['name']] = group['lr']
        
        # Create new optimizer with updated parameters
        new_params = self.gaussian_model.get_training_params()
        
        # Restore learning rates
        for group in new_params:
            if group.get('name') in current_lrs:
                group['lr'] = current_lrs[group['name']]
        
        self.optimizer = torch.optim.Adam(new_params)
        
        # Note: We lose momentum state, but this is acceptable during densification
        # print("  Optimizer refreshed with new parameters")
            
    def _evaluate(self, epoch):
        """Evaluate on test set and save normal maps
        
        Returns:
            eval_metrics: Dictionary of evaluation metrics for TensorBoard
        """
        self.gaussian_model.eval()
        
        eval_epoch_dir = os.path.join(self.eval_dir, f'epoch_{epoch}')
        normal_dir = os.path.join(eval_epoch_dir, 'normal')
        rgb_dir = os.path.join(eval_epoch_dir, 'rgb')
        
        utils.mkdir_ifnotexists(eval_epoch_dir)
        utils.mkdir_ifnotexists(normal_dir)
        utils.mkdir_ifnotexists(rgb_dir)
        
        print(f"\nEvaluating epoch {epoch}...")
        
        # Metrics accumulators
        total_psnr = 0.0
        total_l1_loss = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        # For TensorBoard image logging
        sample_images = []
        
        with torch.no_grad():
            for batch_idx, (indices, model_input, ground_truth) in enumerate(tqdm(self.test_dataloader)):
                # Move to GPU
                for k, v in model_input.items():
                    if isinstance(v, torch.Tensor):
                        model_input[k] = v.cuda()
                
                for k, v in ground_truth.items():
                    if isinstance(v, torch.Tensor):
                        ground_truth[k] = v.cuda()
                
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
                
                H, W = self.img_res
                rendered_rgb = rendered['rgb']  # (C, H, W)
                # Clamp to valid range and handle NaN
                rendered_rgb = torch.nan_to_num(rendered_rgb, nan=0.0, posinf=1.0, neginf=0.0)
                rendered_rgb = torch.clamp(rendered_rgb, 0.0, 1.0)
                
                gt_rgb = ground_truth['rgb'].view(1, H, W, 3)[0].permute(2, 0, 1)  # (C, H, W)
                gt_rgb = (gt_rgb + 1) / 2  # [-1,1] -> [0,1]
                gt_rgb = torch.clamp(gt_rgb, 0.0, 1.0)
                
                # Compute metrics
                # PSNR
                mse = torch.mean((rendered_rgb - gt_rgb) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                total_psnr += psnr.item()
                
                # L1 loss
                l1 = torch.mean(torch.abs(rendered_rgb - gt_rgb))
                total_l1_loss += l1.item()
                
                # SSIM
                ssim_val = 1.0 - self.ssim_loss(rendered_rgb.unsqueeze(0), gt_rgb.unsqueeze(0))
                total_ssim += ssim_val.item()
                
                num_samples += 1
                
                # Collect sample images for TensorBoard (first 4 samples)
                if len(sample_images) < 4:
                    sample_images.append({
                        'rendered': rendered_rgb.cpu(),
                        'gt': gt_rgb.cpu(),
                        'normal': rendered.get('normal', torch.zeros(3, H, W)).cpu() if 'normal' in rendered else None
                    })
                
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
        
        # Compute average metrics
        eval_metrics = {
            'psnr': total_psnr / max(num_samples, 1),
            'l1_loss': total_l1_loss / max(num_samples, 1),
            'ssim': total_ssim / max(num_samples, 1),
        }
        
        print(f"Evaluation metrics: PSNR={eval_metrics['psnr']:.2f}, SSIM={eval_metrics['ssim']:.4f}, L1={eval_metrics['l1_loss']:.4f}")
        
        # Log sample evaluation images to TensorBoard
        if sample_images:
            import torchvision
            # Create a grid of rendered vs GT images
            rendered_grid = torch.stack([img['rendered'] for img in sample_images])
            gt_grid = torch.stack([img['gt'] for img in sample_images])
            
            # Make grid (2 rows: rendered on top, GT on bottom)
            combined = torch.cat([rendered_grid, gt_grid], dim=0)
            grid = torchvision.utils.make_grid(combined, nrow=4, normalize=False)
            self.writer.add_image('eval/comparison', grid, epoch)
            
            # Log normal maps
            normals_available = [img['normal'] for img in sample_images if img['normal'] is not None]
            if normals_available:
                normal_grid = torch.stack(normals_available)
                normal_grid = (normal_grid + 1) / 2  # Normalize to [0, 1]
                grid_normal = torchvision.utils.make_grid(normal_grid, nrow=4, normalize=False)
                self.writer.add_image('eval/normals', grid_normal, epoch)
        
        print(f"Saved evaluation results to {eval_epoch_dir}")
        
        return eval_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Gaussian Geometry Training')
    parser.add_argument('--conf', type=str, required=True, help='Config file path')
    parser.add_argument('--nepoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--is_continue', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='latest', help='Checkpoint to load')
    parser.add_argument('--load_path', type=str, default='', help='Path to load checkpoint from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip_first_eval', action='store_true', default=False,
                        help='Skip validation at epoch 0 (for faster debugging)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    runner = Stage1TrainRunner(
        conf=args.conf,
        nepochs=args.nepoch,
        is_continue=args.is_continue,
        checkpoint=args.checkpoint,
        load_path=args.load_path,
        skip_first_eval=args.skip_first_eval
    )
    
    runner.run()
