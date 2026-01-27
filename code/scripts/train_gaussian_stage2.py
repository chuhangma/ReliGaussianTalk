"""
Stage 2: Gaussian Relighting Training
Train relighting networks (albedo estimation + lighting) using normal maps from Stage 1.
This stage learns to decompose appearance into albedo and lighting for relighting.
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

# Gaussian modules
from scene.relightable_gaussian_model import RelightableGaussianModel
from gaussian_renderer import RelightableGaussianRenderer
from datasets.gaussian_dataset import GaussianRelightDataset

# Relighting networks (reuse from ReliTalk)
import model.resnet_network as ResNet
import model.unet_network as UNet
from utils.light_util import add_SHlight, normal_shading, normalize

# Utils
import utils.general as utils

import torchvision
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
        x = self.normalize(x)
        y = self.normalize(y)
        return nn.functional.l1_loss(self.vgg(x), self.vgg(y))


class Stage2TrainRunner:
    """Stage 2: Train relighting networks with Gaussian rendering"""
    
    def __init__(self, **kwargs):
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.nepochs = kwargs.get('nepochs', 50)
        self.checkpoint = kwargs.get('checkpoint', 'latest')
        self.is_continue = kwargs.get('is_continue', False)
        self.load_path = kwargs.get('load_path', '')
        self.stage1_path = kwargs.get('stage1_path', '')
        
        # Setup directories
        self.exps_folder = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname', 'GaussianRelight')
        
        train_split = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))
        self.expdir = os.path.join(self.exps_folder, self.subject, self.methodname)
        self.train_dir = os.path.join(self.expdir, train_split, 'train_relight')
        self.eval_dir = os.path.join(self.expdir, train_split, 'eval_relight')
        
        utils.mkdir_ifnotexists(self.train_dir)
        utils.mkdir_ifnotexists(self.eval_dir)
        
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        
        # Save config
        os.system(f'cp -r {kwargs["conf"]} "{os.path.join(self.train_dir, "runconf.conf")}"')
        
        print("="*60)
        print(f"Stage 2: Gaussian Relighting Training")
        print(f"Subject: {self.subject}")
        print(f"Output: {self.train_dir}")
        print("="*60)
        
        # Load dataset (with normal maps from Stage 1)
        print("Loading dataset with normal maps...")
        self.train_dataset = GaussianRelightDataset(
            data_folder=self.conf.get_string('dataset.data_folder'),
            subject_name=self.subject,
            json_name=self.conf.get_string('dataset.json_name'),
            use_semantics=self.conf.get_bool('loss.gt_w_seg', True),
            use_normals=True,  # Stage 2 needs normals from Stage 1
            **self.conf.get_config('dataset.train')
        )
        
        self.test_dataset = GaussianRelightDataset(
            data_folder=self.conf.get_string('dataset.data_folder'),
            subject_name=self.subject,
            json_name=self.conf.get_string('dataset.json_name'),
            use_semantics=self.conf.get_bool('loss.gt_w_seg', True),
            use_normals=True,
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
        
        # Load trained Gaussian model from Stage 1
        print("Loading Gaussian model from Stage 1...")
        self.gaussian_model = self._load_stage1_model()
        
        # Freeze Gaussian geometry (optional, or fine-tune)
        self.freeze_geometry = self.conf.get_bool('train.freeze_geometry', True)
        if self.freeze_geometry:
            for param in [self.gaussian_model._xyz, 
                         self.gaussian_model._scaling,
                         self.gaussian_model._rotation]:
                param.requires_grad = False
            print("Geometry frozen, only training appearance")
        
        # Initialize renderer
        self.renderer = RelightableGaussianRenderer(
            img_height=self.img_res[0],
            img_width=self.img_res[1],
            sh_degree=2,
            bg_color=[1.0, 1.0, 1.0]
        ).cuda()
        
        # Initialize relighting networks (similar to ReliTalk)
        print("Initializing relighting networks...")
        self.albedo_net = ResNet.Albedo_ResnetGenerator().cuda()
        self.normal_net = ResNet.ResnetGenerator(input_nc=6).cuda()
        self.spec_net = UNet.UnetGenerator(output_nc=1, input_nc=6).cuda()
        
        ResNet.init_net(self.albedo_net)
        ResNet.init_net(self.normal_net)
        UNet.init_net(self.spec_net)
        
        # Learnable lighting (9 SH coefficients)
        self.light_var = torch.cuda.FloatTensor([
            0.0, 0.0, 0.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        self.light_var.requires_grad = True
        
        # Sample lights for augmentation (from ReliTalk)
        self.sample_lights = self._create_sample_lights()
        
        # Constant factor for SH computation
        self.constant_factor = self._get_sh_constant_factor().cuda()
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss().cuda()
        
        # Optimizer
        self.lr = self.conf.get_float('train.learning_rate', 1e-3)
        
        param_groups = [
            {'params': self.albedo_net.parameters(), 'lr': self.lr},
            {'params': self.normal_net.parameters(), 'lr': self.lr},
            {'params': self.spec_net.parameters(), 'lr': self.lr},
            {'params': [self.light_var], 'lr': self.lr * 0.1},
        ]
        
        # Add Gaussian appearance parameters if not frozen
        if not self.freeze_geometry:
            param_groups.append({
                'params': [self.gaussian_model._albedo,
                          self.gaussian_model._specular,
                          self.gaussian_model._roughness],
                'lr': self.lr * 0.5
            })
        
        self.optimizer = torch.optim.Adam(param_groups)
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[20, 40],
            gamma=0.5
        )
        
        # Training state
        self.start_epoch = 0
        
        # Load checkpoint if continuing
        if self.is_continue:
            self._load_checkpoint()
            
    def _load_stage1_model(self):
        """Load Gaussian model from Stage 1"""
        gaussian_model = RelightableGaussianModel(
            sh_degree=2,
            num_gaussians=self.conf.get_int('model.num_gaussians', 50000),
            with_motion_net=self.conf.get_bool('model.with_motion_net', True)
        ).cuda()
        
        # Find Stage 1 checkpoint
        if self.stage1_path:
            ckpt_path = self.stage1_path
        else:
            # Default path
            train_split = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))
            stage1_dir = os.path.join(self.exps_folder, self.subject, self.methodname, train_split, 'train')
            ckpt_path = os.path.join(stage1_dir, 'checkpoints', 'gaussian_latest.pth')
        
        if os.path.exists(ckpt_path):
            print(f"Loading Stage 1 checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            gaussian_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise FileNotFoundError(f"Stage 1 checkpoint not found at {ckpt_path}")
        
        return gaussian_model
    
    def _get_sh_constant_factor(self):
        """Get constant factors for SH computation (matching ReliTalk)"""
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
    
    def _create_sample_lights(self):
        """Create sample lighting conditions for augmentation"""
        # 16 different lighting conditions
        lights = []
        for i in range(16):
            theta = 2 * np.pi * i / 16
            light = torch.zeros(9)
            light[0] = 0.5  # Ambient
            light[1] = 0.5 * np.sin(theta)  # Y direction
            light[3] = 0.5 * np.cos(theta)  # X direction
            lights.append(light)
        return torch.stack(lights).cuda()
    
    def _load_checkpoint(self):
        """Load checkpoint for continuing training"""
        ckpt_path = os.path.join(self.checkpoints_path, f'relight_{self.checkpoint}.pth')
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            self.albedo_net.load_state_dict(checkpoint['albedo_net'])
            self.normal_net.load_state_dict(checkpoint['normal_net'])
            self.spec_net.load_state_dict(checkpoint['spec_net'])
            self.light_var = checkpoint['light_var']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
        else:
            print(f"Checkpoint {ckpt_path} not found, starting from scratch")
            
    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'albedo_net': self.albedo_net.state_dict(),
            'normal_net': self.normal_net.state_dict(),
            'spec_net': self.spec_net.state_dict(),
            'light_var': self.light_var,
            'optimizer': self.optimizer.state_dict(),
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.checkpoints_path, 'relight_latest.pth'))
        
        # Save periodic
        if epoch % 5 == 0:
            torch.save(checkpoint, os.path.join(self.checkpoints_path, f'relight_{epoch}.pth'))
            
        # Also save individual network weights (for compatibility with ReliTalk test scripts)
        torch.save(
            {'epoch': epoch, 'model_state_dict': self.albedo_net.state_dict()},
            os.path.join(self.checkpoints_path, 'LightModelParameters', f'{epoch}.pth')
        )
        torch.save(
            {'epoch': epoch, 'model_state_dict': self.normal_net.state_dict()},
            os.path.join(self.checkpoints_path, 'NormalModelParameters', f'{epoch}.pth')
        )
        torch.save(
            {'epoch': epoch, 'model_state_dict': self.spec_net.state_dict()},
            os.path.join(self.checkpoints_path, 'SpecModelParameters', f'{epoch}.pth')
        )
    
    def run(self):
        """Main training loop"""
        # Create subdirectories for checkpoints
        for subdir in ['LightModelParameters', 'NormalModelParameters', 'SpecModelParameters']:
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, subdir))
        
        print(f"\nStarting training from epoch {self.start_epoch}")
        
        # For specular computation
        sample_index = list(range(0, 256*256, 64))
        s = 8  # Specular exponent
        pi = torch.acos(torch.zeros(1)).item() * 2
        
        for epoch in range(self.start_epoch, self.nepochs):
            self.albedo_net.train()
            self.normal_net.train()
            self.spec_net.train()
            
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            
            for batch_idx, (indices, model_input, ground_truth) in enumerate(pbar):
                # Move to GPU
                for k, v in model_input.items():
                    if isinstance(v, torch.Tensor):
                        model_input[k] = v.cuda()
                        
                for k, v in ground_truth.items():
                    if isinstance(v, torch.Tensor):
                        ground_truth[k] = v.cuda()
                
                # Forward pass
                loss, loss_dict = self._train_step(
                    model_input, ground_truth, 
                    sample_index, s, pi, epoch
                )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'rgb': f'{loss_dict.get("rgb", 0):.4f}',
                })
            
            # End of epoch
            self.scheduler.step()
            avg_loss = epoch_loss / len(self.train_dataloader)
            
            # Print current light estimate
            light = torch.clamp(self.light_var, -2.0, 2.0)
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")
            print(f"  Estimated light: {light.detach().cpu().numpy()}")
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
            # Evaluation
            if epoch % 5 == 0:
                self._evaluate(epoch)
                
        print("Training completed!")
        
    def _train_step(self, model_input, ground_truth, sample_index, s, pi, epoch):
        """Single training step for relighting"""
        batch_size = ground_truth['rgb'].shape[0]
        H, W = self.img_res
        
        # Get masks
        semantics = ground_truth.get('semantics', None)
        if semantics is not None:
            face_mask = torch.sum(semantics[:, :, :-2], dim=2).view(-1, 1, H, W)
            shoulder_mask = semantics[:, :, -2].view(-1, 1, H, W)
        else:
            face_mask = model_input['object_mask'].view(-1, 1, H, W)
            shoulder_mask = torch.zeros_like(face_mask)
        
        face_mask = face_mask * model_input['object_mask'].view(-1, 1, H, W)
        
        # Prepare inputs
        rgb_input = ground_truth['rgb'].view(batch_size, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        rgb_input = (rgb_input + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Get normal from Stage 1 output or ground truth
        if 'normal' in ground_truth:
            normal = ground_truth['normal'].view(batch_size, H, W, 3).permute(0, 3, 1, 2)
        else:
            # Render normal from Gaussians
            with torch.no_grad():
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
        
        # Current light estimate
        light = torch.clamp(self.light_var, -2.0, 2.0)
        light_coeffs = light.clone()
        light_coeffs[5:] = light_coeffs[5:] / 10  # Attenuate higher-order terms
        
        # Predict albedo
        albedo = self.albedo_net(rgb_input * 2 - 1)  # Input in [-1, 1]
        
        # Refine normal
        fine_normal = (self.normal_net(torch.cat([rgb_input * 2 - 1, normal], dim=1)) + 1) / 2
        fine_normal = fine_normal * face_mask * 2 - 1
        
        # Compute shading using SH
        shading = add_SHlight(self.constant_factor, fine_normal, 
                             light_coeffs.view(1, -1, 1).repeat(batch_size, 1, 1))
        
        # Normalize shading
        masked_shading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading) + 1e-7) * face_mask
        masked_shading_nonzero = masked_shading[masked_shading.nonzero(as_tuple=True)]
        if masked_shading_nonzero.numel() > 0:
            shading_min = torch.min(masked_shading_nonzero)
            shading_max = torch.max(masked_shading_nonzero)
            masked_shading = torch.clamp((masked_shading - shading_min) / (shading_max - shading_min + 1e-7), 0, 1)
        
        # Compute specular (simplified)
        space_normal, space_shading = normal_shading(light_coeffs)
        length = space_normal.shape[0]
        space_shading_sampled = space_shading.reshape(length, -1) / 255
        space_shading_sampled = space_shading_sampled[sample_index, ...]
        space_normal_sampled = space_normal[sample_index, ...]
        
        h = torch.cuda.FloatTensor(
            normalize(space_normal_sampled + np.array([[0, 0, 1]]).repeat(len(sample_index), 0))
        ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        nh = (fine_normal[:, [0], :, :] * h[:, :, 0, :, :] + 
              fine_normal[:, [1], :, :] * h[:, :, 1, :, :] + 
              fine_normal[:, [2], :, :] * h[:, :, 2, :, :])
        
        z = torch.cuda.FloatTensor(space_shading_sampled[np.newaxis, :, np.newaxis]) * nh
        sep_spec = (s + 2) / (2 * pi) * torch.pow(torch.clamp(z, 0, 1), s)
        spec = torch.sum(sep_spec, dim=1, keepdim=True)
        
        masked_spec = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec) + 1e-7) * face_mask
        masked_spec = (masked_spec - torch.min(masked_spec)) / (torch.max(masked_spec) - torch.min(masked_spec) + 1e-7)
        
        # Predict specular map
        specmap = self.spec_net(torch.cat([rgb_input * 2 - 1, fine_normal.detach()], dim=1))
        scaled_specmap = (specmap + 1) / 2 * 0.2
        
        # Final rendered image
        final_image = albedo * (masked_shading + masked_spec * scaled_specmap)
        final_image = torch.clamp(final_image, 0, 1)
        
        # Handle shoulder region (copy from input)
        final_image = final_image * (1 - shoulder_mask) + rgb_input * shoulder_mask
        
        # Losses
        loss_dict = {}
        
        # RGB reconstruction loss
        rgb_loss = self.l1_loss(final_image * face_mask, rgb_input * face_mask)
        loss_dict['rgb'] = rgb_loss.item()
        
        # VGG perceptual loss
        vgg_loss = self.vgg_loss(final_image, rgb_input)
        loss_dict['vgg'] = vgg_loss.item()
        
        # Albedo smoothness loss
        albedo_smooth = self._smoothness_loss(albedo * face_mask)
        loss_dict['albedo_smooth'] = albedo_smooth.item()
        
        # Normal consistency loss
        if 'normal' in ground_truth:
            normal_loss = self.l1_loss(fine_normal * face_mask, normal * face_mask)
            loss_dict['normal'] = normal_loss.item()
        else:
            normal_loss = 0.0
        
        # Lighting regularization (encourage smooth lighting)
        if epoch < self.nepochs // 3:
            # Early epochs: stronger regularization to avoid overfitting
            light_reg = 0.1 * torch.sum(light_coeffs[4:] ** 2)  # Penalize high-order terms
        else:
            light_reg = 0.01 * torch.sum(light_coeffs[4:] ** 2)
        loss_dict['light_reg'] = light_reg.item()
        
        # Total loss
        total_loss = (
            1.0 * rgb_loss +
            0.1 * vgg_loss +
            0.01 * albedo_smooth +
            0.1 * normal_loss +
            light_reg
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _smoothness_loss(self, img):
        """Compute smoothness loss"""
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()
    
    def _evaluate(self, epoch):
        """Evaluate and save results"""
        self.albedo_net.eval()
        self.normal_net.eval()
        self.spec_net.eval()
        
        eval_epoch_dir = os.path.join(self.eval_dir, f'epoch_{epoch}')
        utils.mkdir_ifnotexists(eval_epoch_dir)
        
        print(f"\nEvaluating epoch {epoch}...")
        
        sample_index = list(range(0, 256*256, 64))
        s = 8
        pi = torch.acos(torch.zeros(1)).item() * 2
        
        with torch.no_grad():
            for batch_idx, (indices, model_input, ground_truth) in enumerate(tqdm(self.test_dataloader)):
                if batch_idx >= 10:  # Only save first 10 samples
                    break
                    
                # Move to GPU
                for k, v in model_input.items():
                    if isinstance(v, torch.Tensor):
                        model_input[k] = v.cuda()
                        
                for k, v in ground_truth.items():
                    if isinstance(v, torch.Tensor):
                        ground_truth[k] = v.cuda()
                
                H, W = self.img_res
                batch_size = ground_truth['rgb'].shape[0]
                
                # Prepare inputs
                rgb_input = ground_truth['rgb'].view(batch_size, H, W, 3).permute(0, 3, 1, 2)
                rgb_input = (rgb_input + 1) / 2
                
                if 'normal' in ground_truth:
                    normal = ground_truth['normal'].view(batch_size, H, W, 3).permute(0, 3, 1, 2)
                else:
                    continue
                
                face_mask = model_input['object_mask'].view(-1, 1, H, W)
                
                # Forward
                albedo = self.albedo_net(rgb_input * 2 - 1)
                fine_normal = (self.normal_net(torch.cat([rgb_input * 2 - 1, normal], dim=1)) + 1) / 2 * face_mask * 2 - 1
                specmap = self.spec_net(torch.cat([rgb_input * 2 - 1, fine_normal], dim=1))
                
                # Render with different lightings
                results = []
                for light_idx, sample_light in enumerate(self.sample_lights[:4]):
                    shading = add_SHlight(self.constant_factor, fine_normal,
                                         sample_light.view(1, -1, 1))
                    shading = (shading - shading.min()) / (shading.max() - shading.min() + 1e-7) * face_mask
                    
                    relit_image = torch.clamp(albedo * shading, 0, 1)
                    results.append(relit_image)
                
                # Save results
                frame_id = model_input['img_name'][0, 0].item()
                
                # Save albedo
                albedo_img = albedo[0].permute(1, 2, 0).cpu().numpy()
                albedo_img = (albedo_img * 255).clip(0, 255).astype(np.uint8)
                
                from PIL import Image
                Image.fromarray(albedo_img).save(
                    os.path.join(eval_epoch_dir, f'{frame_id:05d}_albedo.png')
                )
                
                # Save relit images
                for i, relit in enumerate(results):
                    relit_img = relit[0].permute(1, 2, 0).cpu().numpy()
                    relit_img = (relit_img * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(relit_img).save(
                        os.path.join(eval_epoch_dir, f'{frame_id:05d}_relit_{i}.png')
                    )
        
        print(f"Saved evaluation results to {eval_epoch_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: Gaussian Relighting Training')
    parser.add_argument('--conf', type=str, required=True, help='Config file path')
    parser.add_argument('--nepoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--is_continue', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='latest', help='Checkpoint to load')
    parser.add_argument('--stage1_path', type=str, default='', help='Path to Stage 1 checkpoint')
    
    args = parser.parse_args()
    
    runner = Stage2TrainRunner(
        conf=args.conf,
        nepochs=args.nepoch,
        is_continue=args.is_continue,
        checkpoint=args.checkpoint,
        stage1_path=args.stage1_path
    )
    
    runner.run()
