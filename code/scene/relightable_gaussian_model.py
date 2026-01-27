"""
Relightable Gaussian Model for Talking Head
This module combines 3D Gaussian Splatting with relightable rendering.
Instead of storing view-dependent SH colors, we store albedo and specularity
to support relighting with arbitrary SH lighting.
"""

import torch
import torch.nn as nn
import numpy as np
from simple_knn._C import distCUDA2
from torch import Tensor
from typing import Dict, Optional, Tuple

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def build_rotation(r):
    """Build rotation matrix from quaternion"""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L

class RelightableGaussianModel(nn.Module):
    """
    A Gaussian Splatting model with relighting support.
    
    Key differences from standard GaussianModel:
    - Instead of SH coefficients for color, we store albedo (RGB) and specular coefficients
    - Normals can be computed from local covariance or stored explicitly
    - Supports SH lighting for relighting
    """
    
    def __init__(self, 
                 sh_degree: int = 3,
                 num_gaussians: int = 50000,
                 with_motion_net: bool = True):
        super().__init__()
        
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0  # Will be increased during training
        
        # Gaussian core parameters
        self._xyz = nn.Parameter(torch.zeros(0, 3))  # 3D positions
        self._rotation = nn.Parameter(torch.zeros(0, 4))  # Quaternion rotation
        self._scaling = nn.Parameter(torch.zeros(0, 3))  # Anisotropic scaling
        self._opacity = nn.Parameter(torch.zeros(0, 1))  # Opacity
        
        # Relighting parameters (instead of standard SH colors)
        self._albedo = nn.Parameter(torch.zeros(0, 3))  # Diffuse albedo (RGB)
        self._specular = nn.Parameter(torch.zeros(0, 1))  # Specular intensity
        self._roughness = nn.Parameter(torch.zeros(0, 1))  # Surface roughness
        
        # Optional: explicitly stored normals for smoother lighting
        self._normals = nn.Parameter(torch.zeros(0, 3))
        self.use_explicit_normals = False
        
        # For compatibility with standard 3DGS pipeline, we also have SH features
        # But these will be computed from albedo + lighting
        self._features_dc = None
        self._features_rest = None
        
        # Gradient accumulators for adaptive density control
        self.xyz_gradient_accum = None
        self.denom = None
        self.max_radii2D = None
        
        # Motion/deformation network for talking head
        self.with_motion_net = with_motion_net
        if with_motion_net:
            self._init_deformation_network()
        
        # Optimizer learning rates
        self.setup_lr_dict()
        
    def _init_deformation_network(self):
        """Initialize the deformation network for expression-driven animation"""
        from .deformation_network import DeformationNetwork
        self.deformation_net = DeformationNetwork(
            d_feature=256,
            d_in=3,  # xyz input
            d_out_xyz=3,  # xyz offset
            d_out_rot=4,  # rotation quaternion offset
            d_out_scale=3,  # scale offset
            d_audio=64,  # Audio feature dimension
            d_expr=50,  # Expression feature dimension (FLAME)
        )
        
    def setup_lr_dict(self):
        """Setup learning rate dictionary for different parameters"""
        self.lr_dict = {
            "xyz": 0.00016,
            "rotation": 0.001,
            "scaling": 0.005,
            "opacity": 0.05,
            "albedo": 0.0025,
            "specular": 0.001,
            "roughness": 0.001,
            "normals": 0.001,
        }
        
    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]
    
    @property
    def get_xyz(self) -> Tensor:
        return self._xyz
    
    @property
    def get_rotation(self) -> Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)
    
    @property
    def get_scaling(self) -> Tensor:
        return torch.exp(self._scaling)
    
    @property
    def get_opacity(self) -> Tensor:
        return torch.sigmoid(self._opacity)
    
    @property
    def get_albedo(self) -> Tensor:
        """Get albedo in [0, 1] range"""
        return torch.sigmoid(self._albedo)
    
    @property
    def get_specular(self) -> Tensor:
        """Get specular intensity in [0, 1] range"""
        return torch.sigmoid(self._specular)
    
    @property
    def get_roughness(self) -> Tensor:
        """Get roughness in [0, 1] range"""
        return torch.sigmoid(self._roughness)
    
    @property
    def get_normals(self) -> Tensor:
        """Get normalized normal vectors"""
        if self.use_explicit_normals:
            return torch.nn.functional.normalize(self._normals, dim=-1)
        else:
            # Compute normals from covariance
            return self._compute_normals_from_covariance()
    
    def _compute_normals_from_covariance(self) -> Tensor:
        """
        Compute normals from the covariance matrix of each Gaussian.
        The normal is the eigenvector corresponding to the smallest eigenvalue.
        """
        scales = self.get_scaling
        rotations = self.get_rotation
        
        # Build rotation matrices
        R = build_rotation(rotations)
        
        # The normal is the direction of minimum scale
        min_scale_idx = torch.argmin(scales, dim=-1)
        
        # Get the corresponding column of R
        normals = torch.zeros_like(self._xyz)
        for i in range(3):
            mask = (min_scale_idx == i)
            normals[mask] = R[mask, :, i]
        
        return normals
    
    def get_covariance(self, scaling_modifier: float = 1.0) -> Tensor:
        """Compute 3D covariance matrices for all Gaussians"""
        scales = self.get_scaling * scaling_modifier
        rotations = self.get_rotation
        L = build_scaling_rotation(scales, rotations)
        return L @ L.transpose(1, 2)
    
    def initialize_from_point_cloud(self, 
                                    points: Tensor,
                                    colors: Optional[Tensor] = None,
                                    normals: Optional[Tensor] = None):
        """
        Initialize Gaussians from a point cloud.
        
        Args:
            points: (N, 3) tensor of 3D positions
            colors: (N, 3) tensor of RGB colors [0, 1] - used as initial albedo
            normals: (N, 3) tensor of normal vectors
        """
        device = points.device
        num_points = points.shape[0]
        
        print(f"Initializing {num_points} Gaussians from point cloud")
        
        # Initialize positions
        self._xyz = nn.Parameter(points.float())
        
        # Initialize scales based on nearest neighbor distances
        dist2 = torch.clamp_min(distCUDA2(points.float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.float())
        
        # Initialize rotations as identity quaternions
        rots = torch.zeros((num_points, 4), device=device)
        rots[:, 0] = 1.0  # w=1, x=y=z=0
        self._rotation = nn.Parameter(rots.float())
        
        # Initialize opacity
        opacities = inverse_sigmoid(0.1 * torch.ones((num_points, 1), device=device))
        self._opacity = nn.Parameter(opacities.float())
        
        # Initialize albedo from colors or default to gray
        if colors is not None:
            albedo = inverse_sigmoid(torch.clamp(colors, 0.001, 0.999))
        else:
            albedo = inverse_sigmoid(0.5 * torch.ones((num_points, 3), device=device))
        self._albedo = nn.Parameter(albedo.float())
        
        # Initialize specular (low default value)
        specular = inverse_sigmoid(0.1 * torch.ones((num_points, 1), device=device))
        self._specular = nn.Parameter(specular.float())
        
        # Initialize roughness (medium default value)
        roughness = inverse_sigmoid(0.5 * torch.ones((num_points, 1), device=device))
        self._roughness = nn.Parameter(roughness.float())
        
        # Initialize normals
        if normals is not None:
            self._normals = nn.Parameter(normals.float())
            self.use_explicit_normals = True
        else:
            self._normals = nn.Parameter(torch.zeros((num_points, 3), device=device))
            self.use_explicit_normals = False
        
        # Initialize gradient accumulators
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device=device)
        self.denom = torch.zeros((num_points, 1), device=device)
        self.max_radii2D = torch.zeros((num_points), device=device)
        
    def forward(self, 
                viewpoint_camera,
                audio_features: Optional[Tensor] = None,
                expression_params: Optional[Tensor] = None,
                sh_coeffs: Optional[Tensor] = None,
                scaling_modifier: float = 1.0,
                override_color: Optional[Tensor] = None) -> Dict:
        """
        Forward pass for relightable Gaussian rendering.
        
        Args:
            viewpoint_camera: Camera parameters
            audio_features: Audio features for talking head animation
            expression_params: FLAME expression parameters
            sh_coeffs: Spherical harmonics coefficients for lighting (9 or 27 values)
            scaling_modifier: Scale modifier for Gaussians
            override_color: Override colors (for debugging)
            
        Returns:
            Dictionary containing rendered images and intermediate results
        """
        # Get base Gaussian properties
        xyz = self.get_xyz
        scales = self.get_scaling * scaling_modifier
        rotations = self.get_rotation
        opacity = self.get_opacity
        albedo = self.get_albedo
        specular = self.get_specular
        normals = self.get_normals
        
        # Apply deformation if motion network is available
        if self.with_motion_net and (audio_features is not None or expression_params is not None):
            xyz_offset, rot_offset, scale_offset = self.deformation_net(
                xyz, audio_features, expression_params
            )
            xyz = xyz + xyz_offset
            # Apply rotation offset (quaternion multiplication)
            rotations = self._quaternion_multiply(rotations, rot_offset)
            scales = scales * (1 + scale_offset)
            
            # Update normals based on deformation
            normals = self._deform_normals(normals, rot_offset)
        
        # Compute shaded colors from lighting
        if sh_coeffs is not None:
            shaded_colors = self._compute_sh_shading(
                albedo, normals, specular, sh_coeffs
            )
        else:
            # Default: use albedo directly (no lighting)
            shaded_colors = albedo
        
        if override_color is not None:
            shaded_colors = override_color
        
        return {
            "xyz": xyz,
            "scales": scales,
            "rotations": rotations,
            "opacity": opacity,
            "colors": shaded_colors,
            "albedo": albedo,
            "normals": normals,
            "specular": specular,
        }
    
    def _compute_sh_shading(self, 
                            albedo: Tensor,
                            normals: Tensor,
                            specular: Tensor,
                            sh_coeffs: Tensor) -> Tensor:
        """
        Compute shaded colors using Spherical Harmonics lighting.
        
        This implements the relighting equation from ReliTalk:
        Color = Albedo * (Shading + Specular * SpecularMap)
        
        Args:
            albedo: (N, 3) diffuse albedo
            normals: (N, 3) normal vectors
            specular: (N, 1) specular intensity
            sh_coeffs: (9,) or (27,) SH coefficients for lighting
            
        Returns:
            (N, 3) shaded colors
        """
        # Normalize normals
        normals = torch.nn.functional.normalize(normals, dim=-1)
        
        # Evaluate SH basis functions at normal directions
        sh_basis = self._eval_sh_basis(normals)  # (N, 9)
        
        if sh_coeffs.dim() == 1:
            if sh_coeffs.shape[0] == 9:
                # Grayscale lighting
                shading = (sh_basis * sh_coeffs.unsqueeze(0)).sum(dim=-1, keepdim=True)
                shading = shading.expand(-1, 3)
            elif sh_coeffs.shape[0] == 27:
                # RGB lighting
                sh_coeffs = sh_coeffs.view(3, 9)
                shading = torch.stack([
                    (sh_basis * sh_coeffs[c].unsqueeze(0)).sum(dim=-1)
                    for c in range(3)
                ], dim=-1)
            else:
                raise ValueError(f"Unexpected sh_coeffs shape: {sh_coeffs.shape}")
        else:
            # Batch of SH coefficients
            shading = torch.einsum('ni,bi->bn', sh_basis, sh_coeffs.view(-1, 9))
            shading = shading.unsqueeze(-1).expand(-1, -1, 3)
        
        # Ensure shading is positive
        shading = torch.clamp(shading, min=0.0)
        
        # Simple specular approximation (Phong-like)
        specular_shading = specular * shading.mean(dim=-1, keepdim=True)
        
        # Final color = albedo * (diffuse_shading + specular)
        colors = albedo * (shading + specular_shading)
        
        # Clamp to valid range
        colors = torch.clamp(colors, 0.0, 1.0)
        
        return colors
    
    def _eval_sh_basis(self, directions: Tensor) -> Tensor:
        """
        Evaluate first 9 (order 2) spherical harmonics basis functions.
        
        Uses the same SH order as ReliTalk's light_util.py:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-Y^2
        
        Args:
            directions: (N, 3) normalized direction vectors
            
        Returns:
            (N, 9) SH basis values
        """
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        
        # Normalization constants (matching ReliTalk's light_util.py)
        pi = 3.14159265358979323846
        C0 = 0.5 / (pi ** 0.5)  # 0.28209479
        C1 = (3 ** 0.5) / 2 / (pi ** 0.5)  # 0.48860251
        C2_xy = (15 ** 0.5) / 2 / (pi ** 0.5)  # 1.09254843
        C2_z2 = (5 ** 0.5) / 4 / (pi ** 0.5)  # 0.31539157
        C2_x2y2 = (15 ** 0.5) / 4 / (pi ** 0.5)  # 0.54627421
        
        # Order from ReliTalk: 1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-Y^2
        Y00 = C0 * torch.ones_like(x)
        Y1_Y = C1 * y
        Y1_Z = C1 * z
        Y1_X = C1 * x
        Y2_YX = C2_xy * y * x
        Y2_YZ = C2_xy * y * z
        Y2_3Z2 = C2_z2 * (3 * z * z - 1)
        Y2_XZ = C2_xy * x * z
        Y2_X2Y2 = C2_x2y2 * (x * x - y * y)
        
        return torch.stack([Y00, Y1_Y, Y1_Z, Y1_X, Y2_YX, Y2_YZ, Y2_3Z2, Y2_XZ, Y2_X2Y2], dim=-1)
    
    def _quaternion_multiply(self, q1: Tensor, q2: Tensor) -> Tensor:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def _deform_normals(self, normals: Tensor, rot_offset: Tensor) -> Tensor:
        """Apply rotation to normals"""
        R = build_rotation(rot_offset)
        return torch.einsum('nij,nj->ni', R, normals)
    
    def densify_and_split(self, grads: Tensor, grad_threshold: float, 
                          scene_extent: float, N: int = 2):
        """Split Gaussians with large gradients"""
        n_init_points = self.num_gaussians
        
        # Extract points that need splitting
        padded_grad = torch.zeros((n_init_points,), device=self._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.max(dim=1).values > scene_extent * 0.01
        )
        
        # ... (implementation continues)
        
    def densify_and_clone(self, grads: Tensor, grad_threshold: float, 
                          scene_extent: float):
        """Clone Gaussians with large gradients but small size"""
        # Extract points that need cloning
        selected_pts_mask = torch.where(grads.squeeze() >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.max(dim=1).values <= scene_extent * 0.01
        )
        
        # Clone the selected Gaussians
        new_xyz = self._xyz[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_specular = self._specular[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_normals = self._normals[selected_pts_mask]
        
        self._densification_postfix(
            new_xyz, new_rotation, new_scaling, new_opacity,
            new_albedo, new_specular, new_roughness, new_normals
        )
        
    def _densification_postfix(self, new_xyz, new_rotation, new_scaling, 
                               new_opacity, new_albedo, new_specular, 
                               new_roughness, new_normals):
        """Add new Gaussians after densification"""
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        self._albedo = nn.Parameter(torch.cat([self._albedo, new_albedo], dim=0))
        self._specular = nn.Parameter(torch.cat([self._specular, new_specular], dim=0))
        self._roughness = nn.Parameter(torch.cat([self._roughness, new_roughness], dim=0))
        self._normals = nn.Parameter(torch.cat([self._normals, new_normals], dim=0))
        
        # Update gradient accumulators
        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=device)
        self.denom = torch.zeros((self.num_gaussians, 1), device=device)
        self.max_radii2D = torch.zeros((self.num_gaussians), device=device)
        
    def prune_points(self, mask: Tensor):
        """Remove Gaussians based on mask"""
        valid_points_mask = ~mask
        self._xyz = nn.Parameter(self._xyz[valid_points_mask])
        self._rotation = nn.Parameter(self._rotation[valid_points_mask])
        self._scaling = nn.Parameter(self._scaling[valid_points_mask])
        self._opacity = nn.Parameter(self._opacity[valid_points_mask])
        self._albedo = nn.Parameter(self._albedo[valid_points_mask])
        self._specular = nn.Parameter(self._specular[valid_points_mask])
        self._roughness = nn.Parameter(self._roughness[valid_points_mask])
        self._normals = nn.Parameter(self._normals[valid_points_mask])
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    def save_ply(self, path: str):
        """Save Gaussians to PLY file"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normals.detach().cpu().numpy()
        albedo = self.get_albedo.detach().cpu().numpy()
        opacities = self.get_opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()
        specular = self._specular.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        
        # Write PLY header and data
        with open(path, 'wb') as f:
            # Header
            header = f"""ply
format binary_little_endian 1.0
element vertex {xyz.shape[0]}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float red
property float green
property float blue
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float specular
property float roughness
end_header
"""
            f.write(header.encode())
            
            # Data
            for i in range(xyz.shape[0]):
                data = np.array([
                    xyz[i, 0], xyz[i, 1], xyz[i, 2],
                    normals[i, 0], normals[i, 1], normals[i, 2],
                    albedo[i, 0], albedo[i, 1], albedo[i, 2],
                    opacities[i, 0],
                    scales[i, 0], scales[i, 1], scales[i, 2],
                    rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3],
                    specular[i, 0], roughness[i, 0]
                ], dtype=np.float32)
                f.write(data.tobytes())
                
    def load_ply(self, path: str):
        """Load Gaussians from PLY file"""
        from plyfile import PlyData
        
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        normals = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=1)
        albedo = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1)
        opacities = vertex['opacity'][:, np.newaxis]
        scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
        rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
        specular = vertex['specular'][:, np.newaxis]
        roughness = vertex['roughness'][:, np.newaxis]
        
        device = self._xyz.device if self._xyz.numel() > 0 else 'cuda'
        
        self._xyz = nn.Parameter(torch.from_numpy(xyz).float().to(device))
        self._normals = nn.Parameter(torch.from_numpy(normals).float().to(device))
        self._albedo = nn.Parameter(inverse_sigmoid(torch.from_numpy(albedo).float().clamp(0.001, 0.999)).to(device))
        self._opacity = nn.Parameter(inverse_sigmoid(torch.from_numpy(opacities).float().clamp(0.001, 0.999)).to(device))
        self._scaling = nn.Parameter(torch.from_numpy(scales).float().to(device))
        self._rotation = nn.Parameter(torch.from_numpy(rotations).float().to(device))
        self._specular = nn.Parameter(inverse_sigmoid(torch.from_numpy(specular).float().clamp(0.001, 0.999)).to(device))
        self._roughness = nn.Parameter(inverse_sigmoid(torch.from_numpy(roughness).float().clamp(0.001, 0.999)).to(device))
        
        self.use_explicit_normals = True
        
        # Initialize gradient accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=device)
        self.denom = torch.zeros((self.num_gaussians, 1), device=device)
        self.max_radii2D = torch.zeros((self.num_gaussians), device=device)

    def get_training_params(self):
        """Get all trainable parameters organized by groups"""
        params = [
            {'params': [self._xyz], 'lr': self.lr_dict['xyz'], 'name': 'xyz'},
            {'params': [self._rotation], 'lr': self.lr_dict['rotation'], 'name': 'rotation'},
            {'params': [self._scaling], 'lr': self.lr_dict['scaling'], 'name': 'scaling'},
            {'params': [self._opacity], 'lr': self.lr_dict['opacity'], 'name': 'opacity'},
            {'params': [self._albedo], 'lr': self.lr_dict['albedo'], 'name': 'albedo'},
            {'params': [self._specular], 'lr': self.lr_dict['specular'], 'name': 'specular'},
            {'params': [self._roughness], 'lr': self.lr_dict['roughness'], 'name': 'roughness'},
        ]
        
        if self.use_explicit_normals:
            params.append({'params': [self._normals], 'lr': self.lr_dict['normals'], 'name': 'normals'})
        
        if self.with_motion_net:
            params.append({'params': self.deformation_net.parameters(), 'lr': 1e-4, 'name': 'deformation'})
        
        return params
