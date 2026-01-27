"""
Relightable Gaussian Renderer

This renderer extends the standard 3D Gaussian Splatting renderer to support:
1. Rendering of albedo maps
2. Rendering of normal maps  
3. Relighting with Spherical Harmonics (SH) lighting
4. Depth rendering

The core idea is that instead of baking lighting into the Gaussian colors,
we render intermediate maps (albedo, normals) and apply lighting in image space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, NamedTuple
import math

# Import diff-gaussian-rasterization
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer
    )
    GAUSSIAN_RASTERIZER_AVAILABLE = True
except ImportError:
    print("Warning: diff_gaussian_rasterization not found. Using fallback renderer.")
    GAUSSIAN_RASTERIZER_AVAILABLE = False


class RenderOutput(NamedTuple):
    """Output from the Gaussian renderer"""
    rendered_image: Tensor  # (3, H, W) final rendered image
    albedo_map: Tensor      # (3, H, W) albedo map
    normal_map: Tensor      # (3, H, W) normal map (in camera space)
    depth_map: Tensor       # (1, H, W) depth map
    opacity_map: Tensor     # (1, H, W) accumulated opacity
    shading_map: Optional[Tensor]  # (3, H, W) shading from SH lighting
    specular_map: Optional[Tensor] # (3, H, W) specular component
    radii: Tensor           # (N,) 2D radii of Gaussians


class Camera:
    """Camera parameters for rendering"""
    
    def __init__(self,
                 image_width: int,
                 image_height: int,
                 fov_x: float,
                 fov_y: float,
                 world_view_transform: Tensor,
                 projection_matrix: Tensor,
                 camera_center: Tensor,
                 znear: float = 0.01,
                 zfar: float = 100.0):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = fov_x
        self.FoVy = fov_y
        self.world_view_transform = world_view_transform
        self.projection_matrix = projection_matrix
        self.full_proj_transform = world_view_transform @ projection_matrix
        self.camera_center = camera_center
        self.znear = znear
        self.zfar = zfar


def create_camera_from_params(
    R: Tensor,  # (3, 3) rotation matrix
    T: Tensor,  # (3,) translation vector
    fov_x: float,
    fov_y: float,
    image_width: int,
    image_height: int,
    znear: float = 0.01,
    zfar: float = 100.0
) -> Camera:
    """Create a Camera from rotation and translation"""
    device = R.device
    
    # World to view transform
    world_view = torch.eye(4, device=device)
    world_view[:3, :3] = R.T
    world_view[:3, 3] = -R.T @ T
    
    # Projection matrix
    tan_half_fov_x = math.tan(fov_x / 2)
    tan_half_fov_y = math.tan(fov_y / 2)
    
    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 1 / tan_half_fov_x
    proj[1, 1] = 1 / tan_half_fov_y
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1
    
    camera_center = -R.T @ T
    
    return Camera(
        image_width=image_width,
        image_height=image_height,
        fov_x=fov_x,
        fov_y=fov_y,
        world_view_transform=world_view.T,  # Column-major
        projection_matrix=proj.T,
        camera_center=camera_center,
        znear=znear,
        zfar=zfar
    )


class RelightableGaussianRenderer(nn.Module):
    """
    Gaussian renderer with relighting support.
    
    Rendering pipeline:
    1. Rasterize Gaussians to get albedo, normal, depth maps
    2. Apply SH lighting to get shading map
    3. Combine: rendered = albedo * (shading + specular)
    """
    
    def __init__(self, 
                 sh_degree: int = 2,
                 bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        
        self.sh_degree = sh_degree
        self.register_buffer('bg_color', torch.tensor(bg_color))
        
        # Default lighting (roughly frontal white light)
        default_sh = torch.zeros(9)
        default_sh[0] = 0.5  # Ambient
        default_sh[2] = 0.3  # Z direction (frontal)
        self.register_buffer('default_sh_coeffs', default_sh)
        
    def forward(self,
                gaussians,
                camera: Camera,
                audio_features: Optional[Tensor] = None,
                expression_params: Optional[Tensor] = None,
                sh_coeffs: Optional[Tensor] = None,
                scaling_modifier: float = 1.0,
                compute_cov3D_python: bool = False,
                convert_SHs_python: bool = False) -> RenderOutput:
        """
        Render Gaussians with relighting.
        
        Args:
            gaussians: RelightableGaussianModel
            camera: Camera parameters
            audio_features: Audio features for animation
            expression_params: FLAME expression parameters
            sh_coeffs: (9,) or (27,) SH lighting coefficients
            scaling_modifier: Scale modifier for Gaussians
            compute_cov3D_python: Compute covariance in Python (slower but differentiable)
            convert_SHs_python: Convert SH in Python
            
        Returns:
            RenderOutput containing all rendered maps
        """
        # Get Gaussian properties (possibly deformed)
        gaussian_out = gaussians(
            viewpoint_camera=camera,
            audio_features=audio_features,
            expression_params=expression_params,
            sh_coeffs=None,  # We'll apply lighting in image space
            scaling_modifier=scaling_modifier
        )
        
        xyz = gaussian_out['xyz']
        scales = gaussian_out['scales']
        rotations = gaussian_out['rotations']
        opacity = gaussian_out['opacity']
        albedo = gaussian_out['albedo']
        normals = gaussian_out['normals']
        specular = gaussian_out['specular']
        
        # Use provided SH coeffs or default
        if sh_coeffs is None:
            sh_coeffs = self.default_sh_coeffs
            
        if GAUSSIAN_RASTERIZER_AVAILABLE:
            return self._render_with_rasterizer(
                xyz, scales, rotations, opacity, albedo, normals, specular,
                camera, sh_coeffs, scaling_modifier
            )
        else:
            return self._render_fallback(
                xyz, scales, rotations, opacity, albedo, normals, specular,
                camera, sh_coeffs
            )
    
    def _render_with_rasterizer(self,
                                xyz: Tensor,
                                scales: Tensor,
                                rotations: Tensor,
                                opacity: Tensor,
                                albedo: Tensor,
                                normals: Tensor,
                                specular: Tensor,
                                camera: Camera,
                                sh_coeffs: Tensor,
                                scaling_modifier: float) -> RenderOutput:
        """Render using diff-gaussian-rasterization"""
        device = xyz.device
        
        # Setup rasterization settings
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color.to(device),
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform.to(device),
            projmatrix=camera.full_proj_transform.to(device),
            sh_degree=0,  # We don't use built-in SH
            campos=camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # We need to render multiple passes for albedo, normals, depth
        # Pass 1: Render albedo
        screenspace_points = torch.zeros_like(xyz, requires_grad=True, device=device) + 0
        
        # Create SH features from albedo (just DC component)
        shs_albedo = albedo.unsqueeze(1)  # (N, 1, 3)
        
        albedo_map, radii = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=shs_albedo,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        
        # Pass 2: Render normals
        # Transform normals to camera space
        R_world_to_cam = camera.world_view_transform[:3, :3].T
        normals_cam = (R_world_to_cam @ normals.T).T
        normals_cam = F.normalize(normals_cam, dim=-1)
        
        # Encode normals as colors (map [-1, 1] to [0, 1])
        normals_color = (normals_cam + 1) / 2
        shs_normals = normals_color.unsqueeze(1)
        
        normal_map, _ = rasterizer(
            means3D=xyz,
            means2D=screenspace_points.detach(),
            shs=shs_normals,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        
        # Decode normals back to [-1, 1]
        normal_map = normal_map * 2 - 1
        normal_map = F.normalize(normal_map, dim=0)
        
        # Pass 3: Render depth (using z-coordinate in camera space)
        xyz_cam = (camera.world_view_transform[:3, :3].T @ xyz.T + 
                   camera.world_view_transform[:3, 3:4]).T
        depths = xyz_cam[:, 2:3]  # Z in camera space
        
        # Normalize depth for rendering
        depth_min, depth_max = depths.min(), depths.max()
        depths_norm = (depths - depth_min) / (depth_max - depth_min + 1e-8)
        depths_color = depths_norm.expand(-1, 3)
        shs_depth = depths_color.unsqueeze(1)
        
        depth_map, _ = rasterizer(
            means3D=xyz,
            means2D=screenspace_points.detach(),
            shs=shs_depth,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        
        # Denormalize depth
        depth_map = depth_map[0:1] * (depth_max - depth_min) + depth_min
        
        # Pass 4: Render specular map
        specular_color = specular.expand(-1, 3)
        shs_spec = specular_color.unsqueeze(1)
        
        specular_map, _ = rasterizer(
            means3D=xyz,
            means2D=screenspace_points.detach(),
            shs=shs_spec,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        
        # Render opacity map
        ones_color = torch.ones_like(albedo)
        shs_ones = ones_color.unsqueeze(1)
        
        opacity_map, _ = rasterizer(
            means3D=xyz,
            means2D=screenspace_points.detach(),
            shs=shs_ones,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        opacity_map = opacity_map[0:1]
        
        # Apply SH lighting to get shading
        shading_map = self._compute_sh_shading_image(normal_map, sh_coeffs)
        
        # Final compositing
        # rendered = albedo * (shading + specular * specular_map)
        rendered_image = albedo_map * (shading_map + specular_map * 0.5)
        rendered_image = torch.clamp(rendered_image, 0, 1)
        
        return RenderOutput(
            rendered_image=rendered_image,
            albedo_map=albedo_map,
            normal_map=normal_map,
            depth_map=depth_map,
            opacity_map=opacity_map,
            shading_map=shading_map,
            specular_map=specular_map,
            radii=radii
        )
    
    def _compute_sh_shading_image(self, 
                                   normal_map: Tensor, 
                                   sh_coeffs: Tensor) -> Tensor:
        """
        Compute SH shading for each pixel based on normal map.
        
        Args:
            normal_map: (3, H, W) normal map in camera space
            sh_coeffs: (9,) or (27,) SH coefficients
            
        Returns:
            (3, H, W) shading map
        """
        C, H, W = normal_map.shape
        device = normal_map.device
        
        # Reshape normals to (H*W, 3)
        normals = normal_map.permute(1, 2, 0).reshape(-1, 3)
        
        # Evaluate SH basis
        sh_basis = self._eval_sh_basis(normals)  # (H*W, 9)
        
        if sh_coeffs.shape[0] == 9:
            # Grayscale lighting
            shading = (sh_basis * sh_coeffs.unsqueeze(0)).sum(dim=-1, keepdim=True)
            shading = shading.expand(-1, 3)
        elif sh_coeffs.shape[0] == 27:
            # RGB lighting
            sh_coeffs_rgb = sh_coeffs.view(3, 9)
            shading = torch.stack([
                (sh_basis * sh_coeffs_rgb[c].unsqueeze(0)).sum(dim=-1)
                for c in range(3)
            ], dim=-1)
        else:
            raise ValueError(f"Unexpected sh_coeffs shape: {sh_coeffs.shape}")
        
        # Clamp and reshape
        shading = torch.clamp(shading, min=0.0)
        shading = shading.reshape(H, W, 3).permute(2, 0, 1)
        
        return shading
    
    def _eval_sh_basis(self, directions: Tensor) -> Tensor:
        """
        Evaluate order-2 SH basis functions.
        
        Uses the same SH order as ReliTalk's light_util.py:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-Y^2
        
        With proper normalization constants from spherical harmonics theory.
        """
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        
        # Normalization constants
        # C0 = 0.5 / sqrt(pi) for Y00
        # C1 = sqrt(3) / (2 * sqrt(pi)) for Y1m
        # C2 varies for Y2m
        
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
    
    def _render_fallback(self,
                         xyz: Tensor,
                         scales: Tensor,
                         rotations: Tensor,
                         opacity: Tensor,
                         albedo: Tensor,
                         normals: Tensor,
                         specular: Tensor,
                         camera: Camera,
                         sh_coeffs: Tensor) -> RenderOutput:
        """Fallback renderer when CUDA rasterizer is not available"""
        device = xyz.device
        H, W = camera.image_height, camera.image_width
        
        # Simple point-based rendering (very slow, for debugging only)
        print("Warning: Using fallback renderer (slow)")
        
        # Project points
        xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=device)], dim=-1)
        xyz_clip = (camera.full_proj_transform @ xyz_homo.T).T
        xyz_ndc = xyz_clip[:, :3] / xyz_clip[:, 3:4]
        
        # Convert to pixel coordinates
        px = ((xyz_ndc[:, 0] + 1) / 2 * W).long()
        py = ((1 - xyz_ndc[:, 1]) / 2 * H).long()  # Flip Y
        
        # Filter valid points
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (xyz_ndc[:, 2] > 0)
        
        # Initialize output maps
        albedo_map = self.bg_color.view(3, 1, 1).expand(3, H, W).clone()
        normal_map = torch.zeros(3, H, W, device=device)
        depth_map = torch.zeros(1, H, W, device=device)
        opacity_map = torch.zeros(1, H, W, device=device)
        
        # Simple depth-ordered splatting
        depths = xyz_ndc[:, 2]
        order = torch.argsort(depths[valid], descending=True)
        
        px_valid = px[valid][order]
        py_valid = py[valid][order]
        albedo_valid = albedo[valid][order]
        normals_valid = normals[valid][order]
        opacity_valid = opacity[valid][order]
        
        for i in range(len(px_valid)):
            x, y = px_valid[i], py_valid[i]
            a = opacity_valid[i, 0]
            albedo_map[:, y, x] = a * albedo_valid[i] + (1 - a) * albedo_map[:, y, x]
            normal_map[:, y, x] = a * normals_valid[i] + (1 - a) * normal_map[:, y, x]
            opacity_map[:, y, x] = a + (1 - a) * opacity_map[:, y, x]
        
        # Apply shading
        shading_map = self._compute_sh_shading_image(normal_map, sh_coeffs)
        rendered_image = albedo_map * shading_map
        
        return RenderOutput(
            rendered_image=rendered_image,
            albedo_map=albedo_map,
            normal_map=normal_map,
            depth_map=depth_map,
            opacity_map=opacity_map,
            shading_map=shading_map,
            specular_map=None,
            radii=torch.zeros(xyz.shape[0], device=device)
        )


def render_relightable(gaussians,
                       camera: Camera,
                       audio_features: Optional[Tensor] = None,
                       expression_params: Optional[Tensor] = None,
                       sh_coeffs: Optional[Tensor] = None,
                       bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                       scaling_modifier: float = 1.0) -> RenderOutput:
    """
    Convenience function for relightable rendering.
    
    Args:
        gaussians: RelightableGaussianModel
        camera: Camera parameters
        audio_features: Audio features for animation
        expression_params: FLAME expression parameters  
        sh_coeffs: SH lighting coefficients
        bg_color: Background color
        scaling_modifier: Scale modifier
        
    Returns:
        RenderOutput
    """
    renderer = RelightableGaussianRenderer(bg_color=bg_color)
    renderer = renderer.to(gaussians._xyz.device)
    
    return renderer(
        gaussians=gaussians,
        camera=camera,
        audio_features=audio_features,
        expression_params=expression_params,
        sh_coeffs=sh_coeffs,
        scaling_modifier=scaling_modifier
    )
