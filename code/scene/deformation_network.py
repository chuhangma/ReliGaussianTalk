"""
Deformation Network for Talking Head Animation
Maps audio/expression features to Gaussian deformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for input coordinates"""
    
    def __init__(self, d_in: int, n_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.d_in = d_in
        self.n_frequencies = n_frequencies
        self.include_input = include_input
        
        # Compute output dimension
        self.d_out = d_in * n_frequencies * 2
        if include_input:
            self.d_out += d_in
            
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0, n_frequencies - 1, n_frequencies)
        self.register_buffer('freq_bands', freq_bands)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (*, d_in) input coordinates
            
        Returns:
            (*, d_out) positionally encoded coordinates
        """
        out = []
        if self.include_input:
            out.append(x)
            
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
            
        return torch.cat(out, dim=-1)


class AudioEncoder(nn.Module):
    """Encode audio features for the deformation network"""
    
    def __init__(self, d_audio_in: int = 29, d_audio_out: int = 64, d_hidden: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_audio_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_audio_out),
        )
        
    def forward(self, audio_features: Tensor) -> Tensor:
        """
        Args:
            audio_features: (B, T, d_audio_in) or (B, d_audio_in) audio features
            
        Returns:
            (B, d_audio_out) encoded audio features
        """
        if audio_features.dim() == 3:
            # Average over time dimension
            audio_features = audio_features.mean(dim=1)
        return self.network(audio_features)


class ExpressionEncoder(nn.Module):
    """Encode FLAME expression parameters"""
    
    def __init__(self, d_expr_in: int = 50, d_expr_out: int = 64, d_hidden: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_expr_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_expr_out),
        )
        
    def forward(self, expression_params: Tensor) -> Tensor:
        """
        Args:
            expression_params: (B, d_expr_in) FLAME expression parameters
            
        Returns:
            (B, d_expr_out) encoded expression features
        """
        return self.network(expression_params)


class DeformationMLP(nn.Module):
    """MLP for predicting deformations"""
    
    def __init__(self, 
                 d_in: int,
                 d_hidden: int,
                 d_out: int,
                 n_layers: int = 4,
                 skip_layer: int = 2):
        super().__init__()
        
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_layers = n_layers
        self.skip_layer = skip_layer
        
        # Build layers
        layers = []
        for i in range(n_layers):
            if i == 0:
                layer_in = d_in
            elif i == skip_layer:
                layer_in = d_hidden + d_in
            else:
                layer_in = d_hidden
                
            if i == n_layers - 1:
                layer_out = d_out
            else:
                layer_out = d_hidden
                
            layers.append(nn.Linear(layer_in, layer_out))
            
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (*, d_in) input features
            
        Returns:
            (*, d_out) output
        """
        h = x
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class DeformationNetwork(nn.Module):
    """
    Network that predicts deformations for Gaussians based on audio/expression.
    
    Similar to GaussianTalker's deformation network but optimized for relighting.
    """
    
    def __init__(self,
                 d_feature: int = 256,
                 d_in: int = 3,
                 d_out_xyz: int = 3,
                 d_out_rot: int = 4,
                 d_out_scale: int = 3,
                 d_audio: int = 64,
                 d_expr: int = 50,
                 n_frequencies_xyz: int = 10,
                 use_audio: bool = True,
                 use_expression: bool = True):
        super().__init__()
        
        self.use_audio = use_audio
        self.use_expression = use_expression
        
        # Positional encoding for xyz
        self.pos_enc = PositionalEncoding(d_in, n_frequencies_xyz, include_input=True)
        d_xyz_encoded = self.pos_enc.d_out
        
        # Encoders for conditioning signals
        if use_audio:
            self.audio_encoder = AudioEncoder(d_audio_in=29, d_audio_out=d_audio)
        else:
            d_audio = 0
            
        if use_expression:
            self.expr_encoder = ExpressionEncoder(d_expr_in=d_expr, d_expr_out=d_expr)
        else:
            d_expr = 0
        
        # Total input dimension = encoded xyz + audio + expression
        d_total_in = d_xyz_encoded + d_audio + d_expr
        
        # Shared feature network
        self.feature_net = DeformationMLP(
            d_in=d_total_in,
            d_hidden=d_feature,
            d_out=d_feature,
            n_layers=4,
            skip_layer=2
        )
        
        # Output heads
        self.xyz_head = nn.Sequential(
            nn.Linear(d_feature, d_feature // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_feature // 2, d_out_xyz),
        )
        
        self.rot_head = nn.Sequential(
            nn.Linear(d_feature, d_feature // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_feature // 2, d_out_rot),
        )
        
        self.scale_head = nn.Sequential(
            nn.Linear(d_feature, d_feature // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_feature // 2, d_out_scale),
        )
        
        # Initialize outputs near zero for stable training
        self._init_output_heads()
        
    def _init_output_heads(self):
        """Initialize output heads to produce near-zero outputs initially"""
        for head in [self.xyz_head, self.rot_head, self.scale_head]:
            # Initialize last layer with small weights
            nn.init.zeros_(head[-1].bias)
            nn.init.normal_(head[-1].weight, std=0.001)
    
    def forward(self,
                xyz: Tensor,
                audio_features: Optional[Tensor] = None,
                expression_params: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict deformations for Gaussian positions.
        
        Args:
            xyz: (N, 3) Gaussian positions
            audio_features: (B, T, 29) or (B, 29) audio features
            expression_params: (B, 50) FLAME expression parameters
            
        Returns:
            xyz_offset: (N, 3) position offsets
            rot_offset: (N, 4) rotation quaternion offsets (normalized)
            scale_offset: (N, 3) scale offsets
        """
        N = xyz.shape[0]
        device = xyz.device
        
        # Encode xyz positions
        xyz_encoded = self.pos_enc(xyz)  # (N, d_xyz_encoded)
        
        # Build conditioning features
        cond_features = []
        
        if self.use_audio and audio_features is not None:
            audio_feat = self.audio_encoder(audio_features)  # (B, d_audio)
            # Expand to all Gaussians
            audio_feat = audio_feat.expand(N, -1)  # (N, d_audio)
            cond_features.append(audio_feat)
        elif self.use_audio:
            # Use zeros if no audio provided
            cond_features.append(torch.zeros(N, 64, device=device))
            
        if self.use_expression and expression_params is not None:
            expr_feat = self.expr_encoder(expression_params)  # (B, d_expr)
            # Expand to all Gaussians
            expr_feat = expr_feat.expand(N, -1)  # (N, d_expr)
            cond_features.append(expr_feat)
        elif self.use_expression:
            # Use zeros if no expression provided
            cond_features.append(torch.zeros(N, 50, device=device))
        
        # Concatenate all features
        if cond_features:
            features = torch.cat([xyz_encoded] + cond_features, dim=-1)
        else:
            features = xyz_encoded
        
        # Get shared features
        shared_feat = self.feature_net(features)
        
        # Predict outputs
        xyz_offset = self.xyz_head(shared_feat)
        rot_offset = self.rot_head(shared_feat)
        scale_offset = self.scale_head(shared_feat)
        
        # Normalize rotation quaternion
        # Start with identity rotation (1, 0, 0, 0) + small offset
        rot_offset = F.normalize(
            rot_offset + torch.tensor([1., 0., 0., 0.], device=device),
            dim=-1
        )
        
        # Scale offset should be small (multiplicative)
        scale_offset = 0.1 * torch.tanh(scale_offset)
        
        return xyz_offset, rot_offset, scale_offset


class FLAMEDeformationNetwork(nn.Module):
    """
    Deformation network that uses FLAME-based LBS for more accurate deformation.
    This version is closer to ReliTalk's deformation approach.
    """
    
    def __init__(self,
                 d_feature: int = 256,
                 n_joints: int = 5,  # Number of FLAME joints
                 d_expr: int = 50):
        super().__init__()
        
        self.n_joints = n_joints
        
        # Network to predict LBS weights for each Gaussian
        self.lbs_weight_net = nn.Sequential(
            nn.Linear(3, 128),  # xyz input
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_joints),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
        
        # Network to predict expression-dependent offset
        self.expr_offset_net = nn.Sequential(
            nn.Linear(3 + d_expr, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),  # xyz offset
        )
        
        # Initialize to produce small outputs
        nn.init.zeros_(self.expr_offset_net[-1].bias)
        nn.init.normal_(self.expr_offset_net[-1].weight, std=0.001)
        
    def forward(self,
                xyz: Tensor,
                joint_transforms: Tensor,
                expression_params: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Apply FLAME-based deformation to Gaussian positions.
        
        Args:
            xyz: (N, 3) canonical Gaussian positions
            joint_transforms: (J, 4, 4) joint transformation matrices
            expression_params: (B, 50) FLAME expression parameters
            
        Returns:
            xyz_deformed: (N, 3) deformed positions
            lbs_weights: (N, J) LBS weights
        """
        N = xyz.shape[0]
        device = xyz.device
        
        # Predict LBS weights
        lbs_weights = self.lbs_weight_net(xyz)  # (N, J)
        
        # Apply LBS transformation
        # Convert xyz to homogeneous coordinates
        xyz_homo = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=-1)  # (N, 4)
        
        # Apply weighted sum of transformations
        xyz_deformed = torch.zeros_like(xyz)
        for j in range(self.n_joints):
            T_j = joint_transforms[j]  # (4, 4)
            xyz_transformed = (T_j @ xyz_homo.T).T[:, :3]  # (N, 3)
            xyz_deformed += lbs_weights[:, j:j+1] * xyz_transformed
        
        # Add expression-dependent offset
        if expression_params is not None:
            expr_expanded = expression_params.expand(N, -1)  # (N, d_expr)
            offset_input = torch.cat([xyz, expr_expanded], dim=-1)
            expr_offset = self.expr_offset_net(offset_input)
            xyz_deformed = xyz_deformed + expr_offset
            
        return xyz_deformed, lbs_weights


class HybridDeformationNetwork(nn.Module):
    """
    Hybrid deformation network that combines:
    1. FLAME-based LBS for coarse deformation
    2. Audio-driven offsets for lip sync
    3. Expression-driven refinement
    """
    
    def __init__(self,
                 d_feature: int = 256,
                 n_joints: int = 5,
                 d_audio: int = 64,
                 d_expr: int = 50):
        super().__init__()
        
        self.n_joints = n_joints
        
        # FLAME-based LBS component
        self.flame_deform = FLAMEDeformationNetwork(
            d_feature=d_feature,
            n_joints=n_joints,
            d_expr=d_expr
        )
        
        # Audio-driven lip sync refinement
        self.audio_encoder = AudioEncoder(d_audio_in=29, d_audio_out=d_audio)
        
        # Lip region mask predictor
        self.lip_region_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 weight for lip region
        )
        
        # Audio-to-offset network (only affects lip region)
        self.audio_offset_net = nn.Sequential(
            nn.Linear(3 + d_audio, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        
        # Initialize audio offset to small values
        nn.init.zeros_(self.audio_offset_net[-1].bias)
        nn.init.normal_(self.audio_offset_net[-1].weight, std=0.001)
        
    def forward(self,
                xyz: Tensor,
                joint_transforms: Optional[Tensor] = None,
                audio_features: Optional[Tensor] = None,
                expression_params: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply hybrid deformation.
        
        Args:
            xyz: (N, 3) canonical positions
            joint_transforms: (J, 4, 4) FLAME joint transforms
            audio_features: (B, T, 29) audio features
            expression_params: (B, 50) expression parameters
            
        Returns:
            xyz_deformed: (N, 3) deformed positions
            rot_offset: (N, 4) rotation offsets (identity for now)
            scale_offset: (N, 3) scale offsets (zero for now)
        """
        N = xyz.shape[0]
        device = xyz.device
        
        # Start with canonical positions
        xyz_deformed = xyz.clone()
        
        # Apply FLAME-based LBS if transforms provided
        if joint_transforms is not None:
            xyz_deformed, lbs_weights = self.flame_deform(
                xyz, joint_transforms, expression_params
            )
        
        # Apply audio-driven lip sync offset
        if audio_features is not None:
            # Encode audio
            audio_feat = self.audio_encoder(audio_features)  # (B, d_audio)
            audio_feat_expanded = audio_feat.expand(N, -1)  # (N, d_audio)
            
            # Predict lip region weights
            lip_weights = self.lip_region_net(xyz)  # (N, 1)
            
            # Predict audio-driven offset
            audio_input = torch.cat([xyz, audio_feat_expanded], dim=-1)
            audio_offset = self.audio_offset_net(audio_input)  # (N, 3)
            
            # Apply offset weighted by lip region
            xyz_deformed = xyz_deformed + lip_weights * audio_offset
        
        # Return identity rotations and zero scale offsets
        rot_offset = torch.zeros(N, 4, device=device)
        rot_offset[:, 0] = 1.0  # Identity quaternion
        
        scale_offset = torch.zeros(N, 3, device=device)
        
        return xyz_deformed - xyz, rot_offset, scale_offset
