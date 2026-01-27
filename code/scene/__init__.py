"""
Scene module for Relightable Gaussian Talking Head

This module contains:
- RelightableGaussianModel: 3D Gaussian model with relighting support
- DeformationNetwork: Network for expression/audio-driven animation
"""

from .relightable_gaussian_model import RelightableGaussianModel
from .deformation_network import (
    DeformationNetwork,
    FLAMEDeformationNetwork,
    HybridDeformationNetwork,
    AudioEncoder,
    ExpressionEncoder,
    PositionalEncoding,
)

__all__ = [
    'RelightableGaussianModel',
    'DeformationNetwork',
    'FLAMEDeformationNetwork',
    'HybridDeformationNetwork',
    'AudioEncoder',
    'ExpressionEncoder',
    'PositionalEncoding',
]
