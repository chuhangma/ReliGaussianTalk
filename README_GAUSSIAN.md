# Relightable Gaussian Talking Head

This project combines **3D Gaussian Splatting** with **Relightable Rendering** for talking head synthesis.

## Features

- **3D Gaussian Splatting**: Efficient real-time rendering using 3D Gaussians
- **Relightable Rendering**: Support for novel lighting conditions using Spherical Harmonics
- **Audio-driven Animation**: Lip sync driven by audio features
- **Expression Control**: FLAME-based expression parameters for facial animation

## Architecture

```
ReliTalk/
├── code/
│   ├── gaussian_renderer/     # Relightable Gaussian Renderer
│   │   └── __init__.py       # Camera, RelightableGaussianRenderer
│   ├── scene/                # Model definitions
│   │   ├── __init__.py
│   │   ├── relightable_gaussian_model.py  # Main Gaussian model
│   │   └── deformation_network.py         # Animation networks
│   ├── scripts/              # Training and inference
│   │   ├── train_gaussian_relight.py      # Training script
│   │   └── inference_relight.py           # Inference script
│   └── utils/                # Utilities from original ReliTalk
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x or 12.x
- PyTorch 2.1.0+

### Install Dependencies

```bash
# Create conda environment
conda create -n relight_gaussian python=3.8
conda activate relight_gaussian

# Install PyTorch
pip install torch==2.1.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diff-gaussian-rasterization
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# Install simple-knn
pip install git+https://github.com/camenduru/simple-knn.git

# Install other dependencies
pip install numpy scipy pillow opencv-python tqdm tensorboard plyfile
```

## Usage

### Training

```bash
python code/scripts/train_gaussian_relight.py \
    --data_dir /path/to/data \
    --subject person_name \
    --iterations 30000 \
    --output_dir ./output
```

### Inference

#### Render with different lighting conditions
```bash
python code/scripts/inference_relight.py \
    --checkpoint ./output/person_name_xxx/final_model.pth \
    --output ./results/comparison \
    --mode comparison
```

#### Render a video with lighting animation
```bash
python code/scripts/inference_relight.py \
    --checkpoint ./output/person_name_xxx/final_model.pth \
    --output ./results/video.mp4 \
    --mode video \
    --lighting frontal \
    --lighting_end dramatic \
    --num_frames 100
```

#### Render with audio
```bash
python code/scripts/inference_relight.py \
    --checkpoint ./output/person_name_xxx/final_model.pth \
    --output ./results/talking.mp4 \
    --mode video \
    --audio /path/to/audio.npy \
    --num_frames 100
```

## Data Format

Expected data structure:
```
data_dir/
├── subject_name/
│   ├── images/        # RGB images (PNG or JPG)
│   ├── masks/         # Face masks (optional)
│   ├── cameras.npz    # Camera parameters
│   ├── audio/         # Audio features per frame (.npy)
│   ├── flame/         # FLAME parameters per frame (.npz)
│   └── lighting/      # SH coefficients per frame (.npy, optional)
```

## Lighting Presets

Available lighting presets:
- `frontal`: Standard frontal lighting
- `left`: Left-side lighting
- `right`: Right-side lighting
- `top`: Top lighting
- `bottom`: Bottom lighting
- `ambient`: Uniform ambient lighting
- `dramatic`: High-contrast dramatic lighting
- `soft`: Soft, diffused lighting

## Technical Details

### Relightable Gaussian Model

The `RelightableGaussianModel` extends standard 3D Gaussian Splatting with:

1. **Albedo instead of SH colors**: Each Gaussian stores view-independent albedo (RGB)
2. **Explicit normals**: Normals for each Gaussian (computed from covariance or stored explicitly)
3. **Specular coefficients**: Per-Gaussian specular intensity
4. **Roughness**: Per-Gaussian surface roughness

### Rendering Equation

```
Color = Albedo × (SH_Shading + Specular × Specular_Map)
```

where:
- `SH_Shading` is computed using Spherical Harmonics basis functions evaluated at normal directions
- `Specular_Map` is the rendered specular intensity

### Spherical Harmonics (Order 2)

We use order-2 (9 coefficients) spherical harmonics for lighting:
- `Y_00`: Ambient term
- `Y_1m`, `Y_10`, `Y_1p`: First-order directional terms
- `Y_2m2`, `Y_2m1`, `Y_20`, `Y_2p1`, `Y_2p2`: Second-order terms

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [GaussianTalker](https://github.com/xxx/GaussianTalker)
- [ReliTalk](https://github.com/ywq/ReliTalk)
- [FLAME](https://flame.is.tue.mpg.de/)

## License

This project is for research purposes only.

## Citation

If you use this code, please cite:

```bibtex
@article{relightable_gaussian_talking_head,
  title={Relightable Gaussian Talking Head},
  year={2024}
}
```
