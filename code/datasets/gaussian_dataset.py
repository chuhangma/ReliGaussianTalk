"""
Dataset for Relightable Gaussian Talking Head
Supports loading video frames, FLAME parameters, audio features, and optional normal maps.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple
import glob


class GaussianRelightDataset(Dataset):
    """
    Dataset for training relightable Gaussian talking head.
    
    Expected directory structure:
    data_folder/
    └── subject_name/
        └── sub_dir/
            ├── image/          # RGB images (*.png or *.jpg)
            ├── mask/           # Foreground masks
            ├── normal/         # Normal maps (optional, from Stage 1)
            ├── parsing/        # Face parsing/semantics
            ├── flame_params.json   # FLAME parameters
            └── audio_features.npy  # Audio features (optional)
    """
    
    def __init__(self,
                 data_folder: str,
                 subject_name: str,
                 sub_dir: List[str],
                 json_name: str = "flame_params.json",
                 img_res: List[int] = [512, 512],
                 sample_size: int = -1,
                 subsample: int = 1,
                 use_semantics: bool = True,
                 use_normals: bool = False,
                 only_json: bool = False,
                 **kwargs):
        """
        Args:
            data_folder: Root data folder
            subject_name: Name of the subject
            sub_dir: List of subdirectories to use (e.g., ['train', 'val'])
            json_name: Name of the FLAME parameters JSON file
            img_res: Image resolution [H, W]
            sample_size: Number of pixels to sample per image (-1 for all)
            subsample: Subsample factor for frames
            use_semantics: Whether to load semantic/parsing maps
            use_normals: Whether to load normal maps (for Stage 2)
            only_json: Only load JSON parameters, skip images
        """
        self.data_folder = data_folder
        self.subject_name = subject_name
        self.sub_dirs = sub_dir if isinstance(sub_dir, list) else [sub_dir]
        self.img_res = img_res
        self.sample_size = sample_size
        self.subsample = subsample
        self.use_semantics = use_semantics
        self.use_normals = use_normals
        self.only_json = only_json
        
        self.total_pixels = img_res[0] * img_res[1]
        
        # Load data
        self.data = self._load_data(json_name)
        
        print(f"Loaded {len(self)} samples from {self.sub_dirs}")
        
    def _load_data(self, json_name: str) -> Dict:
        """Load all data from disk"""
        data = {
            "image_paths": [],
            "mask_paths": [],
            "normal_paths": [],
            "parsing_paths": [],
            "sub_dirs": [],
            "frame_ids": [],
            "expressions": [],
            "flame_pose": [],
            "world_mats": [],
            "intrinsics": [],
        }
        
        for sub_dir in self.sub_dirs:
            base_path = os.path.join(self.data_folder, self.subject_name, sub_dir)
            
            # Load FLAME parameters
            json_path = os.path.join(base_path, json_name)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    flame_params = json.load(f)
            else:
                print(f"Warning: {json_path} not found, using default parameters")
                flame_params = {"frames": []}
            
            # Find all images
            image_dir = os.path.join(base_path, "image")
            if not os.path.exists(image_dir):
                image_dir = base_path  # Images might be in root
                
            image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            if not image_paths:
                image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            
            # Subsample
            image_paths = image_paths[::self.subsample]
            
            for i, img_path in enumerate(image_paths):
                frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
                
                data["image_paths"].append(img_path)
                data["sub_dirs"].append(sub_dir)
                data["frame_ids"].append(frame_id)
                
                # Mask path
                mask_dir = os.path.join(base_path, "mask")
                mask_path = os.path.join(mask_dir, f"{frame_id:05d}.png")
                if not os.path.exists(mask_path):
                    mask_path = img_path.replace("image", "mask")
                data["mask_paths"].append(mask_path)
                
                # Normal path (for Stage 2)
                if self.use_normals:
                    normal_dir = os.path.join(base_path, "normal")
                    normal_path = os.path.join(normal_dir, f"{frame_id:05d}.png")
                    if not os.path.exists(normal_path):
                        normal_path = None
                    data["normal_paths"].append(normal_path)
                
                # Parsing/semantics path
                if self.use_semantics:
                    parsing_dir = os.path.join(base_path, "parsing")
                    parsing_path = os.path.join(parsing_dir, f"{frame_id:05d}.png")
                    if not os.path.exists(parsing_path):
                        parsing_path = None
                    data["parsing_paths"].append(parsing_path)
                
                # FLAME parameters
                if frame_id < len(flame_params.get("frames", [])):
                    frame_params = flame_params["frames"][frame_id]
                    data["expressions"].append(frame_params.get("expression", [0.0] * 50))
                    data["flame_pose"].append(frame_params.get("pose", [0.0] * 15))
                    data["world_mats"].append(frame_params.get("world_mat", np.eye(4).tolist()))
                    data["intrinsics"].append(frame_params.get("intrinsics", np.eye(3).tolist()))
                else:
                    # Default parameters
                    data["expressions"].append([0.0] * 50)
                    data["flame_pose"].append([0.0] * 15)
                    data["world_mats"].append(np.eye(4).tolist())
                    data["intrinsics"].append(np.eye(3).tolist())
        
        # Convert to tensors
        data["expressions"] = torch.tensor(data["expressions"], dtype=torch.float32)
        data["flame_pose"] = torch.tensor(data["flame_pose"], dtype=torch.float32)
        data["world_mats"] = torch.tensor(data["world_mats"], dtype=torch.float32)
        data["intrinsics"] = torch.tensor(data["intrinsics"], dtype=torch.float32)
        
        # Load shape parameters (shared across all frames)
        shape_path = os.path.join(self.data_folder, self.subject_name, "shape_params.npy")
        if os.path.exists(shape_path):
            self.shape_params = torch.tensor(np.load(shape_path), dtype=torch.float32)
        else:
            self.shape_params = torch.zeros(100, dtype=torch.float32)
        
        # Load audio features if available
        audio_path = os.path.join(self.data_folder, self.subject_name, self.sub_dirs[0], "audio_features.npy")
        if os.path.exists(audio_path):
            data["audio_features"] = torch.tensor(np.load(audio_path), dtype=torch.float32)
        else:
            data["audio_features"] = None
            
        return data
    
    def __len__(self) -> int:
        return len(self.data["image_paths"])
    
    def __getitem__(self, idx: int) -> Tuple[int, Dict, Dict]:
        """
        Returns:
            idx: Sample index
            model_input: Dictionary with inputs for the model
            ground_truth: Dictionary with ground truth data
        """
        model_input = {}
        ground_truth = {}
        
        # Basic info
        model_input["idx"] = torch.tensor([idx])
        model_input["sub_dir"] = self.data["sub_dirs"][idx]
        model_input["img_name"] = torch.tensor([[self.data["frame_ids"][idx]]])
        
        # FLAME parameters
        model_input["expression"] = self.data["expressions"][idx]
        model_input["flame_pose"] = self.data["flame_pose"][idx]
        model_input["cam_pose"] = self.data["world_mats"][idx]
        model_input["intrinsics"] = self.data["intrinsics"][idx]
        
        # Audio features (if available)
        if self.data["audio_features"] is not None and idx < len(self.data["audio_features"]):
            model_input["audio_features"] = self.data["audio_features"][idx]
        
        if self.only_json:
            return idx, model_input, ground_truth
        
        # Load RGB image
        img_path = self.data["image_paths"][idx]
        img = self._load_image(img_path)
        ground_truth["rgb"] = img.reshape(-1, 3)  # (H*W, 3)
        
        # Load mask
        mask_path = self.data["mask_paths"][idx]
        if os.path.exists(mask_path):
            mask = self._load_mask(mask_path)
        else:
            mask = torch.ones((self.img_res[0], self.img_res[1]), dtype=torch.float32)
        model_input["object_mask"] = mask.reshape(-1)  # (H*W,)
        
        # Load normal map (for Stage 2)
        if self.use_normals and idx < len(self.data["normal_paths"]):
            normal_path = self.data["normal_paths"][idx]
            if normal_path and os.path.exists(normal_path):
                normal = self._load_normal(normal_path)
                ground_truth["normal"] = normal.reshape(-1, 3)  # (H*W, 3)
        
        # Load semantics/parsing
        if self.use_semantics and idx < len(self.data["parsing_paths"]):
            parsing_path = self.data["parsing_paths"][idx]
            if parsing_path and os.path.exists(parsing_path):
                semantics = self._load_semantics(parsing_path)
                ground_truth["semantics"] = semantics.reshape(-1, semantics.shape[-1])
        
        # Generate UV coordinates for all pixels
        uv = self._get_uv_grid()
        model_input["uv"] = uv.reshape(-1, 2)  # (H*W, 2)
        
        # Sampling if sample_size > 0
        if self.sample_size > 0 and self.sample_size < self.total_pixels:
            indices = torch.randperm(self.total_pixels)[:self.sample_size]
            
            model_input["uv"] = model_input["uv"][indices]
            model_input["object_mask"] = model_input["object_mask"][indices]
            ground_truth["rgb"] = ground_truth["rgb"][indices]
            
            if "normal" in ground_truth:
                ground_truth["normal"] = ground_truth["normal"][indices]
            if "semantics" in ground_truth:
                ground_truth["semantics"] = ground_truth["semantics"][indices]
        
        return idx, model_input, ground_truth
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess RGB image"""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.img_res[1], self.img_res[0]), Image.BILINEAR)
        img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        # Convert to [-1, 1] range
        img = img * 2 - 1
        return img
    
    def _load_mask(self, path: str) -> torch.Tensor:
        """Load foreground mask"""
        mask = Image.open(path).convert('L')
        mask = mask.resize((self.img_res[1], self.img_res[0]), Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.float32) / 255.0
        return mask
    
    def _load_normal(self, path: str) -> torch.Tensor:
        """Load normal map"""
        normal = Image.open(path).convert('RGB')
        normal = normal.resize((self.img_res[1], self.img_res[0]), Image.BILINEAR)
        normal = torch.tensor(np.array(normal), dtype=torch.float32) / 255.0
        # Convert from [0,1] to [-1,1]
        normal = normal * 2 - 1
        # Normalize
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
        return normal
    
    def _load_semantics(self, path: str) -> torch.Tensor:
        """Load semantic parsing map"""
        parsing = Image.open(path).convert('L')
        parsing = parsing.resize((self.img_res[1], self.img_res[0]), Image.NEAREST)
        parsing = torch.tensor(np.array(parsing), dtype=torch.long)
        
        # Convert to one-hot encoding
        # Classes: 0=background, 1=skin, 2=eyebrow, 3=eye, 4=nose, 5=lip, 6=hair, 7=neck, 8=cloth
        num_classes = 9
        one_hot = torch.zeros((parsing.shape[0], parsing.shape[1], num_classes), dtype=torch.float32)
        for c in range(num_classes):
            one_hot[:, :, c] = (parsing == c).float()
        
        return one_hot
    
    def _get_uv_grid(self) -> torch.Tensor:
        """Generate UV coordinate grid"""
        v, u = torch.meshgrid(
            torch.linspace(-1, 1, self.img_res[0]),
            torch.linspace(-1, 1, self.img_res[1]),
            indexing='ij'
        )
        uv = torch.stack([u, v], dim=-1)
        return uv
    
    def collate_fn(self, batch):
        """Custom collate function for DataLoader"""
        indices = [b[0] for b in batch]
        model_inputs = {}
        ground_truths = {}
        
        # Stack tensors
        for key in batch[0][1].keys():
            vals = [b[1][key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                model_inputs[key] = torch.stack(vals, dim=0)
            else:
                model_inputs[key] = vals
        
        for key in batch[0][2].keys():
            vals = [b[2][key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                ground_truths[key] = torch.stack(vals, dim=0)
            else:
                ground_truths[key] = vals
        
        return indices, model_inputs, ground_truths


class GaussianInitDataset(Dataset):
    """
    Dataset for initializing Gaussian point cloud from FLAME mesh.
    
    NOTE: Following GaussianTalker's approach, we use the exact number of 
    vertices from the 3DMM (FLAME/BFM) mesh for initialization instead of
    arbitrary upsampling. This provides better initialization for talking heads.
    
    - BFM: 34,650 vertices (GaussianTalker default)
    - FLAME: ~5,023 vertices
    """
    
    def __init__(self, 
                 flame_model_path: str,
                 shape_params: torch.Tensor,
                 num_samples: int = None,  # Now defaults to None = use mesh vertices directly
                 use_mesh_vertices: bool = True):  # New flag to use direct mesh vertices
        """
        Args:
            flame_model_path: Path to FLAME model
            shape_params: FLAME shape parameters (100,)
            num_samples: Number of points to sample (if None and use_mesh_vertices=True, 
                         use exact mesh vertex count like GaussianTalker)
            use_mesh_vertices: If True, use exact mesh vertices without up/downsampling
        """
        self.flame_model_path = flame_model_path
        self.shape_params = shape_params
        self.num_samples = num_samples
        self.use_mesh_vertices = use_mesh_vertices
        
    def generate_init_points(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate initial point cloud from FLAME mesh.
        
        Following GaussianTalker approach:
        - If use_mesh_vertices=True, use exact mesh vertex positions
        - Otherwise, up/downsample to num_samples
        
        Returns:
            points: (N, 3) 3D positions
            colors: (N, 3) initial colors (gray)
            normals: (N, 3) normal vectors
        """
        mesh_loaded = False
        
        try:
            # Try to load FLAME model
            from flame.FLAME import FLAME
            
            # Get shape parameters count from self.shape_params
            n_shape = self.shape_params.shape[0] if len(self.shape_params.shape) == 1 else self.shape_params.shape[1]
            n_exp = 50  # Default expression params count
            
            # Prepare shape_params for FLAME initialization
            shape_params_2d = self.shape_params.unsqueeze(0) if len(self.shape_params.shape) == 1 else self.shape_params
            
            flame = FLAME(
                flame_model_path=self.flame_model_path,
                n_shape=n_shape,
                n_exp=n_exp,
                shape_params=shape_params_2d
            ).cuda()
            
            # Get canonical mesh vertices using forward() method
            # forward() takes: expression_params (N, n_exp), full_pose (N, 15)
            with torch.no_grad():
                expression = torch.zeros(1, n_exp).cuda()
                full_pose = torch.zeros(1, 15).cuda()  # FLAME uses 15 pose params
                
                # forward returns: vertices, pose_feature, transformations
                vertices, _, _ = flame.forward(expression, full_pose)
                vertices = vertices[0].cpu()  # (V, 3), move back to CPU
                
                # Get faces for normal computation
                faces = flame.faces_tensor.cpu()
                
                # Compute vertex normals
                normals = self._compute_vertex_normals(vertices, faces)
                
            mesh_loaded = True
            original_vertex_count = vertices.shape[0]
            print(f"[GaussianInitDataset] Loaded FLAME mesh with {original_vertex_count} vertices")
                
        except Exception as e:
            print(f"Failed to load FLAME model: {e}")
            print("Using random sphere initialization instead")
            mesh_loaded = False
        
        # Determine final point count
        if self.use_mesh_vertices and mesh_loaded:
            # Use exact mesh vertices like GaussianTalker
            final_count = vertices.shape[0]
            print(f"[GaussianInitDataset] Using exact mesh vertices: {final_count} Gaussians")
        elif self.num_samples is not None:
            final_count = self.num_samples
        else:
            # Default fallback
            final_count = 5023  # FLAME default vertex count
            print(f"[GaussianInitDataset] Using default FLAME vertex count: {final_count}")
            
        if not mesh_loaded:
            # Random sphere initialization with final_count points
            theta = torch.rand(final_count) * 2 * np.pi
            phi = torch.acos(2 * torch.rand(final_count) - 1)
            r = 0.2 + 0.1 * torch.rand(final_count)  # Head-sized sphere
            
            x = r * torch.sin(phi) * torch.cos(theta)
            y = r * torch.sin(phi) * torch.sin(theta) 
            z = r * torch.cos(phi)
            
            vertices = torch.stack([x, y, z], dim=1)
            normals = vertices / torch.norm(vertices, dim=1, keepdim=True)
        
        # Only apply up/downsampling if NOT using mesh vertices directly
        if not self.use_mesh_vertices and mesh_loaded and self.num_samples is not None:
            if vertices.shape[0] < self.num_samples:
                # Upsample by adding noise
                repeat_factor = (self.num_samples // vertices.shape[0]) + 1
                vertices = vertices.repeat(repeat_factor, 1)[:self.num_samples]
                normals = normals.repeat(repeat_factor, 1)[:self.num_samples]
                
                # Add small noise to break exact duplicates
                vertices = vertices + 0.001 * torch.randn_like(vertices)
                print(f"[GaussianInitDataset] Upsampled to {self.num_samples} points")
            elif vertices.shape[0] > self.num_samples:
                # Random subsample
                indices = torch.randperm(vertices.shape[0])[:self.num_samples]
                vertices = vertices[indices]
                normals = normals[indices]
                print(f"[GaussianInitDataset] Downsampled to {self.num_samples} points")
        
        # Final count for colors
        actual_count = vertices.shape[0]
        
        # Initialize colors as gray (0.5)
        colors = 0.5 * torch.ones((actual_count, 3))
        
        print(f"[GaussianInitDataset] Final initialization: {actual_count} Gaussians")
        
        return vertices, colors, normals
    
    def _compute_vertex_normals(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Compute per-vertex normals from mesh"""
        # Get face vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Compute face normals
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
        
        # Accumulate to vertex normals
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.index_add_(0, faces[:, i], face_normals)
        
        # Normalize
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
        
        return vertex_normals
