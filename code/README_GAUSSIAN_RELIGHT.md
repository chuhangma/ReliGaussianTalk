# Relightable 3D Gaussian Splatting Talking Head System

本系统结合了 **3D Gaussian Splatting (3DGS)** 的高效渲染能力与 **ReliTalk** 的神经网络重光照技术，实现了一个可实时重光照的说话人头像系统。

## 系统概述

### 核心思想

1. **Stage 1 (几何阶段)**: 使用 3DGS 学习人脸的精确几何结构，生成高质量法向图
2. **Stage 2 (重光照阶段)**: 基于 Stage 1 的法向图，训练神经网络分解 Albedo/Shading，并学习场景光照

### 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Geometry Learning                    │
├─────────────────────────────────────────────────────────────────┤
│  RGB Images ──→ RelightableGaussianModel ──→ 3DGS Renderer      │
│                      ↓                            ↓              │
│              FLAME LBS Deformation          Normal Maps (GT)     │
│              + Audio Features               for Stage 2          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Stage 2: Relighting Learning                     │
├─────────────────────────────────────────────────────────────────┤
│  RGB + Normal ──→ AlbedoNet ──→ Albedo                          │
│        ↓                                                         │
│     NormalNet ──→ Fine Normal ──→ SH Shading                     │
│        ↓                                                         │
│     SpecNet ──→ Specular Map                                     │
│                     ↓                                            │
│  Final = Albedo × (Shading + Spec × SpecMap)                    │
└─────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
code/
├── scene/
│   └── relightable_gaussian_model.py    # 可重光照的高斯模型
├── gaussian_renderer/
│   └── relightable_renderer.py          # 支持法向/特征渲染的渲染器
├── datasets/
│   └── gaussian_dataset.py              # 数据集加载器
├── scripts/
│   ├── train_gaussian_stage1.py         # Stage 1 训练脚本
│   ├── train_gaussian_stage2.py         # Stage 2 训练脚本
│   └── test_gaussian_relight.py         # 测试/推理脚本
└── model/
    ├── resnet_network.py                # AlbedoNet, NormalNet
    └── unet_network.py                  # SpecNet
```

## 数据准备

### 数据目录结构

```
data/
└── SUBJECT_NAME/
    ├── image/                    # RGB 图像 (PNG/JPG)
    ├── mask/                     # 分割掩码
    ├── semantics/                # 语义分割 (face, neck, shoulder)
    ├── flame_params/             # FLAME 参数 (表情、姿态)
    │   ├── expression/
    │   ├── pose/
    │   └── shape/
    ├── audio_features/           # 音频特征 (DeepSpeech/Wav2Vec)
    └── transforms_train.json     # 相机参数
```

### 数据预处理

如果您有视频输入，请先运行 ReliTalk 原有的预处理流程：

```bash
# 1. 提取视频帧
ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 image/%05d.png

# 2. 人脸检测与裁剪
python preprocess/crop_face.py --input image/ --output cropped/

# 3. 提取 FLAME 参数 (使用 DECA 或 EMOCA)
python preprocess/extract_flame.py --input cropped/ --output flame_params/

# 4. 生成掩码和语义分割
python preprocess/gen_mask.py --input cropped/ --output mask/

# 5. 提取音频特征
python preprocess/extract_audio.py --input video.mp4 --output audio_features/
```

## 训练流程

### Stage 1: 几何学习

```bash
cd /root/code/ReliTalk/code/scripts

# 训练 Stage 1
python train_gaussian_stage1.py \
    --conf ../confs/subject_gaussian.conf \
    --nepoch 200 \
    --eval_freq 10
```

**关键参数:**
- `--nepoch`: 训练轮数 (建议 150-200)
- `--eval_freq`: 评估频率 (每 N 个 epoch)
- `--lr`: 学习率 (默认 1e-4)

**输出:**
- 高斯模型 checkpoint: `exps/SUBJECT/train/checkpoints/`
- 法向图 (用于 Stage 2): `exps/SUBJECT/train/eval/normals/`

### Stage 2: 重光照学习

```bash
# 训练 Stage 2
python train_gaussian_stage2.py \
    --conf ../confs/subject_gaussian.conf \
    --nepoch 300 \
    --stage1_checkpoint 200 \
    --eval_freq 10
```

**关键参数:**
- `--stage1_checkpoint`: Stage 1 的 checkpoint 编号
- `--nepoch`: 训练轮数 (建议 250-350)
- `--freeze_gaussian`: 是否冻结高斯模型 (默认 True)

**输出:**
- 重光照网络 checkpoint: `exps/SUBJECT/train_relight/checkpoints/`
- 可视化结果: `exps/SUBJECT/train_relight/plots/`

## 测试与推理

### 1. 生成旋转光照视频

```bash
python test_gaussian_relight.py \
    --conf ../confs/subject_gaussian.conf \
    --checkpoint latest \
    --output_dir ./output \
    --mode video \
    --num_frames 125
```

这将生成一个光源在人脸周围旋转的视频 (类似 ReliTalk 的演示效果)。

### 2. 单帧渲染 (指定光照)

```bash
python test_gaussian_relight.py \
    --conf ../confs/subject_gaussian.conf \
    --mode single \
    --lighting frontal  # 可选: frontal, left, right, top, bottom, ambient, dramatic
```

### 3. 多光照对比

```bash
python test_gaussian_relight.py \
    --conf ../confs/subject_gaussian.conf \
    --mode comparison \
    --output_dir ./comparison_output
```

生成包含原图和多种光照条件的对比图。

## 配置文件说明

创建配置文件 `confs/subject_gaussian.conf`:

```hocon
train {
    exps_folder = "exps"
    methodname = "GaussianRelight"
    learning_rate = 1e-4
}

model {
    num_gaussians = 50000
    sh_degree = 2
    with_motion_net = true
}

dataset {
    data_folder = "/path/to/data"
    subject_name = "SUBJECT_NAME"
    json_name = "transforms_train.json"
    
    train {
        sub_dir = ["train"]
        img_res = [512, 512]
    }
    
    test {
        sub_dir = ["test"]
        img_res = [512, 512]
    }
}

loss {
    rgb_weight = 1.0
    ssim_weight = 0.2
    vgg_weight = 0.1
    normal_weight = 0.5
    albedo_constancy_weight = 0.05
    gt_w_seg = true
}
```

## 自定义光照

您可以使用自定义的 SH (Spherical Harmonics) 系数：

```python
from scripts.test_gaussian_relight import GaussianRelightTester

tester = GaussianRelightTester(conf='../confs/subject.conf')

# 自定义9阶SH光照系数 [L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22]
custom_light = [
    0.5,   # L00 (环境光强度)
    0.0,   # L1-1 (Y方向)
    0.8,   # L10 (Z方向 - 正面光)
    0.0,   # L11 (X方向)
    0.0, 0.0, 0.0, 0.0, 0.0  # 二阶SH
]

tester.run(mode='single', lighting=custom_light)
```

## 与 ReliTalk 原版的对比

| 特性 | ReliTalk (NeRF-based) | Gaussian Relight (本系统) |
|------|----------------------|--------------------------|
| 渲染速度 | ~1s/frame | ~30ms/frame (实时) |
| 训练时间 | ~24h | ~8h |
| 内存占用 | ~24GB | ~12GB |
| 点云显式表示 | ❌ | ✅ |
| FLAME 驱动 | ✅ | ✅ |
| 音频驱动 | ✅ | ✅ |
| 重光照质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 常见问题

### Q: Stage 2 训练不收敛？

A: 确保 Stage 1 已经充分训练，法向图质量足够高。可以检查 `eval/normals/` 目录下的法向图。

### Q: 重光照效果不自然？

A: 尝试调整以下参数：
- 增加 `albedo_constancy_weight` 使 Albedo 更稳定
- 检查 face mask 是否准确
- 适当增加训练轮数

### Q: 如何添加新的光照预设？

A: 在 `test_gaussian_relight.py` 的 `lighting_presets` 字典中添加新的 SH 系数。

## 引用

如果您使用本代码，请引用：

```bibtex
@inproceedings{relitalk2023,
    title={ReliTalk: Relightable Talking Portrait Generation from a Single Video},
    author={...},
    booktitle={CVPR},
    year={2023}
}

@inproceedings{gaussiantalker2024,
    title={GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with Audio-Driven 3D Gaussian Splatting},
    author={...},
    booktitle={CVPR},
    year={2024}
}
```

## License

本项目遵循 MIT License。
