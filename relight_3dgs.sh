# ===== 1. 数据预处理 (如已有预处理数据可跳过) =====
cd /root/code/ReliTalk

# 提取视频帧、FLAME参数、掩码等
python preprocess/preprocess_video.py --video your_video.mp4 --output data/SUBJECT/ --resize 256

# ===== 2. Stage 1: 几何学习 =====
cd code/scripts
python train_gaussian_stage1.py \
    --conf ../confs/subject_gaussian.conf \
    --nepoch 200

# ===== 3. Stage 2: 重光照学习 =====
python train_gaussian_stage2.py \
    --conf ../confs/subject_gaussian.conf \
    --nepoch 300 \
    --stage1_checkpoint 200

# ===== 4. 测试/推理 =====
# 生成旋转光照视频
python test_gaussian_relight.py \
    --conf ../confs/subject_gaussian.conf \
    --mode video \
    --num_frames 125

# 多光照对比
python test_gaussian_relight.py \
    --conf ../confs/subject_gaussian.conf \
    --mode comparison