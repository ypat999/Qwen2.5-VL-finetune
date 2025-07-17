#!/bin/bash
# ===============================================
# Qwen2.5-VL-3B 指令微调启动脚本（含详细中文注释）
# ===============================================

# ===== 分布式配置 =====
MASTER_ADDR="127.0.0.1"                          # 主节点 IP（本地单机多卡使用 127.0.0.1）
MASTER_PORT=$(shuf -i 20000-29999 -n 1)          # 随机选取端口，避免冲突
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l) # 自动检测可用 GPU 数量

# ===== 路径配置 =====
MODEL_PATH="/public/ywj/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"  # 模型权重目录
OUTPUT_DIR="./checkpoints"                       # 模型输出目录
CACHE_DIR="./cache"                              # 模型缓存目录
DATASETS="zdjy%100"                              # 数据集名称及采样比例，例如 dataset_name%percentage

# ===== 模型训练参数 =====
TUNE_MM_LLM=true                                  # 是否训练语言模型部分
TUNE_MM_VISION=false                              # 是否训练视觉编码器（VIT）
TUNE_MM_MLP=false                                  # 是否训练多模态投影层 MLP
VISION_TOWER_LR=1e-6                              # 视觉编码器的学习率
MM_PROJECTOR_LR=1e-5                              # 多模态投影层的学习率
LEARNING_RATE=2e-7                                # 主学习率（用于语言模型部分）
OPTIM="adamw_torch"                               # 优化器类型
WEIGHT_DECAY=0.01                                 # 权重衰减系数（L2 正则）

# ===== 精度与内存优化 =====
USE_BF16=true                                     # 是否使用 bfloat16 精度（推荐 A100）
BATCH_SIZE=1                                      # 每张 GPU 的 batch size
GRAD_ACC=1                                        # 梯度累积步数（等效大 batch size）

# ===== 训练调度参数 =====
NUM_EPOCHS=3                                      # 总训练轮数
WARMUP_RATIO=0.03                                 # 学习率预热比例
LR_SCHEDULER_TYPE="cosine"                        # 学习率调度策略，可选 linear/cosine 等

# ===== 序列处理参数 =====
MODEL_MAX_LENGTH=1024                             # 最大文本序列长度
DATA_FLATTEN=true                                 # 是否将一批样本合并成一个长序列
DATA_PACKING=true                                 # 是否使用数据打包（packing）

# ===== 图像处理参数 =====
MAX_PIXELS=$((576*28*28))                         # 图像最大像素数（H×W）
MIN_PIXELS=$((16*28*28))                          # 图像最小像素数

# ===== 视频处理参数 =====
BASE_INTERVAL=2                                   # 视频帧采样间隔（秒）
VIDEO_MAX_FRAMES=8                                # 每段视频的最大帧数
VIDEO_MIN_FRAMES=4                                # 每段视频的最小帧数
VIDEO_MAX_FRAME_PIXELS=$((1664*28*28))            # 单帧图像最大像素
VIDEO_MIN_FRAME_PIXELS=$((256*28*28))             # 单帧图像最小像素

# ===== 日志与检查点参数 =====
LOGGING_STEPS=10                                  # 日志记录步数间隔
SAVE_STEPS=500                                    # 检查点保存间隔（步）
SAVE_TOTAL_LIMIT=3                                # 最多保留的检查点数量

# ===== DeepSpeed 配置 =====
DEEPSPEED_CONFIG="qwen-vl-finetune/scripts/zero3_offload.json"             # DeepSpeed 配置文件路径

# ===== 启动训练 =====
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwen-vl-finetune/qwenvl/train/train_qwen.py \
         --model_name_or_path "$MODEL_PATH" \
         --tune_mm_llm $TUNE_MM_LLM \
         --tune_mm_vision $TUNE_MM_VISION \
         --tune_mm_mlp $TUNE_MM_MLP \
         --dataset_use "$DATASETS" \
         --output_dir "$OUTPUT_DIR" \
         --cache_dir "$CACHE_DIR" \
         --bf16 $USE_BF16 \
         --per_device_train_batch_size $BATCH_SIZE \
         --gradient_accumulation_steps $GRAD_ACC \
         --learning_rate $LEARNING_RATE \
         --mm_projector_lr $MM_PROJECTOR_LR \
         --vision_tower_lr $VISION_TOWER_LR \
         --optim $OPTIM \
         --model_max_length $MODEL_MAX_LENGTH \
         --data_flatten $DATA_FLATTEN \
         --data_packing $DATA_PACKING \
         --max_pixels $MAX_PIXELS \
         --min_pixels $MIN_PIXELS \
         --base_interval $BASE_INTERVAL \
         --video_max_frames $VIDEO_MAX_FRAMES \
         --video_min_frames $VIDEO_MIN_FRAMES \
         --video_max_frame_pixels $VIDEO_MAX_FRAME_PIXELS \
         --video_min_frame_pixels $VIDEO_MIN_FRAME_PIXELS \
         --num_train_epochs $NUM_EPOCHS \
         --warmup_ratio $WARMUP_RATIO \
         --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
         --weight_decay $WEIGHT_DECAY \
         --logging_steps $LOGGING_STEPS \
         --save_steps $SAVE_STEPS \
         --save_total_limit $SAVE_TOTAL_LIMIT \
         --deepspeed $DEEPSPEED_CONFIG \

         
