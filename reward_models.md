# LeRobot 奖励模型对比与用法

## 四种奖励模型一览

| | SARM | TOPReward | Robometer | Reward Classifier |
|---|---|---|---|---|
| 注册名 | `sarm` | `topreward` | `robometer` | `reward_classifier` |
| 性质 | 可训练 | 零样本 | 预训练即用 | 可训练 |
| 骨干 | CLIP ViT-B/32 + 自有 Transformer | Qwen3-VL-8B（冻结） | Qwen3-VL-4B + 进度/成功头 | ResNet10 / Transformer |
| 输出 | progress ∈ [0,1]（阶段+进度） | log P("True"\|video+task) | progress ∈ [0,1] 或 success ∈ {0,1} | success ∈ {0,1} |
| 需要训练 | 是 | 否 | 否（用 `lerobot/Robometer-4B`） | 是 |
| 监督数据 | 任务文本+视频帧（single_stage 零标注；dense/dual 需 VLM 标子任务） | 无 | 无 | 人类标注的成功/失败帧 |
| 用途 | RA-BC 后训练、离线打分 | RA-BC 打分、零样本奖励 | RA-BC 打分、成功判定 | HIL-SERL 在线 RL 的成功奖励 |
| 安装 extra | `[sarm]` | `[topreward]` | `[robometer]` | `[hilserl]` |
| GPU 需求 | 低（CLIP + 小 Transformer） | 高（~16GB，8B VLM） | 中（~8GB，4B VLM） | 低 |

## 优缺点

### SARM
- 优点：轻量、可针对自己任务定制；阶段感知，长时序任务能区分"在哪个子任务+进度多少"；single_stage 模式零标注即可训
- 缺点：需要训练步骤；dense/dual 模式要跑 VLM 标注；奖励质量取决于训练数据量
- 适合：有自己数据集、想要高质量进度奖励、任务有多阶段

### TOPReward
- 优点：完全零样本，拿来即用；不挑任务，只要能给自然语言任务描述
- 缺点：8B VLM 推理慢、显存大；只有 Qwen3-VL 一个 client；返回的是 log-prob，尺度需要自己调 `success_threshold`
- 适合：快速验证、不想训 RM、任务能用一句话描述

### Robometer
- 优点：预训练好的 4B 模型直接用，比 TOPReward 轻；同时给 progress 和 success 两个输出；离散进度桶可选
- 缺点：依赖特定预训练权重 `lerobot/Robometer-4B`；Qwen3-VL-4B backbone 仍需 ~8GB 显存
- 适合：想要 progress 奖励但不想自己训、显存有限

### Reward Classifier
- 优点：最轻量、训练快；二分类明确
- 缺点：只判成功/失败，没有 dense progress；需要人类标注的成功/失败帧；泛化性弱
- 适合：HIL-SERL 在线 RL（要一个能在线判成功的函数）、简单短任务

## 用法

### 通用加载方式

```python
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors

cfg = make_reward_model_config("sarm", device="cuda")  # 换成 topreward/robometer/reward_classifier
model = make_reward_model(cfg)
preprocessor, postprocessor = make_reward_pre_post_processors(cfg)
```

### SARM：训练 + 打分 + RA-BC

```bash
# 1. 训 SARM（single_stage 零标注）
lerobot-train \
  --dataset.repo_id=your-user/your-dataset \
  --reward_model.type=sarm \
  --reward_model.annotation_mode=single_stage \
  --reward_model.image_key=observation.images.top \
  --output_dir=outputs/sarm --batch_size=32 --steps=5000

# dense/dual 模式先跑 VLM 标注：
# python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
#   --repo-id your-user/your-dataset \
#   --dense-subtasks "step1,step2,step3" \
#   --video-key observation.images.top --push-to-hub

# 2. 给数据集每帧打进度分
python -m lerobot.rewards.sarm.compute_rabc_weights \
  --dataset-repo-id your-user/your-dataset \
  --reward-model-path outputs/sarm/checkpoints/last \
  --head-mode sparse --device cuda

# 3. RA-BC 后训练 policy（支持 pi0 / pi05 / smolvla）
lerobot-train \
  --dataset.repo_id=your-user/your-dataset \
  --policy.type=pi0 \
  --sample_weighting.type=rabc \
  --sample_weighting.head_mode=sparse \
  --sample_weighting.kappa=0.01 \
  --output_dir=outputs/pi0_rabc
```

### TOPReward：零样本打分

```bash
# 安装
uv sync --extra topreward

# 离线给数据集打分（生成 topreward_progress.parquet，schema 与 SARM 兼容）
uv run python -m lerobot.rewards.topreward.compute_rabc_weights \
  --dataset-repo-id lerobot/libero_10_image \
  --num-samples 15 --device cuda

# 之后同样用 --sample_weighting.type=rabc 跑 RA-BC
```

### Robometer：预训练即用

```python
from lerobot.rewards import make_reward_model, make_reward_model_config

cfg = make_reward_model_config(
    "robometer",
    pretrained_path="lerobot/Robometer-4B",
    reward_output="progress",  # 或 "success"
    device="cuda",
)
model = make_reward_model(cfg)
# model.compute_reward(batch) -> Tensor[B]
```

### Reward Classifier：训练 + HIL-SERL

```bash
# 训练（需要带成功/失败标签的数据）
lerobot-train \
  --dataset.repo_id=your-user/your-labeled-dataset \
  --policy.type=reward_classifier \
  --output_dir=outputs/reward_classifier

# 在 HIL-SERL RL 训练里作为成功检测器使用（见 hilserl.mdx）
```

## 怎么选

- 有数据、要质量、任务多阶段 → **SARM**
- 没数据、要快、一句话能描述任务 → **TOPReward**
- 没数据、显存有限、想要 progress → **Robometer**
- 在线 RL、只要成功/失败信号 → **Reward Classifier**
