# RLT Stage 1 训练：在 pi0.5 上附加 RL-Token 模块

本文档描述如何在已有的 pi0.5 checkpoint 基础上，通过 Stage 1 训练得到带 RL-Token 编解码模块的 RLT 模型，供 Machine A 在在线 RL 阶段使用。

---

## 一、为什么需要 Stage 1 训练

原始开源的 pi0.5 模型没有 RL-Token 的编解码部分。在线 RL 系统（Machine A）需要在每次推理时同时返回：

- `ref_chunk`：VLA 生成的参考动作块
- `z_rl`：紧凑的 RL-Token 特征向量（2048 维）

`z_rl` 由 `RLTokenEncoder` 将 VLA 的 prefix embeddings 压缩而来，因此需要训练这个新增的编解码模块。

**关键点：pi0.5 本体默认完全冻结，不需要重新训练。** Stage 1 只训练新增的轻量 RL-Token 模块。

---

## 二、整体两阶段流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1：训练 RL-Token 编解码模块（本文档）                     │
│                                                                  │
│  输入：已有的 pi0.5 / pi0.5-finetuned checkpoint                 │
│  训练：只更新 RLTokenEncoder + RLTokenDecoder（新增的小模块）     │
│  VLA 主体：完全冻结（alpha=0，默认配置）                          │
│  损失：RL-Token 重建 VLA prefix embeddings 的 MSE（自编码器损失） │
└─────────────────────────────┬───────────────────────────────────┘
                              │ 产出：带 rlt_module 的 checkpoint
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  Stage 2：在线 RL（rlt_online_rl 系统，见 online_rl_pipeline.md）│
│  Machine A 加载 Stage 1 checkpoint                               │
│  推理时同时输出 z_rl + ref_chunk                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、RL-Token 模块架构

### 3.1 整体结构

`RLTTrainModel`（训练时）和 `RLTInferenceModel`（推理时）的组合结构：

```
RLTTrainModel / RLTInferenceModel
├── self.vla           ← 原始 pi0.5 模型（Stage 1 中完全冻结）
└── self.rlt_module    ← 新增的轻量 RL-Token 编解码器（只训练这个）
    ├── RLTokenEncoder   VLA prefix embeddings → RL Token
    └── RLTokenDecoder   RL Token → 重建 prefix embeddings
```

### 3.2 RLTokenEncoder

**文件：** `src/openpi/models/rl_token.py`

将 VLA prefix embeddings 压缩为少量 RL-Token：

```
VLA prefix embeddings [B, seq_len, 2048]
        │
        │  （可选 input_proj 线性投影：2048 → embed_dim=512）
        │
        ▼  num_layers=2 层 Cross-Attention
        │  query: 可学习的 RL-Token 位置编码 [num_rl_tokens, 512]
        │  key/value: prefix embeddings + 位置编码
        │
        ▼
RL Token  [B, num_rl_tokens=1, embed_dim=512]
```

默认配置（`RLTokenConfig`）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_rl_tokens` | 1 | 压缩后 RL-Token 数量 |
| `num_layers` | 2 | Cross-Attention 层数 |
| `embed_dim` | 512 | RL-Token 内部维度 |
| `input_dim` | 2048 | VLA prefix embedding 维度（Gemma 2B hidden size）|
| `num_heads` | 8 | 注意力头数 |
| `dropout_rate` | 0.0 | Dropout 概率 |

### 3.3 RLTokenDecoder

将 RL-Token 重建回 prefix embeddings（用于计算重建损失）：

```
RL Token [B, num_rl_tokens, 512]
        │
        ▼  num_layers=2 层 Cross-Attention
        │  query: 目标序列位置编码 [seq_len, 512]
        │  key/value: RL-Token + 位置编码
        │
        │  （output_proj 线性投影：512 → 2048）
        │
        ▼
重建 prefix embeddings [B, seq_len, 2048]
```

### 3.4 推理时的 z_rl

推理服务（`serve_rlt_policy.py`）中，`z_rl` 由 RL-Token flatten 得到：

```python
# RL Token: [B, num_rl_tokens=1, embed_dim=512]
# → flatten → z_rl: [B, num_rl_tokens * embed_dim] = [B, 512]
# 但实际 z_dim 配置为 2048，因此会有 output_proj 或配置匹配
```

实际 `z_dim=2048` 的情况下，`embed_dim` 设为 2048 或通过 `input_dim=embed_dim=2048` 直接输出 2048 维。

---

## 四、Stage 1 训练损失

### 4.1 核心损失：RL-Token 重建损失（Lro）

**文件：** `scripts/train_rlt.py` — `RLTTrainModel.compute_rlt_loss`

```python
def compute_rlt_loss(self, rng, observation, actions, alpha, *, train=False):
    # Step 1: 用冻结的 VLA 提取 prefix embeddings（图像+语言 Transformer 中间特征）
    prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(
        rng, observation, train=False, image_only=True
    )

    # Step 2: stop_gradient 确保梯度不流回 VLA
    prefix_embs_sg = jax.lax.stop_gradient(prefix_embs)
    prefix_embs_f32 = prefix_embs_sg.astype(jnp.float32)

    # Step 3: encode → decode → MSE 重建损失
    rlt_loss, info = self.rlt_module(prefix_embs_f32, mask=None, train=train)
    # rlt_loss = MSE(decoder(encoder(prefix_embs)), prefix_embs)
```

$$\mathcal{L}_{\text{ro}} = \text{MSE}\left(\text{Decoder}\left(\text{Encoder}(\text{prefix\_embs})\right),\ \text{stop\_gradient}(\text{prefix\_embs})\right)$$

`stop_gradient` 保证重建损失的梯度**只流向 RLT 模块**，不影响 VLA 权重。

### 4.2 可选：联合训练（alpha > 0）

设置 `--rlt_alpha 0.1` 时，VLA 也参与训练：

```python
if alpha > 0.0:
    # 联合前向：同时计算 VLA diffusion loss 和 RLT 重建 loss
    vla_loss, prefix_embs, prefix_mask = self.vla.compute_loss_with_prefix(
        rng, observation, actions, train=True, image_only=True
    )
    total_loss = rlt_loss + alpha * vla_loss
else:
    # 默认：完全冻结 VLA，total_loss = rlt_loss
    total_loss = rlt_loss
```

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ro}} + \alpha \cdot \mathcal{L}_{\text{VLA}}$$

| 模式 | alpha | 可训练参数 | 适用场景 |
|------|-------|-----------|---------|
| **冻结 VLA（默认）** | 0.0 | 仅 `rlt_module` | 快速 Stage 1，计算代价小 |
| **联合训练** | > 0.0 | `rlt_module` + 全部 `vla` | 需要 VLA 适应新任务时 |

---

## 五、权重加载机制

Stage 1 训练会先加载已有的 pi0.5 checkpoint，再初始化新增的 `rlt_module`：

**文件：** `scripts/train_rlt.py` — `_load_vla_weights`

```python
def _load_vla_weights(loader, full_params_shape):
    # full_params_shape 中的参数树结构：
    #   vla/PaliGemma/...    ← 来自 pi0.5 checkpoint
    #   rlt_module/...       ← 新增，随机初始化
    #
    # 从 pi0.5 checkpoint 加载 vla/ 子树，rlt_module/ 保持随机初始化
    vla_shape = {k[1:]: v for k, v in flat_shape.items() if k[0] == "vla"}
    loaded_vla = loader.load(vla_shape)
    reprefixed = {("vla",) + k: v for k, v in flat_loaded.items()}
    return reprefixed
```

加载完成后，VLA 权重转为 `bfloat16` 并冻结（不加入优化器状态），只有 `rlt_module` 保持 float32 并参与梯度更新。

---

## 六、训练步骤

### 6.1 环境准备

```bash
# 在仓库根目录，使用 openpi 主环境（非 rlt_online_rl310）
conda activate <openpi_env>
```

### 6.2 计算 norm stats（如果使用新数据集）

```bash
python scripts/compute_norm_stats.py --config <your_config>
```

### 6.3 启动 Stage 1 训练

```bash
python scripts/train_rlt.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --exp-name my_rlt_stage1 \
  --checkpoint-dir checkpoints/rlt_stage1
```

常用额外参数：

```bash
# 联合训练（同时微调 VLA）
--rlt_alpha 0.1

# 自定义 RL-Token 数量（默认 1）
--rlt_num_tokens 1

# 自定义 RL-Token 内部维度
--rlt_embed_dim 512

# 自定义 Cross-Attention 层数
--rlt_num_layers 2

# 启用 W&B 记录
--wandb_enabled true
```

### 6.4 训练输出

每隔 `save_interval` 步保存一次 checkpoint，结构如下：

```
checkpoints/rlt_stage1/
├── <step>/
│   ├── params/
│   │   ├── vla/         ← VLA 权重（冻结，来自 pi0.5）
│   │   └── rlt_module/  ← 训练后的 RL-Token 编解码模块
│   └── ...
└── wandb_id.txt
```

W&B 记录的指标：

| 指标 | 含义 |
|------|------|
| `loss` | 总损失 |
| `rlt_loss` | RL-Token 重建损失（Lro）|
| `mse` | prefix embeddings 重建 MSE |
| `vla_loss` | VLA diffusion loss（仅 alpha > 0 时有）|
| `grad_norm` | 梯度范数 |
| `param_norm` | 参数范数 |

---

## 七、Stage 1 完成后：启动 Machine A 推理服务

Stage 1 checkpoint 产出后，即可用于 Machine A：

```bash
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir checkpoints/rlt_stage1/<step> \
  --port 8000 \
  --shared-prefix-inference
```

推理服务 `RLTInferenceModel.infer` 会同时返回：

```python
def infer(self, rng, observation):
    # 1. VLA 提取 prefix embeddings
    prefix_embs, _ = self.vla.extract_prefix_embeddings(rng, observation, image_only=True)

    # 2. RLTokenEncoder 压缩为 z_rl
    rl_token = self.rlt_module.encode(prefix_embs.astype(jnp.float32))
    z_rl = rl_token.reshape(batch_size, -1)  # flatten → [B, z_dim]

    # 3. VLA 生成 ref_chunk（使用 shared prefix 或独立前向）
    actions = self.vla.sample_actions(rng, observation, ...)

    return actions, z_rl  # → ref_chunk, z_rl
```

---

## 八、训练代价估算

| 组件 | 参数量 | 是否训练 |
|------|--------|---------|
| pi0.5 VLA（PaliGemma + Flow Matching）| ~3B | ❌ 冻结 |
| RLTokenEncoder（2层 Cross-Attention）| ~数 M | ✅ 训练 |
| RLTokenDecoder（2层 Cross-Attention）| ~数 M | ✅ 训练 |

由于 VLA 冻结，**Stage 1 的训练代价远小于 pi0.5 本身的微调**，主要计算开销在于 VLA 的前向传播（提取 prefix embeddings）以及小模块的梯度更新。

---

## 九、关键代码文件索引

| 文件 | 作用 |
|------|------|
| `src/openpi/models/rl_token.py` | `RLTokenEncoder`, `RLTokenDecoder`, `RLTokenModel` 定义 |
| `scripts/train_rlt.py` | Stage 1 训练入口，`RLTTrainModel`, `compute_rlt_loss`, `train_step` |
| `scripts/serve_rlt_policy.py` | Stage 1 checkpoint 推理服务，`RLTInferenceModel` |
| `src/openpi/training/config.py` | 训练配置，含 `rlt_num_tokens`, `rlt_alpha` 等字段 |
| `src/openpi/models/utils/fsq_tokenizer.py` | `CrossAttentionLayer` 实现 |
