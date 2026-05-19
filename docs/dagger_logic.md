# DAgger 逻辑深度解析

本文档聚焦于当前仓库中 DAgger（Dataset Aggregation）的具体实现方式，说明其与经典 DAgger 的对应关系，以及在代码中的体现。

---

## 一、经典 DAgger 回顾

原始 DAgger（Ross et al., 2011）的核心思想：

```
for each iteration:
    1. 用当前策略 π_t 在环境中 rollout，收集状态分布 d_t
    2. 在 d_t 中的每个状态 s，查询专家策略 π* 得到专家动作 a* = π*(s)
    3. 将 (s, a*) 加入聚合数据集 D
    4. 在 D 上做监督学习，得到 π_{t+1}
```

关键点：**训练分布由学生策略决定（on-policy）+ 监督标签由专家提供**，从而解决 BC 的 covariate shift 问题。

---

## 二、本仓库的 DAgger 实现

### 2.1 角色映射

| 经典 DAgger | 本仓库实现 |
|-------------|------------|
| 专家策略 π* | 冻结 VLA/RLT 模型（Machine A，`serve_rlt_policy.py`） |
| 学生策略 π_t | `ChunkActor`（Machine B，`actor_service`） |
| 专家动作查询 | Machine A 在每个 chunk 边界返回 `ref_chunk` |
| 聚合数据集 D | `replay_manager` 中的 replay buffer + journal |
| 监督学习更新 | `learner_service` 中 actor 的 BC loss（`bc_penalty`）|
| On-policy rollout | `EnvDriver` 用当前 actor 在机器人上执行动作 |

### 2.2 "Chunk 级 DAgger"结构

本仓库将 DAgger 扩展到 **chunk 级动作**（而不是单步动作）：

每个 chunk 边界的流程：

```
当前观测 obs_t
    │
    ├──→ Machine A（专家）
    │       输入: obs_t（图像、关节状态）
    │       输出: ref_chunk [chunk_len × action_dim]   ← 专家动作
    │             z_rl [z_dim]                          ← 压缩特征
    │
    └──→ Machine B actor（学生）
            输入: z_rl, proprio, ref_chunk
            输出: refined_chunk [chunk_len × action_dim]  ← 学生动作
                │
                ▼
          机器人执行 refined_chunk（On-policy）
                │
                ▼
          记录 (obs_t, z_rl, proprio, ref_chunk, refined_chunk, reward)
```

**与经典 DAgger 的对应**：
- `ref_chunk` = 专家动作标签 a*（在学生所访问的状态上实时查询）
- `refined_chunk` = 学生实际执行的动作（on-policy distribution）
- BC loss 训练学生输出趋近 `ref_chunk`

---

## 三、代码级实现细节

### 3.1 专家查询：Machine A Feature Client

**文件：** `rlt_online_rl/src/rlt_online_rl/inference.py` — `MachineAFeatureClient`

```python
class MachineAFeatureClient:
    def get_features(self, observation: dict[str, Any]) -> dict[str, Any]:
        # 向 Machine A 发送观测，获取 {z_rl, ref_chunk}
        return self._infer(observation)
```

每次 chunk 边界，`EnvDriver` 调用此方法，即在**学生策略当前访问的状态**上查询专家，这是 DAgger 的核心操作。

### 3.2 学生策略执行：ChunkActor

**文件：** `rlt_online_rl/src/rlt_online_rl/networks.py` — `ChunkActor`

```python
@dataclasses.dataclass(frozen=True)
class ChunkActor:
    z_dim: int        # 2048
    proprio_dim: int  # 7
    chunk_len: int    # 10
    action_dim: int   # 7
    hidden_dim: int
    num_layers: int
    fixed_std: float  # 0.05

    def _encode_inputs(self, params, z_rl, proprio, ref_chunk):
        # ref_chunk 被投影后与 z_rl、proprio 特征拼接
        # 学生的输入包含专家参考，这是"有条件的 DAgger"
        ref_flat = ref_chunk.reshape(batch_size, chunk_len * action_dim)
        z_feat    = layer_norm(z_rl    @ z_proj)
        proprio_feat = tanh(layer_norm(proprio @ proprio_proj))
        ref_feat  = tanh(layer_norm(ref_flat @ ref_proj))
        return concat([z_feat, proprio_feat, ref_feat])  # 256+64+256=576 维

    def actor_mean(self, params, z_rl, proprio, ref_chunk):
        # 输出精炼后的 action chunk
        features = self._encode_inputs(params, z_rl, proprio, ref_chunk)
        mu = mlp_forward(params["trunk"], features)
        return mu.reshape(batch_size, chunk_len, action_dim)
```

**关键设计**：学生 actor 接收专家 `ref_chunk` 作为**条件输入**，学习如何"编辑"专家动作。这不同于经典 DAgger（学生不看专家动作），而是一种**条件 DAgger**或 **Reference-Conditioned Policy**。

### 3.3 Reference Dropout：防止对专家的过度依赖

**文件：** `rlt_online_rl/src/rlt_online_rl/networks.py` — `apply_reference_dropout`

训练时以概率 `reference_dropout_prob`（默认 0.5）将 `ref_chunk` 置零：

```python
def apply_reference_dropout(
    rng: jax.Array,
    ref_chunk: jax.Array,
    dropout_prob: float,
) -> jax.Array:
    # 以 dropout_prob 的概率将整个 ref_chunk 归零
    # 防止学生策略在没有专家输入时完全失效
```

**Trainer 中的调用：**

```python
def loss_fn(actor_params):
    dropout_rng, sample_rng = jax.random.split(actor_rng)
    dropped_ref = apply_reference_dropout(
        dropout_rng,
        batch["ref_chunk"],
        rl_config.reference_dropout_prob,   # 0.5
    )
    action_chunk = actor.sample_action(
        actor_params, sample_rng,
        batch["z_rl"], batch["proprio"], dropped_ref,  # 使用 dropout 后的 ref
    )
```

这确保了学生在**没有专家指导时**也能独立工作，是 DAgger 向自主策略泛化的重要机制。

### 3.4 BC Loss：DAgger 的监督学习部分

**文件：** `rlt_online_rl/src/rlt_online_rl/trainer.py` — `update_actor`

```python
# BC target 根据 source_chunk 动态选择：
bc_target = jnp.where(
    human_mask[..., None],
    batch["action_chunk"],   # HUMAN/MIXED 步骤 → 人工演示动作
    batch["ref_chunk"],      # BASE/RL 步骤 → 专家 VLA 参考动作（DAgger 标签）
)

# BC loss = 学生动作与 BC target 之间的 MSE
bc_error = jnp.mean(jnp.square(action_chunk - bc_target), axis=-1)
bc_penalty = jnp.mean(bc_error)
```

**这正是 DAgger 的监督学习步骤**：
- 训练样本的状态分布来自**学生策略**（on-policy，因为机器人执行的是 actor 的输出）
- 训练标签（`ref_chunk`）来自**专家策略**（Machine A，在相同状态上实时查询）

### 3.5 TransitionSource：追踪数据来源

**文件：** `rlt_online_rl/src/rlt_online_rl/replay.py`

```python
class TransitionSource(enum.IntEnum):
    BASE  = 0   # 冻结 VLA 执行（warmup 或 non-critical 段）
    RL    = 1   # ChunkActor 执行（online 关键段）
    HUMAN = 2   # 人工接管
    MIXED = 3   # 同一 window 中混合了人工和策略步骤
```

**BC target 选择逻辑**（每步粒度，基于 `source_chunk`）：

```
source_chunk[t] == HUMAN 或 MIXED  →  bc_target[t] = action_chunk[t]（人工动作）
source_chunk[t] == BASE 或 RL      →  bc_target[t] = ref_chunk[t]（VLA 专家动作）
```

这一设计意味着：人工干预数据教会 actor 如何**将 VLA 参考修正为人工执行**；BASE/RL 数据教会 actor 如何**模仿 VLA 专家**。两种数据都在 DAgger 框架下被合理利用。

### 3.6 超出纯 DAgger：增加 RL 信号

本仓库在 DAgger 基础上增加了 RL 信号，完整的 actor loss 为：

$$\mathcal{L}_{\text{actor}} = \underbrace{\lambda_{\text{bc}} \cdot \mathcal{L}_{\text{bc}}}_{\text{DAgger BC 项}} - \underbrace{\lambda_q \cdot \mathbb{E}[Q(s, \pi(s))]}_{\text{RL Q 值最大化项}} + \underbrace{\lambda_\delta \cdot \mathcal{L}_{\delta}}_{\text{平滑正则项}}$$

```python
actor_loss = (
    bc_weight    * bc_penalty      # DAgger 部分
    - q_weight   * actor_q         # RL 部分：最大化 Q 值
    + delta_weight * delta_penalty  # 正则：step-to-step 平滑
)
```

- **Warmup 阶段**（`global_step < warmup_required_updates`）：使用 `warmup_bc_weight / warmup_q_weight`，通常 BC 权重更高，类似纯 DAgger
- **Online 阶段**：使用 `online_bc_weight / online_q_weight`，RL 信号加入，指导超越专家的行为

### 3.7 Replay Buffer：DAgger 的聚合数据集

**文件：** `rlt_online_rl/src/rlt_online_rl/replay.py` — `ReplayManager`

```python
class ReplayManager:
    capacity: int = 200_000       # 聚合数据集最大容量
    journal_path: str = ...       # append-only pickle 日志，对应 DAgger 的持续聚合
    sample_strategy: str = "uniform" | "stratified"
```

Replay buffer 对应 DAgger 中的**聚合数据集 D**，持续累积所有历史 episodes 的 transitions（包括 warmup 演示和 online 数据），并支持分层采样（`stratified`）来平衡新旧数据和人工干预数据的比例。

---

## 四、Warmup 作为 DAgger 第一轮迭代

Warmup 阶段本质上是 **DAgger 的初始化轮**（类似 BC 预训练）：

```
Warmup:
  π_0 = ref_chunk（专家策略直接执行）
  在专家策略的状态分布下收集数据
  D_0 = {(s, ref_chunk(s))} for s in warmup episodes
  在 D_0 上训练 → 得到初始 ChunkActor π_1
```

当 `replay_size >= warmup_min_size` 且 warmup 训练步数满足后，切换到 Online 阶段（DAgger 迭代）。

---

## 五、与经典 DAgger 的差异总结

| 特性 | 经典 DAgger | 本仓库实现 |
|------|-------------|------------|
| 动作粒度 | 单步动作 | Chunk（10步）|
| 学生输入 | 仅状态 s | 状态 + 专家参考（条件化）|
| 专家标签提供方式 | 离线/在线查询专家 | Machine A 实时 WebSocket 查询 |
| 数据聚合 | 每轮替换数据集 | Replay buffer 持续累积（含 journal）|
| 学习目标 | 纯 BC（MSE on expert actions）| BC + RL（Q 值最大化）+ 平滑正则 |
| 人工干预 | 不支持 | 支持 teleop 接管，HUMAN 步骤 BC target = 人工动作 |
| Dropout | 无 | `reference_dropout_prob=0.5`，防止依赖专家输入 |
| 专家类型 | 通常是人类或规划器 | 冻结的大型 VLA 扩散模型 |

---

## 六、关键代码文件索引

| 文件 | 作用 |
|------|------|
| `src/rlt_online_rl/inference.py` | `MachineAFeatureClient`（专家查询），`EnvDriver`（rollout 驱动），`ActorService`（学生推理服务） |
| `src/rlt_online_rl/trainer.py` | `update_actor`（BC+RL loss），`LearnerService`（训练循环），`train_step`（JIT 编译训练步） |
| `src/rlt_online_rl/networks.py` | `ChunkActor`（学生策略网络），`TwinCritic`（Critic），`apply_reference_dropout` |
| `src/rlt_online_rl/replay.py` | `ReplayManager`（聚合数据集），`RLTTransition`（transition 数据结构），`TransitionSource` |
| `src/rlt_online_rl/config.py` | `RLTOnlineRLConfig`（包含 `reference_dropout_prob`, `bc_weight` 等关键 DAgger 超参数）|
| `scripts/serve_rlt_policy.py` | Machine A 专家服务入口 |
| `scripts/run_online_rl.py` | Machine B 完整训练循环入口 |
