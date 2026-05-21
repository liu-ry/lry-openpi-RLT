# DAgger 逻辑深度解析

本文档聚焦于当前仓库中人为介入（Teleop）机制与 DAgger（Dataset Aggregation）的对应关系，说明为什么**人工接管才是本仓库 DAgger 的真正体现**，并结合代码分析完整流程。

---

## 一、先厘清两件事的本质区别

| 行为 | 是否算 DAgger | 原因 |
|------|-------------|------|
| RL 自己 rollout + 记录 + 训练 | **不算** | 没有"专家在 on-policy 状态上纠正" |
| 人为接管 → 提供纠正动作 → 进 replay → 训练 actor 模仿 | **算** | 人类就是"专家"，在策略出错状态上实时标注正确动作 |

经典 DAgger（Ross et al., 2011）的核心要求：
1. **学生策略 rollout** → 产生 on-policy 状态分布
2. **专家在这些状态上标注** → 得到纠正动作
3. **把 `(state, expert_action)` 加入数据集训练学生** → 纠正 covariate shift

本仓库的人工接管完全满足以上三点，人类扮演"专家"角色。

---

## 二、DAgger 三要素与代码的对应

| DAgger 要素 | 本仓库代码对应 |
|------------|--------------|
| 学生策略 rollout | `PikaChunkEnvAdapter.execute_chunk()` — ChunkActor 正在执行，机器人在运动中 |
| 专家在 on-policy 状态上标注 | 操作员按 `t` 键触发 `toggle_teleop()`，在策略出错的当前状态人工接管 |
| 专家动作 | `_sample_latest_human_action()` — 读取人工遥控的关节指令 |
| 加入数据集 | `source=HUMAN` 的 transition 写入 `ReplayBuffer` |
| 训练学生模仿 | `bc_target = where(human_mask, action_chunk, ...)` — HUMAN 步骤 BC target 是人工动作 |

---

## 三、完整人为介入 DAgger 代码流程

### Step 1：人工判断策略出错，按 `t` 接管

**文件：** `rlt_online_rl/keyboard_toggle_teleop_record_reward_isolation.py`

```python
def toggle_teleop(self):
    # 按 't' 键触发 ROS service 调用
    self._toggle_local_teleop()     # 通知 HumanInterventionState: policy_enabled=False
    self._toggle_hardware_teleop()  # 硬件端切换为人工遥控模式
```

**文件：** `rlt_online_rl/train_deploy_alignment/pika_sync_ros.py` — `HumanInterventionState.toggle_policy()`

```python
def toggle_policy(self, *, resume_delay_s: float):
    self._policy_enabled = not self._policy_enabled   # 翻转策略/人工控制标志
    if self._policy_enabled:
        self._resume_until = time.time() + resume_delay_s  # 恢复有冷却期，防止抖动
```

---

### Step 2：`execute_chunk` 检测到 `policy_enabled=False`，转为读取人工动作

**文件：** `rlt_online_rl/train_deploy_alignment/pika_sync_ros.py` — `PikaChunkEnvAdapter.execute_chunk()`

```python
for local_step in range(horizon):
    policy_enabled = bool(
        self._intervention_state.is_policy_enabled()
        and not self._intervention_state.in_resume_cooldown()
    )

    if policy_enabled and current_plan is not None:
        # 策略模式：执行 ChunkActor 输出的动作
        raw_action = current_plan.action_chunk[plan_cursor]
        step_sources.append(int(TransitionSource.RL))
        human_controlled.append(False)
    else:
        # ← 人工介入模式：读取人工遥控动作（就在策略出错的当前状态上！）
        human_action = self._sample_latest_human_action(step_observation)
        executed.append(human_action)
        ref_actions.append(human_action.copy())  # ref_action 也记为人工动作
        step_sources.append(int(TransitionSource.HUMAN))
        human_controlled.append(True)            # 标记此步为人工控制
```

> **关键设计**：人工动作同时被记录为 `action`（实际执行）和 `ref_action`，保证后续 BC target 能正确选取。

---

### Step 3：Episode 结束，构建 Transition，`source=HUMAN/MIXED` 写入 Replay

```python
# pika_sync_ros.py: execute_chunk() 末尾
if human_intervened:
    source = int(
        TransitionSource.MIXED    # 同一 chunk 中既有人工步也有策略步
        if any(not flag for flag in human_controlled)
        else TransitionSource.HUMAN  # 整个 chunk 全是人工步
    )

# 每步都携带 source 和 human_controlled 标记
step_trace = [
    {
        "action":          executed[idx],          # 人工动作 or 策略动作
        "ref_action":      ref_actions[idx],        # 人工步=人工动作；策略步=VLA参考
        "source":          step_sources[idx],       # HUMAN / RL / BASE
        "human_controlled": human_controlled[idx],
        ...
    }
    for idx in range(len(executed))
]
```

最终经由 `EnvDriver._append_raw_chunk` → `ReplayClient.add_transitions` 写入 Replay Buffer。

---

### Step 4：Learner 采样到 HUMAN 数据，用人工动作做 BC Target 训练 actor

**文件：** `rlt_online_rl/src/rlt_online_rl/trainer.py` — `update_actor()` → `loss_fn()`

```python
source_chunk = batch["source_chunk"]  # 每步的 source 标记 (HUMAN/RL/BASE/MIXED)

# HUMAN 或 MIXED 步骤 → bc_target = 人工实际动作（DAgger 监督信号）
# BASE 或 RL 步骤    → bc_target = VLA 参考动作（约束 actor 别跑偏）
human_mask = (source_chunk == HUMAN) | (source_chunk == MIXED)
bc_target = where(
    human_mask[..., None],
    batch["action_chunk"],   # ← 人工演示的动作（专家标签）
    batch["ref_chunk"],      # ← VLA 锚点（约束正则）
)

bc_penalty = MSE(action_chunk, bc_target)
actor_loss = bc_weight * bc_penalty   # DAgger 监督部分
           - q_weight  * actor_q      # RL 优化部分
           + delta_weight * delta     # 平滑正则
```

---

## 四、与经典 DAgger 的精确对应

```
经典 DAgger                            本仓库人为介入流程
──────────────────────────────────────────────────────────────────────
学生策略 rollout，访问状态 s_t          ChunkActor 在机器人上执行，产生真实 on-policy 状态分布
         ↓                                          ↓
策略出错 / 需要纠正                       操作员观察到机器人行为异常，按 't' 键
         ↓                                          ↓
专家在 s_t 上标注 a* = π*(s_t)           人工遥控机器人，执行 human_action（就在出错状态上）
         ↓                                          ↓
将 (s_t, a*) 加入聚合数据集 D            step_trace[source=HUMAN] → ReplayBuffer
         ↓                                          ↓
在 D 上监督学习：min MSE(π(s), a*)        bc_target = action_chunk（when human_mask）
                                          actor_loss += bc_weight * MSE(actor_output, human_action)
```

---

## 五、比经典 DAgger 更细粒度的设计：步级 MIXED 标记

经典 DAgger 通常以 episode 为粒度切换策略/专家。本仓库实现了**步级（step-level）的混合标记**：

同一个 chunk 内可以有人工步和策略步混合（`MIXED`），BC target 会**逐步**区分：

```
Chunk: [策略步0, 策略步1, 人工步2, 人工步3, 人工步4]
source: [  RL,     RL,    HUMAN,  HUMAN,  HUMAN ]
          ↓                  ↓
  bc_target=ref_chunk    bc_target=human_action
```

- **策略步**：学习"不偏离 VLA 太远"（约束稳定性）
- **人工步**：学习"从策略出错的这个状态开始，正确动作是什么"（DAgger 纠正信号）

这比经典 DAgger 用整 episode 替换更细粒度，充分利用了人工介入数据。

---

## 六、关键文件索引

| 功能 | 文件 | 核心类/函数 |
|------|------|-----------|
| 键盘触发人工接管 | `rlt_online_rl/keyboard_toggle_teleop_record_reward_isolation.py` | `KeyboardTeleopRecordRewardToggle.toggle_teleop()` |
| 人工/策略状态管理 | `rlt_online_rl/train_deploy_alignment/pika_sync_ros.py` | `HumanInterventionState` |
| ROS 接管服务 | `rlt_online_rl/train_deploy_alignment/pika_sync_ros.py` | `TeleopTriggerNode` |
| 每步读取人工动作 | `rlt_online_rl/train_deploy_alignment/pika_sync_ros.py` | `PikaChunkEnvAdapter.execute_chunk()` |
| 动作/source 标记 | `rlt_online_rl/src/rlt_online_rl/replay.py` | `TransitionSource`, `RLTTransition` |
| DAgger BC Loss | `rlt_online_rl/src/rlt_online_rl/trainer.py` | `update_actor()` → `bc_target` |

---

## 七、总结

**纯 RL 自迭代不算 DAgger**；**人工介入才是本系统 DAgger 的真正体现**：

> 人类作为专家，在学生策略（ChunkActor）on-policy 访问的出错状态上，通过遥控操作实时标注纠正动作；这些数据以 `source=HUMAN` 标记存入 Replay Buffer，Learner 采样后以人工动作为 BC target 训练 actor，完全符合 DAgger 的核心定义。

在此基础上叠加 RL Q-value 优化项，允许 actor 在人类纠正的方向上进一步超越人类示范，构成 **"DAgger + Actor-Critic RL"** 的混合训练范式。
