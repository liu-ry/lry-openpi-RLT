# Q 值与 Twin Critic 深度解析

---

## 一、本项目的 Q 值是什么？

本项目采用的是 **纯 Q 函数（动作价值函数）**，对应经典 TD3/SAC 中的 Twin Q-network 结构，**不是** A2C/A3C 那种 "动作价值 + 状态价值" 的拆分设计。

### 1.1 Q 函数的输入与输出

```
输入：
  z_rl        (RL token，状态特征)
  proprio     (本体感知)
  action_chunk（完整动作序列）

                     ↓ QNetwork

输出：
  标量 Q(s, a)  ← 在状态 s 下执行动作 a 的预期累积回报
```

**Q 值的直觉含义**：执行完这个 action chunk 之后，未来能拿到多少总奖励？

---

## 二、什么是 Actor-Critic 里的 "动作价值 + 状态价值"？

这是另一类 AC 算法（如 A2C、A3C、PPO）中的设计，与本项目**无关**，但容易混淆，特此对比。

### 2.1 两个函数的定义

| 函数 | 符号 | 含义 |
|------|------|------|
| **状态价值函数** | $V(s)$ | 在状态 $s$ 下，按当前策略执行，预期总回报 |
| **动作价值函数** | $Q(s, a)$ | 在状态 $s$ 下执行动作 $a$，再按当前策略执行，预期总回报 |

### 2.2 优势函数 A(s, a)

这类算法的核心是将两者相减，得到 **优势函数（Advantage Function）**：

$$A(s, a) = Q(s, a) - V(s)$$

**直觉**：$A(s, a) > 0$ 说明动作 $a$ 比"按策略平均"更好；$A(s, a) < 0$ 说明更差。

### 2.3 网络结构示意

```
观测 s
  ↓
共享骨干网络
  ├─→ V head → 标量 V(s)       ← 状态价值
  └─→ Policy head → π(a|s)    ← 策略
```

Actor loss 用 $A(s,a)$ 来加权策略梯度，**Critic 只需要学 $V(s)$**，不需要 action 作为输入。

---

## 三、什么是 TD3/SAC 里的 Twin Q-network？

这是本项目实际使用的结构。

### 3.1 为什么需要 Twin（双网络）？

单个 Q 网络存在**过估计问题（Overestimation Bias）**：

> 用 $\max_a Q(s', a)$ 计算 TD target 时，Q 网络会系统性地高估 Q 值，导致 Actor 被错误引导去执行实际上并不好的动作。

**Twin Q-network 的解法（Clipped Double Q）**：
- 训练两个完全独立的 Q 网络：Q1 和 Q2
- 计算 TD target 时取 $\min(Q1_{target}, Q2_{target})$，主动保守估计

$$y_{TD} = r + \gamma \cdot (1 - \text{done}) \cdot \min\bigl(Q1_{target}(s', a'), Q2_{target}(s', a')\bigr)$$

### 3.2 本项目的 TwinCritic 结构

```
TwinCritic
├── Q1 (QNetwork)
│     z_proj    → 256 维
│     proprio_proj → 64 维   → concat → LayerNorm → MLP → 标量
│     action_proj → 256 维
│
└── Q2 (QNetwork)  (结构与 Q1 完全相同，参数独立)
```

代码对应（`networks.py`）：
```python
def q_values(self, params, z_rl, proprio, action_chunk):
    q1 = q_network.apply(params["q1"], z_rl, proprio, action_chunk)
    q2 = q_network.apply(params["q2"], z_rl, proprio, action_chunk)
    return q1, q2
```

### 3.3 两个 Q 网络的分工

| 场景 | 用法 |
|------|------|
| **训练 Critic** | Q1、Q2 分别对 TD target 求 MSE，同时反传梯度 |
| **计算 TD target** | 用 target 网络的 $\min(Q1_{tgt}, Q2_{tgt})$，防高估 |
| **训练 Actor** | 只用 Q1（`q1, _ = critic.q_values(...)`），最大化 $\mathbb{E}[Q1]$ |

Critic 的完整 loss：

$$\mathcal{L}_{critic} = \underbrace{\mathbb{E}[(Q1 - y_{TD})^2]}_{\text{Q1 的 TD 误差}} + \underbrace{\mathbb{E}[(Q2 - y_{TD})^2]}_{\text{Q2 的 TD 误差}}$$

---

## 四、TD Target 的构成详解

本项目的一个 transition 是一个 **chunk**（多步动作序列），TD target 融合了 chunk 内的多步奖励：

$$y_{TD} = \underbrace{\sum_{t=0}^{T-1} \gamma^t r_t}_{\text{chunk 内折扣累积奖励}} + \underbrace{(1 - \text{done}) \cdot \gamma^T \cdot \min(Q1_{tgt}(s', a'), Q2_{tgt}(s', a'))}_{\text{下一状态的 bootstrap 价值}}$$

代码对应（`networks.py`）：
```python
def build_td_target(...):
    next_action = target_actor.sample_action(target_actor_params, ...)
    next_q1, next_q2 = target_critic.q_values(target_critic_params, next_z_rl, next_proprio, next_action)
    bootstrap = (1.0 - done) * (gamma ** chunk_len) * jnp.minimum(next_q1, next_q2)
    return _discounted_chunk_rewards(rewards, gamma) + bootstrap
```

这里的两项是**时序上的组合**，而不是 A2C 那种 $Q - V$ 的结构分解。

---

## 五、Actor 如何使用 Q 值？

Actor loss 中 Q 值以 **负号** 出现，即最大化 Q：

$$\mathcal{L}_{actor} = \underbrace{\lambda_{bc} \cdot \mathcal{L}_{BC}}_{\text{行为约束}} - \underbrace{\lambda_q \cdot \mathbb{E}[Q1(\pi)]}_{\text{最大化Q值}} + \underbrace{\lambda_{\delta} \cdot \mathcal{L}_{\Delta}}_{\text{平滑约束}}$$

**Q 值在这里的作用**：引导 Actor 生成的动作朝着高回报区域移动，而 BC 项防止 Actor 偏离已有数据分布太远（离线 RL 的核心约束）。

---

## 六、三种 AC 结构对比总结

| | 本项目 (TD3-style) | A2C/PPO-style | SAC |
|--|-------------------|---------------|-----|
| **Critic 输出** | $Q(s, a)$，需要 action 输入 | $V(s)$，不需要 action | $Q(s, a)$，需要 action |
| **双网络防高估** | ✅ Twin Q | ❌ 单个 V 网络 | ✅ Twin Q |
| **Actor 梯度来源** | $\nabla_\theta Q(s, \pi_\theta(s))$ | $A(s,a) \cdot \nabla_\theta \log\pi$ | $\nabla_\theta [Q - \alpha H(\pi)]$ |
| **BC 约束** | ✅（本项目额外加入） | ❌ | ❌（SAC 用熵正则化） |
| **适合 Offline RL** | ✅ | ❌（需要在线采样） | 需改造 |
