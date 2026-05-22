# PPO（Proximal Policy Optimization）算法详解

---

## 零、前置概念：四个术语的关系

### 0.1 两个独立维度

这四个概念来自**两个完全不同的维度**，经常被混淆：

| 维度 | 含义 |
|------|------|
| **Online RL** | 训练时**同时与环境交互**，实时采集新数据 |
| **Offline RL** | 训练时**只用已有数据集**，不与环境交互 |
| **On-policy** | 训练数据**必须来自当前策略**，用完即丢 |
| **Off-policy** | 训练数据**可以来自任意旧策略**，反复复用 |

### 0.2 四种组合

这两个维度可以任意组合，形成四种情况：

|  | **Online**（实时采集） | **Offline**（静态数据集） |
|--|----------------------|--------------------------|
| **On-policy** | PPO、A3C、TRPO | ❌ 逻辑上不可能（旧数据不满足 on-policy 要求） |
| **Off-policy** | **TD3/SAC + ReplayBuffer（本项目 Online RL）** | BCQ、IQL、CQL（本项目 Offline 训练脚本） |

### 0.3 本项目的定位

```
offline_train_from_replay.py  →  Offline RL + Off-policy（静态数据集，BCQ 风格）
run_online_rl.py              →  Online RL  + Off-policy（实时采集，TD3 风格）
```

**Online RL ≠ On-policy**，这是最容易混淆的地方。

### 0.4 为什么机器人用 Online RL + Off-policy？

```
机器人一个 episode 需要几十秒到几分钟
                ↓
如果用 On-policy（PPO）：
  用完数据就丢 → 机器人重新执行 → 时间成本极高

如果用 Off-policy（TD3）：
  每条 transition 复用几十次 → 数据效率极高
  同时机器人持续采集新数据 → 策略持续进化（Online）
```

**"Online" 的核心是闭环反馈**：机器人行为 → 新数据 → 训练 → 新 actor → 机器人行为进化。这与 Offline（静态数据，训完即止）有本质区别，与 on/off-policy 是两个独立问题。

---

## 一、PPO 是什么？

PPO 是 OpenAI 在 2017 年提出的 on-policy 策略梯度算法，目前是 **LLM RLHF（如 ChatGPT）、游戏 AI（OpenAI Five）、人形机器人（Figure、Tesla Optimus）** 中最流行的算法之一。

它的核心目标只有一句话：

> **每次更新策略时，不要走太大的步子。**

---

## 二、为什么需要 PPO？从 Policy Gradient 说起

最基础的策略梯度（REINFORCE）的更新公式：

$$\nabla J(\theta) = \mathbb{E}\left[ \sum_t A(s_t, a_t) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

直觉：**做了好动作（$A > 0$）就增加它的概率，做了坏动作（$A < 0$）就降低它的概率。**

**但存在一个致命问题**：步长太大会导致策略崩溃。

```
假设某个动作偶然得到高奖励
→ 梯度更新把这个动作概率大幅提高
→ 下一轮几乎只执行这个动作
→ 采集到的数据严重偏斜
→ 策略彻底崩溃，无法恢复
```

---

## 三、PPO 的核心机制：Clipped Surrogate Objective

PPO 引入一个**重要性采样比率**：

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

这个比率衡量**新策略和旧策略在同一动作上的概率之比**。

PPO 的 loss：

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \cdot A_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t \right) \right]$$

用图来理解 clip 机制：

```
当 A > 0（好动作，想增加概率）：
  r_t 增大 → 但超过 1+ε 时被截断 → 不会无限增大

当 A < 0（坏动作，想降低概率）：
  r_t 减小 → 但低于 1-ε 时被截断 → 不会无限降低

ε 通常取 0.2，即策略变化最多 ±20%
```

**这就是"Proximal（近端）"的含义：强制新旧策略保持在邻域内。**

---

## 四、PPO 的完整训练流程

PPO 是 **on-policy** 的，每轮循环：

```
① 用当前策略 π_old 采集 T 步数据（rollout）
         ↓
② 计算每步的优势函数 A_t（用 GAE 方法）
         ↓
③ 用同一批数据做 K 个 epoch 的梯度更新
   （每次更新都用 clip 约束不走太远）
         ↓
④ 丢弃这批数据，π_old ← π_new
         ↓
⑤ 重复 ①
```

> **关键：数据用完就丢**，这是 on-policy 的代价。

### 优势函数 GAE（Generalized Advantage Estimation）

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\lambda \to 0$：低方差，高偏差（近似 TD，短视）
- $\lambda \to 1$：低偏差，高方差（近似 Monte Carlo，长视）

---

## 五、PPO 和本项目（TD3-style）的对比

| | PPO | 本项目（TD3 + Replay） |
|--|-----|----------------------|
| **policy 类型** | On-policy | Off-policy |
| **数据复用** | 每批数据用完丢弃 | 每条数据复用几十次 |
| **Critic 学什么** | $V(s)$，不需要动作输入 | $Q(s,a)$，需要动作输入 |
| **策略约束** | clip $r_t$ 到 $[1-\epsilon, 1+\epsilon]$ | BC loss 约束动作不偏离 ref_chunk |
| **适合场景** | 数据采集便宜（仿真、游戏、LLM）| 数据采集昂贵（真实机器人）|
| **为什么 LLM 用 PPO** | LLM 推理快，可以大量并行采样 | - |
| **为什么机器人用 TD3** | 机器人执行慢，数据极贵，必须充分复用 | - |

---

## 六、为什么 LLM/推文里大量看到 PPO？

**RLHF（Reinforcement Learning from Human Feedback）的标准流程**：

```
① SFT（监督微调）→ 得到基础模型
② 训练 Reward Model（人类标注偏好数据）
③ PPO 微调语言模型：
     rollout：让模型生成回答
     reward：用 Reward Model 打分
     update：PPO 更新模型参数
```

LLM 用 PPO 的原因：
- **生成成本低**：GPU 上并行生成几千条回答只需几秒
- **环境是确定的**：文本生成是离散空间，重置简单
- **无需物理等待**：不像机器人要等几十秒执行动作

即便如此，PPO 在 LLM 中也在被挑战——DeepSeek-R1 用的 **GRPO**（Group Relative Policy Optimization），去掉了 Critic 网络，进一步简化。

---

## 七、On-policy 与 Replay Buffer 的关系

### On-policy 不能用传统 Replay Buffer

| | On-policy（PPO） | Off-policy（TD3/SAC） |
|--|-----------------|----------------------|
| **数据要求** | 数据**必须来自当前策略** | 数据可来自**任意旧策略** |
| **Replay Buffer** | ❌ 不能用（旧数据已失效） | ✅ 核心组件 |
| **数据复用** | 同批数据做 K epoch，然后丢弃 | 数据存入 buffer，反复采样 |

**根本原因**：PPO 的 loss 推导基于"数据是由 $\pi_{old}$ 采集的"这一假设。如果从 buffer 里取出几轮前的旧数据，这个假设被违反，$r_t(\theta)$ 的重要性权重偏差过大，训练不稳定甚至发散。

### PPO 的"伪 Replay"：同批数据多 epoch

PPO 并不是完全不复用数据，而是：

```
采集 N 步数据（来自 π_old）
     ↓
在这 N 步数据上做 K 个 epoch 更新（K 通常为 4-10）
     ↓
当策略变化超过 clip 阈值 ε 时，clip 自动截断梯度
     ↓
丢弃数据，重新采集
```

这是一种**有限的、当场复用**，而非跨轮次的历史复用。

### 有一种变体可以结合：PPO + 短窗口 Replay

学术上存在 **"PPO with experience replay"** 的变体，通过重要性采样（IS）权重修正旧数据的偏差：

$$\mathcal{L} = \mathbb{E}\left[ \frac{\pi_\theta(a|s)}{\pi_{behavior}(a|s)} \cdot A \right]$$

但这实际上就在向 off-policy 靠拢，工程复杂度大幅上升，实践中不如直接用 SAC/TD3。

---

## 八、一句话总结

> PPO 的本质是：**用 clip 机制给策略梯度上了一个"安全带"**，防止每次更新步子太大导致策略崩溃。它在数据采集便宜的场景（LLM、游戏仿真）中极为流行；而在数据昂贵的真实机器人场景中，off-policy 的 TD3/SAC + Replay Buffer 才是更合适的选择。On-policy 算法从根本上不兼容 Replay Buffer，这是算法设计上的本质区别，而非工程限制。

---

## 九、补充：PPO 也需要存储数据——Rollout Buffer vs Replay Buffer

PPO 采集的数据同样需要临时存储，但这与 TD3/SAC 的 Replay Buffer **在设计意图和使用方式上有本质区别**。

### 9.1 两种 Buffer 的对比

| | PPO 的 **Rollout Buffer** | TD3/SAC 的 **Replay Buffer** |
|--|--------------------------|------------------------------|
| **容量** | 只存当前轮采集的 N 步（如 2048 步） | 存历史所有数据（如 20 万条） |
| **生命周期** | 用完 K epoch 后**立即清空** | **永久保留**，新数据覆盖最旧的 |
| **数据来源** | 只来自**当前策略** | 来自**任意历史策略** |
| **采样方式** | 遍历全部（shuffle 后分 mini-batch） | 随机采样任意历史数据 |
| **本质** | 临时工作区（Working Memory） | 长期经验库（Experience Database） |

### 9.2 形象类比

```
PPO Rollout Buffer：
  像一个白板 → 今天写满 → 用完擦掉 → 明天重新写

TD3 Replay Buffer：
  像一本日记本 → 每天追加 → 翻到任意一页都能学 → 满了就覆盖最早的页
```

### 9.3 "不能用 Replay Buffer"的准确说法

PPO **不能用跨轮次的历史 Replay Buffer**，但**必须有当轮的临时存储（Rollout Buffer）**。

两者的数据流对比：

```
PPO 的数据流：
  采集 2048 步 → 存入 rollout_buffer
  → 做 10 epoch 训练（全部用这 2048 步）
  → rollout_buffer.clear()   ← 关键！
  → 重新采集

TD3 的数据流：
  每采集 1 步 → append 进 replay_buffer（永不主动清空）
  → 训练时随机采 256 条（可能是数天前的数据）
```

> **一句话**：PPO 需要的是一个"用完即清"的**临时暂存区**，TD3 需要的是一个"越积越多"的**历史经验库**。名字都叫 buffer，但设计哲学完全相反。
