# 基于当前仓库学习 RL 的知识点清单（Q&A）

这份文档面向“边看代码边学 RL”的场景，目标不是只解释抽象概念，而是把当前仓库
`rlt_online_rl` 里所有值得系统理解的知识点尽量完整地罗列出来，并明确它们分别在什么代码里落地。

建议阅读顺序：

1. 先读 [README](../README.md)
2. 再读 [actor_critic_training_explained.md](./actor_critic_training_explained.md)
3. 然后把本文当作学习提纲，逐条对照代码理解

---

## 一、总览：这个仓库在做什么

### Q1：这个仓库里的 RL 到底解决什么问题？

A：它不是从零学习整个机器人策略，而是在一个已经存在的基础策略之上做在线 refinement。

更具体地说：

- Machine A 提供冻结的上游策略输出
- 上游策略返回 `z_rl` 和 `ref_chunk`
- RL actor 读取 `z_rl / proprio / ref_chunk`
- actor 学习如何把 `ref_chunk` 修正成一个更适合当前任务关键阶段的动作 chunk

所以它是：

- 不是端到端从图像直接学动作的完整 RL
- 不是纯行为克隆
- 而是“参考动作条件下的在线 actor-critic refinement”

代码入口见 [README](../README.md) 和 [trainer.py](../src/rlt_online_rl/trainer.py:67)。

### Q2：这个仓库更像哪类 RL？

A：从实现风格上看，它更接近：

- 离策略 RL
- actor-critic
- twin critic
- target network
- 带行为克制项的 deterministic-ish continuous control

但它又不是教科书版 TD3/SAC，因为它有自己的任务结构：

- 动作是 chunk 而不是单步
- actor 以 `ref_chunk` 为条件
- actor loss 里同时有 BC 项、Q 项、delta 平滑项
- replay 数据里混合了 base policy、RL policy、人类接管数据

### Q3：为什么说这是一个很适合学习 RL 实战的仓库？

A：因为它同时包含了 RL 学习中最重要的几个层面：

- 算法层：actor、critic、TD target、target network、replay、bootstrapping
- 工程层：服务拆分、在线训练、checkpoint、snapshot、日志
- 数据层：warmup、online、human intervention、mixed data
- 部署层：训练时随机、推理时确定、实时执行、失败回退

很多教材只讲 loss，不讲系统；很多系统只讲流程，不讲算法。这个仓库两边都有。

---

## 二、先建立 RL 基础概念

### Q4：什么是 state、action、reward、transition？

A：

- `state`：当前环境状态
- `action`：当前策略采取的动作
- `reward`：环境对动作结果给的反馈
- `transition`：`(s, a, r, s')`

在这个仓库里，它们的对应关系不是 textbook 的单步形式，而是 chunk 形式：

- `state` 对应 `z_rl + proprio`
- `action` 对应 `action_chunk`
- `reward` 对应一个 chunk 内的 `rewards`
- `next_state` 对应 `next_z_rl + next_proprio`

见 [replay.py](../src/rlt_online_rl/replay.py:80)。

### Q5：什么是 policy？

A：policy 就是“给定状态，输出动作”的规则。常写作：

```text
a ~ pi(a | s)
```

在这个仓库里更准确地说是：

```text
a_chunk ~ pi(a_chunk | z_rl, proprio, ref_chunk)
```

注意这里 policy 还显式依赖 `ref_chunk`。

### Q6：什么是 value、Q value、critic？

A：

- value function `V(s)`：状态本身值多少钱
- action-value function `Q(s, a)`：状态下采取动作 `a` 值多少钱
- critic：近似 `Q` 或 `V` 的网络

这个仓库里用的是 `Q(s, a_chunk)`，所以 critic 输入状态和动作 chunk，输出标量 Q。

见 [networks.py](../src/rlt_online_rl/networks.py:130)。

### Q7：什么是 actor？

A：actor 就是策略网络。它决定“该做什么动作”。

本仓库的 actor 不是无条件输出动作，而是：

- 给定状态
- 给定参考动作 `ref_chunk`
- 输出一个 refined chunk

见 [networks.py](../src/rlt_online_rl/networks.py:53)。

### Q8：什么是 actor-critic？

A：actor-critic 的核心分工是：

- actor 负责提议动作
- critic 负责评价动作

然后：

- critic 先学会打分
- actor 再利用 critic 的分数改进自己的输出

这正是本仓库 `train_step()` 的更新顺序。

见 [trainer.py](../src/rlt_online_rl/trainer.py:132) 和 [trainer.py](../src/rlt_online_rl/trainer.py:175)。

### Q9：什么是离策略（off-policy）学习？

A：离策略指的是：

- 训练时使用的数据
- 不一定来自当前最新 policy

只要 replay 里存了历史样本，之后反复采样训练，就是离策略。

本仓库显然是离策略，因为：

- 有 replay buffer
- learner 持续从 replay 采样
- 样本可来自 warmup/base/RL/human/mixed

### Q10：什么是 bootstrapping？

A：bootstrapping 指的是：

- 不是只看眼前 reward
- 还看“下一状态继续执行后未来能值多少钱”

对应公式：

```text
target = r + gamma * next_value
```

本仓库里是 chunk 版：

```text
target_q = discounted_chunk_reward + gamma^H * min(next_q1, next_q2)
```

见 [networks.py](../src/rlt_online_rl/networks.py:226)。

### Q11：什么是折扣因子 gamma？

A：`gamma` 控制未来奖励的重要性。

- `gamma` 越大，越看重长期收益
- `gamma` 越小，越看重短期收益

本仓库默认是 `0.99`，见 [config.py](../src/rlt_online_rl/config.py:21)。

### Q12：什么是连续动作空间？

A：连续动作空间指动作不是离散类别，而是实数向量。

机器人关节角、位移、夹爪开合都属于连续动作。

本仓库的动作通常是：

- `action_dim = 7`
- 一个 chunk 里有 `chunk_len` 个时间步

所以一次输出的是一个连续值矩阵。

---

## 三、理解本仓库的状态、动作和任务建模

### Q13：这里的状态为什么拆成 `z_rl` 和 `proprio`？

A：

- `z_rl` 是 Machine A 提供的高层特征
- `proprio` 是机器人当前本体状态

这么拆的好处是：

- 上游大模型负责提取复杂表征
- 轻量 actor/critic 只处理低维结构化输入

见 [inference.py](../src/rlt_online_rl/inference.py:167)。

### Q14：`ref_chunk` 是什么？

A：`ref_chunk` 是上游基础策略给出的参考动作序列。

可以把它理解成：

- base policy 的建议动作
- actor 要围绕它做编辑，而不是从零生成

这是这个仓库和经典 RL 最大的差别之一。

### Q15：为什么动作不是单步，而是 chunk？

A：因为机器人实际执行时通常不是每个 tick 都重新完整推理一次，而是一次规划一小段动作序列。

这样做的好处：

- 降低推理调用频率
- 更贴近上游 VLA 的 chunk 输出方式
- replay 与控制执行天然对齐

代价是：

- credit assignment 更复杂
- 一个样本变成“状态 + 整段动作 + 整段奖励”

### Q16：`chunk_len` 的作用是什么？

A：`chunk_len` 表示每个 transition / actor 输出 / critic 打分对应多少个连续动作步。

例如：

- `chunk_len = 10`
- `action_dim = 7`

那么 actor 一次输出 `10 x 7`，critic 一次对整个 `10 x 7` 的 chunk 打一个 Q 分。

### Q17：这个任务为什么要区分 `critical_phase` 和 `full_task`？

A：

- `critical_phase`：只训练/执行关键精细阶段
- `full_task`：完整任务，但只有关键阶段才切到 RL

这体现了一个重要工程思想：

- RL 不一定非要接管整个任务
- 可以只接管最难、最需要在线修正的部分

见 [README](../README.md)。

---

## 四、Actor 网络要学习什么

### Q18：actor 的输入是什么？

A：`ChunkActor` 输入：

- `z_rl`
- `proprio`
- `ref_chunk`

见 [networks.py](../src/rlt_online_rl/networks.py:76)。

### Q19：actor 的结构是什么？

A：

1. `z_rl -> Linear(256)`
2. `proprio -> Linear(64)`
3. `ref_chunk(flatten) -> Linear(256)`
4. 拼接成 `576` 维
5. 经过 MLP trunk
6. 输出 `chunk_len * action_dim`
7. reshape 成 `[chunk_len, action_dim]`

MLP 隐层结构是：

- `Linear -> LayerNorm -> GELU`

见 [networks.py](../src/rlt_online_rl/networks.py:63)。

### Q20：actor 学到的到底是什么？

A：它学到的是：

- 在当前状态下
- 如何把参考动作 `ref_chunk`
- 改成一个回报更高、同时尽量合理的动作 chunk

不是：

- 从零生成整套动作逻辑
- 替代 Machine A 的全部能力

### Q21：为什么说它是 reference-conditioned actor？

A：因为 `ref_chunk` 是 actor 的显式输入。

所以 actor 表达的不是：

```text
pi(a | s)
```

而是：

```text
pi(a | s, ref)
```

这是一个非常关键的建模差异。

### Q22：actor 输出的是动作，还是分布？

A：严格说，它输出的是分布参数里的均值 `mu`。

代码定义是：

- `mu = actor_mean(...)`
- `std = fixed_std`
- 训练采样 `mu + std * noise`
- 推理时可以直接返回 `mu`

见 [networks.py](../src/rlt_online_rl/networks.py:102)。

### Q23：为什么说这个输出是“均值”不是“随便叫的名字”？

A：因为后续算法把它代入了高斯采样公式：

```text
a = mu + std * noise
```

谁处在这个公式的中心位置，谁就是均值。

如果它是方差，就应该：

- 保证非负
- 用来缩放噪声
- 而不是直接作为中心动作值

### Q24：如果动作是 `50 x 8`，是不是就有 400 个高斯？

A：按当前实现理解，是的。

更准确地说：

- actor 输出 `400` 个均值
- 每一维共享同一个固定标准差 `fixed_std`
- 组合成一个对角高斯分布

没有学习：

- 每一维单独的方差
- 维度之间的协方差

### Q25：为什么推理时常直接用 `mu`？

A：因为：

- `mu` 是分布中心
- 推理通常希望稳定可复现
- 训练时的噪声主要用于探索，而不是部署

所以推理时一般用 deterministic actor mean。

### Q26：为什么不直接输出一个确定动作，而要加随机性？

A：主要原因是探索。

如果始终输出同一个动作：

- critic 只能看到一个点
- actor 很难发现附近更优的动作

加入小噪声后：

- 能在 `mu` 周围尝试邻域动作
- critic 才能学习“附近哪里更好”

### Q27：这个 actor 是 SAC 风格吗？

A：不是完整 SAC 风格。

因为它没有：

- 学习状态相关方差
- 熵温度
- reparameterization + entropy term 的完整 SAC 目标

它更像：

- 固定噪声的 Gaussian actor
- 配合 twin critic
- 再叠加 BC 约束

---

## 五、Critic 网络要学习什么

### Q28：critic 的输入是什么？

A：单个 `QNetwork` 的输入是：

- `z_rl`
- `proprio`
- `action_chunk`

见 [networks.py](../src/rlt_online_rl/networks.py:149)。

### Q29：critic 的结构是什么？

A：和 actor 很像，只是第三路输入从 `ref_chunk` 换成了 `action_chunk`：

1. `z_rl -> Linear(256)`
2. `proprio -> Linear(64)`
3. `action_chunk(flatten) -> Linear(256)`
4. 拼接
5. MLP trunk
6. 输出一个标量 Q

见 [networks.py](../src/rlt_online_rl/networks.py:139)。

### Q30：为什么 critic 输入的是 `action_chunk` 而不是 `ref_chunk`？

A：因为 critic 评价的是：

- 这个状态下
- 这段具体执行动作
- 能带来多少回报

也就是说它评价的是“执行结果”，不是“参考建议”。

### Q31：为什么要 twin critic？

A：因为单个 critic 容易高估 Q 值。

双 Q 的典型做法是：

- 用两个 critic 独立估计
- bootstrap 时取 `min(q1, q2)`

这样能缓解过高估计。

见 [networks.py](../src/rlt_online_rl/networks.py:247)。

### Q32：critic 学到的 Q 值直觉上表示什么？

A：可以理解成：

- 当前这段动作 chunk 如果现在执行
- 后续继续按当前 policy 走
- 预计能带来的累计折扣回报

### Q33：为什么 critic 只输出一个标量，而不是输出每一步的 value？

A：因为当前建模把整个 chunk 当作一个动作。

所以 critic 回答的问题是：

```text
这个 chunk 整体值多少钱？
```

而不是：

```text
chunk 内第 3 步值多少钱？
```

---

## 六、训练目标和损失函数

### Q34：critic loss 是什么？

A：

```text
Lcritic = MSE(q1, target_q) + MSE(q2, target_q)
```

见 [networks.py](../src/rlt_online_rl/networks.py:274)。

### Q35：`target_q` 是怎么构造的？

A：

1. 用 `target_actor` 在下一状态采样 `next_action`
2. 用 `target_critic` 估计 `next_q1, next_q2`
3. 取 `min(next_q1, next_q2)`
4. 加上当前 chunk 内折扣奖励和 bootstrap 项

对应 [networks.py](../src/rlt_online_rl/networks.py:226)。

### Q36：为什么这里对一个 chunk 用 `gamma^H`？

A：因为一个 transition 对应整段 `H = chunk_len` 个时间步。

当前 chunk 的 reward 已经显式求和了，所以 bootstrap 要跨过整段 chunk，再乘 `gamma^H`。

### Q37：actor loss 是什么？

A：

```text
actor_loss = bc_weight * bc_penalty - q_weight * actor_q + delta_weight * delta_penalty
```

见 [README](../README.md) 和 [trainer.py](../src/rlt_online_rl/trainer.py:191)。

### Q38：`bc_penalty` 表示什么？

A：它约束 actor 不要偏离合理示范太远。

但这里的 target 不是统一固定的，而是按 `source_chunk` 逐步切换：

- human/mixed 步：对齐 `action_chunk`
- base/RL 步：对齐 `ref_chunk`

这很关键，见 [trainer.py](../src/rlt_online_rl/trainer.py:212)。

### Q39：为什么 human 数据要对齐 `action_chunk`，而 policy 数据对齐 `ref_chunk`？

A：因为 human takeover 的意义是：

- 教 actor 学会怎么修正 base reference

而不是把 actor 输入也替换成人类轨迹。

部署时 actor 仍然看到的是 `ref_chunk`，所以训练也要保持这个条件结构。

### Q40：`actor_q` 表示什么？

A：它是当前 actor 生成动作后，经 critic 打分得到的平均 Q。

在 loss 里是 `- q_weight * actor_q`，表示：

- 希望 actor 生成的动作拿到更高 Q

### Q41：`delta_penalty` 是什么？

A：它是动作平滑/几何合理性的约束项。

当前做法是：

- 把训练动作反归一化回绝对动作 chunk
- 比较相邻步之间的 delta
- 让 actor 生成的 step-to-step 变化尽量接近目标动作的 step-to-step 变化

见 [trainer.py](../src/rlt_online_rl/trainer.py:224)。

### Q42：为什么 actor loss 同时要有 BC 和 Q？

A：只用 BC 会变成纯模仿学习，只会复制参考或人类动作。

只用 Q 会有两个问题：

- critic 早期不稳定
- actor 容易偏到不合理区域

所以这套设计是：

- 用 BC 提供稳定先验
- 用 Q 提供超越示范的优化方向

这也是许多离线/在线混合 RL 方法的核心思想。

### Q43：warmup 和 online 为什么用不同的 `bc_weight` / `q_weight`？

A：因为早期 critic 不够可靠，训练更依赖 BC；
后期 critic 稳定一些，Q 项可以逐渐发挥更大作用。

配置见 [config.py](../src/rlt_online_rl/config.py:24)。

### Q44：为什么 actor 不每一步都更新？

A：因为 critic 先要跟上。

当前实现里：

- critic 每步更新
- actor 每 `actor_update_period` 步更新一次

这和 TD3 中“延迟 actor 更新”的想法一致。

见 [trainer.py](../src/rlt_online_rl/trainer.py:263)。

---

## 七、Target Network 和稳定训练

### Q45：什么是 target network？

A：target network 是 online 网络的一个“慢速拷贝”。

它的用途是：

- 计算 bootstrap target 时不要用瞬息万变的最新网络
- 降低训练目标抖动

### Q46：这个仓库有哪些 target network？

A：

- `target_actor_params`
- `target_critic_params`

定义见 [trainer.py](../src/rlt_online_rl/trainer.py:51)。

### Q47：target network 如何更新？

A：使用 soft update：

```text
target = (1 - tau) * target + tau * source
```

见 [trainer.py](../src/rlt_online_rl/trainer.py:115)。

### Q48：为什么不直接 hard copy？

A：hard copy 会让 target 周期性剧烈跳变；soft update 更平滑，通常更稳定。

### Q49：这里的 `target_tau` 有什么含义？

A：它控制 target 跟随 online 参数的速度。

- 小 `tau`：更稳定，但更新慢
- 大 `tau`：更快，但更不稳定

---

## 八、Replay Buffer 和数据构造

### Q50：为什么 RL 需要 replay buffer？

A：因为离策略训练需要反复重用过去的数据。

好处：

- 提高样本效率
- 允许 learner 与 rollout 解耦
- 混合不同来源数据

### Q51：这个仓库里 replay 存的是什么？

A：存的是 chunk-level transition，不是原始单步 transition。

主要字段有：

- `z_rl`
- `proprio`
- `ref_chunk`
- `action_chunk`
- `rewards`
- `done`
- `next_z_rl`
- `next_proprio`
- `next_ref_chunk`
- `source`
- `source_chunk`

见 [replay.py](../src/rlt_online_rl/replay.py:80)。

### Q52：`source` 和 `source_chunk` 有什么区别？

A：

- `source`：整个 chunk 的主标签
- `source_chunk`：chunk 内每个时间步的来源标签

后者更细，因为一个 chunk 内可能混有人类和策略动作。

### Q53：为什么 replay 里同时保留 `ref_chunk` 和 `action_chunk`？

A：因为两者在训练中扮演不同角色：

- critic 评价实际执行的 `action_chunk`
- actor 以 `ref_chunk` 为条件生成修正动作
- BC 目标在不同 source 下切换

缺一个都不行。

### Q54：为什么 raw episode trace 和 replay transition 要分开存？

A：因为两者用途不同：

- raw trace：完整记录真实执行过程，方便回放、调试、重建
- replay transition：训练时高效采样使用

这是典型的“原始日志”和“训练样本”分层。

### Q55：chunk transition 是怎么从 step trace 构造出来的？

A：在 episode 结束后：

1. 收集整个 episode 的逐步记录
2. 按 chunk 窗口切片
3. padding 对齐
4. 计算 `next_ref_chunk`
5. 汇总 `source` 和 `source_chunk`
6. 写入 replay

见 [replay.py](../src/rlt_online_rl/replay.py:247)。

### Q56：为什么 episode 结束后再建 replay，而不是边跑边写？

A：因为需要：

- 完整 chunk 窗口
- next anchor
- 可能的人类接管恢复点
- 终止边界处理

这些信息往往要等 episode 完整结束后才更容易正确构造。

### Q57：为什么 replay 支持 warmup / online / human intervention 的混合采样？

A：因为真实机器人在线学习数据稀缺且昂贵。

只依赖纯在线 RL 数据通常不够稳定，所以要混合：

- warmup 示范
- online RL 执行
- 人类接管纠正

### Q58：什么是 stratified replay sampling？

A：就是按比例从不同数据子集采样，而不是完全均匀随机。

这个仓库配置里支持：

- `uniform`
- `stratified`

这样可以控制近期在线样本、人类干预样本、warmup 样本的占比。

见 [config.py](../src/rlt_online_rl/config.py:72)。

---

## 九、Warmup、Online 和学习预算

### Q59：什么是 warmup？

A：warmup 是 RL 接管前的数据积累与预训练阶段。

在本仓库里：

- actor 不控制机器人
- 先收集足够 replay
- learner 开始训练 actor/critic

### Q60：为什么 warmup 很重要？

A：因为：

- critic 在零数据下没有意义
- actor 直接在线探索风险太大
- 真实机器人上必须尽量减少随机试错

所以先用安全的 base/reference 数据打底。

### Q61：`warmup_min_size` 的作用是什么？

A：表示 replay 至少积累到多少条 transition 后 learner 才开始训练。

### Q62：`warmup_post_collect_updates` 是什么？

A：它表示：

- warmup 数据收集达到阈值后
- 还要额外做多少次 learner update
- 达标后才允许 online rollout

这能避免“刚够数据就立刻上线”的不稳定状态。

### Q63：如果 `warmup_post_collect_updates` 是 `null` 呢？

A：那么 warmup 所需训练步数会按：

```text
warmup_ready_adds_total * grad_updates_per_cycle
```

自动推导。

见 [trainer.py](../src/rlt_online_rl/trainer.py:465)。

### Q64：什么是 `grad_updates_per_cycle`？

A：这是 update-to-data ratio 的实现方式。

它控制：

- 每增加一定量数据
- learner 理论上要追多少步梯度更新

### Q65：为什么 learner 会有 `pending_update_budget`？

A：因为 learner 不一定无限快地训练。

仓库把“应该训练多少步”和“已经训练多少步”分开计数，得到一个待消耗预算：

- 有预算就继续 train
- 没预算就 idle

这避免 learner 无限制过拟合旧数据。

### Q66：什么是 `ready_for_online`？

A：表示是否满足在线使用 actor 的条件。

通常要同时满足：

- replay 已达到 warmup 要求
- learner 已完成 warmup 所需更新步数

见 [trainer.py](../src/rlt_online_rl/trainer.py:488)。

---

## 十、人类接管与混合监督

### Q67：为什么真实机器人 RL 常常需要 human takeover？

A：因为：

- 机器人探索成本高
- 错误动作可能损坏设备或任务样本
- 人类能在关键时刻提供安全纠偏

### Q68：human data 在这个仓库里起什么作用？

A：

- 不是简单当作“另一批 demonstration”
- 而是教 actor 学会如何在 `ref_chunk` 条件下做纠正

### Q69：为什么 `MIXED` source 很重要？

A：因为真实接管经常发生在 chunk 中途：

- 前半段是 policy
- 后半段是 human

如果只给整个 chunk 一个粗标签，就丢失了 step-level 监督信息。

### Q70：训练时 human 数据如何改变 BC target？

A：在 `update_actor()` 里：

- human/mixed step 对齐 `action_chunk`
- policy step 对齐 `ref_chunk`

见 [trainer.py](../src/rlt_online_rl/trainer.py:219)。

### Q71：为什么这不等于“直接把 human action 当成新的 ref_chunk”？

A：因为部署时 actor 输入仍然来自 Machine A 的 `ref_chunk`。

训练条件和部署条件必须尽量一致。

---

## 十一、推理与部署

### Q72：训练时 actor 和部署时 actor 有什么不同？

A：

- 训练时通常允许随机采样
- 部署/评估时通常使用 deterministic mean

### Q73：actor service 的作用是什么？

A：它负责：

- 持有当前最新 actor 参数
- 提供低延迟推理接口
- 周期性热加载 learner 导出的 snapshot

### Q74：为什么 learner 不直接参与在线推理？

A：因为：

- learner 可能占 GPU 并持续训练
- 推理要更低延迟、更稳定
- 把训练和推理服务分离更安全

### Q75：`RLTPolicyInferenceWrapper` 做什么？

A：它是一个纯 JAX 推理包装器，用来：

- 构造 actor
- 编译 mean/sample 前向
- 执行单次推理

见 [inference.py](../src/rlt_online_rl/inference.py:201)。

### Q76：为什么 inference wrapper 里既有 mean 也有 sample？

A：因为：

- 在线训练阶段可能需要 stochastic inference
- eval-only rollout 要强制 deterministic

### Q77：什么是 actor snapshot？

A：是 learner 导出的 actor 参数快照，供 actor service 热加载。

通常包含：

- `version`
- `global_step`
- `rl_config`
- `actor_params`

见 [trainer.py](../src/rlt_online_rl/trainer.py:395)。

---

## 十二、动作表示与归一化

### Q78：为什么需要 `action_representation`？

A：因为机器人动作可以有多种表示方式：

- 绝对动作 `abs`
- 增量动作 `delta_chunk`

不同表示会影响：

- 优化难度
- 数值范围
- 平滑性
- 是否更容易泛化

### Q79：`delta_chunk` 是什么？

A：它表示 chunk 内每一步动作相对于前一步或初始状态的变化量，而不是绝对关节位置。

对精细修正任务通常更自然。

### Q80：为什么动作要归一化？

A：因为不同关节/夹爪量纲和范围不同。

归一化的好处：

- 梯度更稳定
- 各维尺度更接近
- loss 不会被某些大数值维度主导

### Q81：`delta_penalty` 为什么要先反归一化再算？

A：因为它想约束的是“真实可执行几何变化”，不是归一化空间里的抽象数值距离。

### Q82：学习动作表示时应该注意什么？

A：

- 模型输出空间是什么
- replay 存的是什么
- loss 在什么空间上算
- 推理执行前是否要反归一化

这是机器人 RL 中非常容易混乱的一层。

---

## 十三、配置、超参数与实验控制

### Q83：学习这个仓库时最重要的超参数有哪些？

A：

- `gamma`
- `fixed_std`
- `reference_dropout_prob`
- `warmup_bc_weight`
- `warmup_q_weight`
- `online_bc_weight`
- `online_q_weight`
- `delta_weight`
- `actor_hidden_dim`
- `critic_hidden_dim`
- `target_tau`
- `actor_update_period`
- `warmup_min_size`
- `warmup_post_collect_updates`
- `grad_updates_per_cycle`

定义见 [config.py](../src/rlt_online_rl/config.py:14)。

### Q84：`fixed_std` 太大或太小会怎样？

A：

- 太大：探索太乱，动作不稳，critic target 噪声大
- 太小：探索不足，actor 很难发现比参考动作更好的邻域

### Q85：`reference_dropout_prob` 是什么？

A：训练时有一定概率把 `ref_chunk` 整块置零。

见 [networks.py](../src/rlt_online_rl/networks.py:210)。

### Q86：为什么要做 reference dropout？

A：为了防止 actor 过度依赖参考动作，提升鲁棒性。

如果没有它，actor 可能只学会“机械复制 ref 附近的小扰动”，对 ref 质量变化不够稳健。

### Q87：`actor_hidden_dim` / `critic_hidden_dim` 学什么时要关注？

A：它们控制模型容量。

- 太小：表达能力不足
- 太大：训练更慢，更易过拟合，也可能更难稳定

### Q88：为什么 actor 和 critic 可以用不同网络宽度？

A：因为二者任务不同：

- actor 学生成
- critic 学评估

表达需求和稳定性需求不一定一致。

---

## 十四、代码阅读顺序建议

### Q89：如果我想真正学懂，推荐先看哪些文件？

A：推荐顺序：

1. [README](../README.md)
2. [config.py](../src/rlt_online_rl/config.py)
3. [networks.py](../src/rlt_online_rl/networks.py)
4. [replay.py](../src/rlt_online_rl/replay.py)
5. [trainer.py](../src/rlt_online_rl/trainer.py)
6. [inference.py](../src/rlt_online_rl/inference.py)
7. [tests/test_trainer.py](../tests/test_trainer.py)
8. [scripts/offline](../scripts/offline/README.md)

### Q90：为什么先看 config？

A：因为 config 决定了整个系统有哪些概念存在。

很多读代码困难，根源是没先建立参数和对象地图。

### Q91：为什么要看测试？

A：因为测试能快速告诉你：

- 作者认为哪些行为是关键保证
- 哪些逻辑最容易出错
- 一个函数应当如何被使用

例如：

- actor update period
- human BC target 切换
- warmup budget
- freeze_after_warmup

都在测试里有直接例子。

---

## 十五、从算法视角应该主动问自己的问题

### Q92：actor 为什么依赖 `ref_chunk`，而 critic 不依赖？

A：因为 actor 的角色是“修正器”，critic 的角色是“裁判”。

裁判评价的是最终动作值，不需要知道这个动作最初参考是谁给的。

### Q93：为什么 critic 训练用 replay 里的执行动作，而不是 actor 当前生成动作？

A：因为 critic 是在做 fitted Q evaluation：

- 用真实执行过的动作
- 对真实收集到的回报做监督

这样监督信号更扎实。

### Q94：为什么 actor 更新时要用 critic 评分自己的动作？

A：因为 actor 要学的是：

- 当前这套参数生成什么动作更值钱

如果不通过 critic，它就只能做行为克隆。

### Q95：为什么 actor loss 用 `q1` 而不是 `min(q1, q2)`？

A：这是当前实现选择，不是唯一标准答案。

直观上：

- critic target 用 `min` 是为降低高估
- actor 更新常见做法是用一个 critic 或某个组合

学习时你应该把这看作“实现选择”，并思考替代方案是否更稳。

### Q96：为什么 actor 采样动作时要先对 `ref_chunk` dropout？

A：因为 actor 更新时如果总把高质量 `ref_chunk` 当成依赖支点，可能对参考过拟合。

dropout 后它需要学会：

- 在部分参考缺失时仍然做出合理修正

### Q97：为什么 rollout 里 warmup 阶段不让 actor 控制？

A：因为 actor 在早期几乎没训练好，真实机器人上直接放权风险过大。

### Q98：为什么 online 切换只发生在 episode 边界？

A：因为：

- 中途切换会破坏当前控制上下文
- 数据语义更混乱
- 工程上更难分析问题

---

## 十六、从工程视角应该主动问自己的问题

### Q99：为什么系统拆成 Machine A / actor_service / learner_service / replay_manager？

A：因为每部分资源需求和职责不同：

- Machine A：大模型/特征提供
- actor_service：低延迟推理
- learner_service：持续训练
- replay_manager：数据存储与采样

这是典型的在线学习系统解耦。

### Q100：为什么 actor_service 要热加载 snapshot，而不是直接拿 learner 内存里的参数？

A：因为服务解耦后：

- 推理不依赖 learner 存活
- learner 崩了也不一定影响推理
- 参数发布可以版本化

### Q101：为什么要记录 `actor_version`？

A：因为在线系统里非常需要知道：

- 当前 rollout 用的是哪一版 actor
- learner 训练到了哪一步
- 某次失败是否和某个参数版本相关

### Q102：为什么保存 checkpoint 和 snapshot 两种文件？

A：

- checkpoint：用于恢复完整训练状态
- snapshot：用于部署推理

两者用途不同，所以分开。

---

## 十七、如何从这个仓库学习 RL 数学

### Q103：学习这个仓库时，哪些公式必须会写？

A：至少应能自己写出：

1. chunk 折扣回报
2. TD target
3. twin critic loss
4. actor loss
5. soft update
6. Gaussian sampling

### Q104：应该自己推导哪些式子？

A：建议至少手推：

- 为什么 `gamma^H` 出现在 chunk bootstrap 前
- 为什么 `Lcritic` 是两个 MSE 相加
- 为什么 actor loss 是 `BC - Q + delta`
- 为什么 deterministic inference 直接取 `mu`

### Q105：学习 RL 时应分清“目标函数”和“实现细节”吗？

A：必须分清。

例如：

- “actor 最大化 Q 并受 BC 约束”是目标思想
- “MLP 用 GELU + LayerNorm”是实现细节

不要把具体实现误认为算法本质。

---

## 十八、如何从这个仓库学习实验分析

### Q106：训练过程中应该重点看哪些指标？

A：

- `critic_loss`
- `actor_loss`
- `actor_q`
- `bc_penalty`
- `bc_ref_penalty`
- `bc_human_penalty`
- `delta_penalty`
- `did_actor_update`
- `ready_for_online`

这些都在 learner metrics 中有记录。

### Q107：`critic_loss` 下降是否就表示策略变好了？

A：不一定。

critic loss 只表示 critic 在拟合 target_q 上更好，不代表：

- actor 一定更优
- 真实机器人成功率一定提升

### Q108：什么情况下 `actor_q` 上升但真实表现变差？

A：典型情况是 critic 偏差变大，actor 学会“利用 critic 漏洞”。

这就是为什么：

- 需要 twin critic
- 需要 target network
- 需要 BC 约束
- 需要真实 rollout 验证

### Q109：为什么要区分 `bc_ref_penalty` 和 `bc_human_penalty`？

A：因为它们反映 actor 在两类监督来源上的拟合情况不同。

如果：

- `bc_ref_penalty` 很低
- `bc_human_penalty` 很高

可能说明 actor 更擅长复制 base ref，不擅长学接管修正。

### Q110：为什么 sample composition metrics 很重要？

A：因为 learner 学到什么，很大程度上取决于“喂给它的 batch 由什么组成”。

代码里显式记录：

- recent online ratio
- warmup demo ratio
- human intervention ratio

见 [trainer.py](../src/rlt_online_rl/trainer.py:324)。

---

## 十九、你应该特别警惕的理解误区

### Q111：误区一：这个 actor 是从状态直接学整个任务策略

A：不对。它是条件在 `ref_chunk` 上的 refinement actor。

### Q112：误区二：actor 输出的是最终一定执行的动作

A：不完全对。训练时通常是 sampled action，部署时常取 mean。

### Q113：误区三：critic 在学 reward model

A：不对。critic 学的是累计折扣回报，不是单步即时 reward 预测器。

### Q114：误区四：有了 human data 就是在做纯 imitation learning

A：不对。这里 human data 只是 actor loss 中 BC target 的一部分来源；整体仍是 actor-critic RL。

### Q115：误区五：replay 只是缓存，和算法无关

A：不对。replay 的构造方式、采样比例、source 标注都会直接改变训练目标的有效含义。

### Q116：误区六：target network 只是工程优化

A：不对。它对 bootstrapped RL 的稳定性通常是核心机制之一。

---

## 二十、建议你按这份清单亲自验证的代码问题

### Q117：你应该自己回答哪些“读代码检查题”？

A：建议至少自己验证以下问题：

1. actor 的输入维度各是多少，拼接后是多少？
2. critic 为什么不输入 `ref_chunk`？
3. `build_td_target()` 里为什么用 `min(q1, q2)`？
4. `actor_update_period=2` 时第几步 actor 会更新？
5. human step 的 BC target 为什么是 `action_chunk`？
6. `delta_penalty` 为什么只看前六个关节？
7. `freeze_after_warmup=True` 时 learner 何时停止更新？
8. `warmup_post_collect_updates=null` 时预算怎么计算？
9. `deterministic=True` 时 inference 路径走哪条代码？
10. snapshot 与 checkpoint 分别保存哪些内容？

如果这些都能自己从代码里回答，你对这个仓库的核心 RL 逻辑就已经比较扎实了。

---

## 二十一、建议的学习路径

### Q118：如果我是 RL 初学者，怎么学这套代码最有效？

A：建议分四轮：

第一轮，只看流程：

- 数据从哪来
- actor/critic 各在哪里
- replay 怎么参与

第二轮，只看损失：

- critic loss
- actor loss
- target update

第三轮，只看数据语义：

- `ref_chunk`
- `action_chunk`
- `source_chunk`
- warmup / online / human

第四轮，只看系统：

- learner
- actor service
- replay manager
- rollout driver

### Q119：学这份代码时，最值得自己动手做什么实验？

A：建议按顺序尝试：

1. 把 `actor_update_period` 改成 `1`，观察训练变化
2. 把 `reference_dropout_prob` 改成 `0`
3. 把 `online_q_weight` 调大，看 actor 是否更激进
4. 把 `delta_weight` 设为 `0`，看动作是否更抖
5. 分析 human intervention 比例变化对结果的影响

### Q120：最终你应该能达到什么理解水平？

A：理想目标是你能独立回答：

- 这个仓库的 state、action、reward、transition 分别是什么
- actor 和 critic 各学什么
- 为什么需要 replay、target network、twin critic
- 为什么 actor loss 不是单一 BC 或单一 RL
- 为什么在线机器人 RL 必须重视 warmup、人类接管和部署架构

如果这些都能讲清楚，你就不只是“看懂代码”，而是真的开始理解 RL 系统了。

---

## 附：最关键的代码文件索引

- [README](../README.md)
- [actor_critic_training_explained.md](./actor_critic_training_explained.md)
- [config.py](../src/rlt_online_rl/config.py)
- [networks.py](../src/rlt_online_rl/networks.py)
- [trainer.py](../src/rlt_online_rl/trainer.py)
- [replay.py](../src/rlt_online_rl/replay.py)
- [inference.py](../src/rlt_online_rl/inference.py)
- [tests/test_trainer.py](../tests/test_trainer.py)
- [tests/test_networks.py](../tests/test_networks.py)
- [scripts/offline/README.md](../scripts/offline/README.md)
