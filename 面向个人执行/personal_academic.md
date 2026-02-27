**收到显存细节——单卡48GB、双卡96GB但无NVLink。这反而让你拥有一个“单卡巨无霸、双卡各自为战”的独特试验台。**

**结论先行**：这台机器是**mHC与TopoLoss验证的黄金配置**，但必须**放弃双卡联合训练，拥抱单卡大batch + 双卡并行搜参**。本月只做两条主线：**Phase 1（mHC+Diffusion Policy）**与**Phase 2（TopoLoss+SparseWorld-TC）**。

---

# 🔬 一、mHC 验证架构（两战场、两靶点）

## 🚗 智驾·最佳靶点：BEVFormer 的 ResNet-101 骨干

**为什么是它？**
- **深度足够**：101层，信号衰减明显，mHC的恒等映射恢复在此收益最可测；
- **改动最小**：只替换 `torchvision.models.resnet101` 的 Bottleneck 残差连接，不动BEVFormer的注意力；
- **显存舒适**：单卡48GB可跑 **batch size 32**（原版BEVFormer batch 8 已够），甚至能塞下时序模块。

**改造方案**  
```python
# 原版
out = self.conv3(out)
out = self.bn3(out)
out += identity  # 残差拼接

# mHC版（示意）
out_4x = torch.cat([identity, out, some_expand], dim=1)  # 扩至4倍通道
weight_matrix = self.sinkhorn(self.learned_matrix)       # 双随机约束
out = torch.einsum('oc,btchw->bthw', weight_matrix, out_4x)
```
- **预训练**：从ImageNet22K预训练权重开始，微调BEVFormer；
- **数据集**：nuScenes 检测任务，重点关注**小目标（锥桶、儿童）**；
- **评估指标**：mAP（低矮障碍物）、训练损失方差（尖峰次数）；
- **基线**：官方BEVFormer + ResNet101；
- **显存**：≈28GB（batch=16），单卡48GB充裕；
- **周期**：2周（代码移植1周 + 训练调优1周）。

**预期收益**  
- 训练损失曲线**无尖峰震荡**；
- **小目标召回率↑3~5%**（信号直达深层，不丢失细节）。

---

## 🦾 具身·最佳靶点：Diffusion Policy 的 1D UNet

**为什么是它？**
- **有效层数超200**：UNet时间展开×空间下采样，梯度消失是扩散模型通病；
- **mHC与扩散天生适配**：每步去噪需能量稳定，mHC的**Amax≈1.6**特性可稳定噪声预测；
- **显存极友好**：DP的UNet参数量≈30M，单卡48GB可跑 **batch 64+**，甚至能塞下8帧观测。

**改造方案**  
- 替换 UNet 中所有 ResBlock 的残差连接为 **4通道mHC**；
- **跳跃连接**改为**可学习mHC加权融合**（原为直接拼接）；
- **数据集**：LIBERO（90个长时任务），或 RoboMimic 的插拔任务；
- **基线**：[Diffusion Policy 官方代码](https://github.com/real-stanford/diffusion_policy)；
- **显存**：≈12GB（batch=32），单卡48GB可同时跑多个超参实验；
- **周期**：1.5周（UNet改造1周 + 训练0.5周）。

**预期收益**  
- **单步去噪质量提升**，推理步数从8→4，**控制频率从7Hz→10Hz**；
- LIBERO 成功率↑3~5%（尤其长序任务）。

**⚠️ 禁止触碰**：π0、OpenVLA（7B+），双5880也跑不动。

---

# 🗂️ 二、拓扑生成验证架构（主攻靶点）

## 🌍 3D生成·最佳靶点：SparseWorld-TC 的拓扑正则化注入

**为什么是它？**
- **与个人理论研究完美契合**：直接作为 `main0.tex` (Topologically Constrained AI) 中 Innovation I 的实验验证平台。
- **与主线工作高度协同**：白天用 Cosmos/AlpaSim 搞工程生成，晚上用 SparseWorld-TC 搞数学/拓扑约束生成，算力与数据资产（nuScenes格式）完全复用。
- **解决生成模型的痛点**：当前 3D 生成模型常出现“拓扑坍塌”（如车道线断裂、车辆粘连），引入持续同调（Persistent Homology）和 Betti 数约束是极具数学美感的解法。

**具体方案**  
1. **拓扑特征提取**：  
   - 在 SparseWorld-TC 的 3D 占据栅格（Occupancy）或点云输出端，接入轻量级的持续同调计算模块（如 `GUDHI` 或 `Ripser`）。
   - 提取生成结果的 Betti 数曲线（$b_0$ 连通分支, $b_1$ 环）。
2. **拓扑损失函数（Topology-Aware Loss）**：  
   - 将生成的拓扑签名与真实 NuScenes 数据的拓扑签名进行对比，计算 Wasserstein 距离。
   - 将此距离作为正则化项 $\mathcal{L}_{\text{topo-reg}}$ 加入 SparseWorld-TC 的训练目标中。
3. **数据集**：nuScenes（直接复用主线下载的数据）。
4. **基线**：原始 SparseWorld-TC。
5. **周期**：2.5周（跑通基线 1周 + 接入拓扑Loss 1周 + 收集对比数据 0.5周）。

**预期收益**  
- **生成质量的结构性提升**：在不增加模型参数的情况下，显著减少生成场景中的物理/结构违和感。
- **直接产出一篇极具数学深度的顶会论文**（即 `main0.tex` 的完全体）。

---

## 🦾 具身·延后方向：RT-1-X 的物体操作记忆（本月不主攻）

**为什么是它？**
- **物体操作记忆是具身最痛的痛点**：抓鸡蛋像第一次见鸡蛋；  
- RT-1-X 是**最小的VLA**（≈35M参数），单卡48GB可训；  
- Engram可存储**成功抓握的力觉/轨迹原型**，以`(物体类别, 材质)`为key。

**具体方案**  
1. **预填充Engram表**：  
   - 用 Isaac Sim 仿真生成1000次抓取；  
   - 对成功轨迹提取**夹爪开度序列 + 末端力曲线**，降维至64维向量；  
   - Key：`(物体ID, 材质标签)` → N-gram哈希。
2. **模型改造**：  
   - RT-1-X 的 FiLM 条件注入层旁路一个**Engram检索分支**；  
   - 检索到的原型向量与语言指令嵌入拼接，共同调制动作解码器。
3. **数据集**：RT-1 机器人数据集子集（或自行采集少量真机数据）；  
4. **基线**：原始RT-1-X；  
5. **显存**：≈20GB（batch=32，Engram表0.5GB）；  
6. **周期**：建议下月单独排期（本月仅保留方案设计与接口草图）。

**预期收益**  
- **零样本抓握陌生物体成功率↑15~20%**（如仿真→真机迁移）；  
- 验证“**记忆即泛化**”范式，是具身VLA的稀缺工作。

---

# ⚙️ 三、双5880 的作战阵型（无NVLink策略）

**铁律**：**绝不跨卡拆分模型**（无NVLink，通信开销>收益）。

**推荐战术**：

| 战术 | 执行方式 | 适合场景 |
|------|--------|--------|
| **单卡大batch** | 每实验独占单卡48GB，batch开到最大 | mHC-BEV、TopoLoss-SparseWorld |
| **双卡并行搜参** | 卡0跑学习率1e-4，卡1跑5e-5，同时验证 | Diffusion Policy、RT-1-X |
| **数据并行（弃用）** | 梯度同步需PCIe，比单卡还慢 | 不推荐 |

**可视化界面利用**：  
- 用 `wandb` 或 `tensorboard` 在本地浏览器看曲线，远程桌面偶尔用于调试UI交互代码。

---

## 4) 一个月执行路线（主线+副线）

| 阶段 | 时间 | 任务 | 验收输出 |
|---|---|---|---|
| M1 | 第1周 | Phase 1 基线复现（Diffusion Policy 小任务/LIBERO-10）；SparseWorld-TC 基线跑通 | 训练曲线、可复现实验脚本、环境说明 |
| M2 | 第2周 | mHC-residual 接入与消融；TopoLoss 原型接入与梯度检查 | 对比曲线（baseline vs 改造版）、稳定性日志 |
| M3 | 第3周 | 双线小规模对比实验（2-3 seeds） | 指标表、可视化图、失败案例清单 |
| M4 | 第4周 | 固化论文证据链（main0/main1素材归档） | 图表包、结论摘要、下一步实验列表 |

## 5) 本月范围冻结（避免分散）
- 必做：`mHC + Diffusion Policy`、`TopoLoss + SparseWorld-TC`。
- 选做：`mHC + BEVFormer` 仅做最小可跑验证，不作为主输出。
- 延后：`Engram + RT-1-X` 仅保留方案，不进入本月训练排期。

## 6) 夜间算力策略（一个月版）
- 卡0：Phase 1（Diffusion Policy）夜间批量训练/消融。
- 卡1：Phase 2（SparseWorld-TC）夜间批量训练/对比。
- 白天只做：日志复盘、图表整理、脚本修复、下一轮任务下发。

## 7) 与论文嵌入关系（本月）
- `main1.tex`：优先嵌入 Gate0/Gate1 的稳定性指标与 mHC 对比曲线。
- `main0.tex`：优先嵌入 TopoLoss 对结构坍塌抑制的可视化与 Betti 曲线。

**最后一句话**：一个月窗口下，坚持“主线闭环可交代 + 副线论文可落图”，不追全栈铺开。

# phase 1与phase 2作为主攻方向 
**确认：Phase 1（mHC+Diffusion Policy）与 Phase 2（TopoLoss+SparseWorld-TC）是绝对主攻方向。**  
Phase 1 优先追求**“可复现 + 可归因 + 可写”**；Phase 2 优先追求**“闭环可跑通 + 子集趋势有效 + 证据链完整”**。  
**双卡无NVLink策略不变**——卡0跑具身，卡1跑生成，互不干扰（但同卡多进程并行训练默认不做，避免显存争抢与不确定性）。

以下采用 Gate 里程碑制，不再使用逐日刚性排期。

---

## ✅ Phase 1（mHC + Diffusion Policy）建议改为“Gate制”，先MVE再DP
> 目的：降低“LIBERO-90基线复现/耗时/环境”对进度的单点卡死风险，并把 `mve-training-warning/` 直接纳入 Phase 1 证据链。

### Gate0（0.5–1天）：先跑通最小验证（MVE）
- 直接复用你已有的：`面向个人执行/mve-training-warning/`
- 产出物（必须可贴图/可写进短文）：
  - Sinkhorn投影误差曲线（constraint_on/off 对比）
  - 指标聚合表（多seed均值/方差）
- 通过标准：
  - 全流程无NaN/Inf
  - 指标与README预期一致（或差异能解释）

> 备注：Gate0 的结果也可以作为 `main1.tex` 的 pilot figure（训练稳定性诊断），同时为后续 DP 接 mHC 的数值稳定性“兜底”。

### Gate1（1–2天）：Diffusion Policy 基线先用“小任务/小套件”复现
- 不再把 `LIBERO-90 ~85%` 作为第一天硬门槛（过于依赖环境与训练时长）。
- 建议顺序：
  1) 单任务（如 kettle/插拔）或 LIBERO-10 先通训练与评估链路
  2) 再升级到 LIBERO-90（作为后续加分项）

### Gate2（2–3天）：只改 ResBlock 残差（mHC），skip保持不动（便于归因）
- 第一版只做 “mHC-residual”，不做“跳跃连接mHC融合”
- 必做单测/断言：
  - 输出shape一致
  - backward可跑通
  - Sinkhorn行列和误差在阈值内（并记录到日志）

### Gate3（3–7天）：2–3个seed的小规模对比 + 推理步数压缩
- 核心对比（最小可发表证据链）：
  - baseline（8步推理）
  - mHC-residual（8步推理）
  - mHC-residual（4步推理）
- 成功标准建议改为（更稳）：
  - “4步推理不显著劣于 baseline 8步”（或在关键任务上持平/小幅提升）
  - 训练曲线尖峰/不稳定事件减少（配合 Gate0 的诊断指标）

### Gate4（可选）：再做 skip-mHC 融合（如果 Gate3 已经有效）
- 只有在 Gate3 明确有效时才扩改 skip，避免“多改动导致不可归因”。

---

# 🌍 Phase 2（TopoLoss + SparseWorld-TC）关键表述降风险：计算开销与可微性
**目标**：验证拓扑约束能防止 3D 生成中的结构坍塌。**作为 main0.tex 的核心实验，冲刺表征学习/生成模型顶会**。
- Phase 2 的优先证据链建议为：
  1) 拓扑计算可微或存在有效的代理梯度（Surrogate Gradient）。
  2) 视觉对比：Baseline 生成的断裂车道线 vs TopoLoss 生成的连续车道线。
  3) 定量指标：生成数据的 Betti 数分布更接近真实数据。

### Phase 2 Gate（建议）
- G1（2–3天）：SparseWorld-TC 基线在 NuScenes mini 上跑通 forward/backward。
- G2（3–5天）：实现批处理的拓扑签名提取，并验证 $\mathcal{L}_{\text{topo-reg}}$ 的梯度能回传。
- G3（≥1周）：子集对比实验（Baseline vs +TopoLoss），导出 Betti 曲线对比图和 3D 可视化结果。

*(注：原逐日排期表已删除，全面改用上述 Gate 里程碑制，利用主线任务的夜间/碎片时间推进，避免与主线工程冲突。)*

---

# ✅ 成功标准

**Phase 1**：
- [ ] Diffusion Policy + mHC 在小任务或 LIBERO 子集上**验证有效（如4步推理不显著劣于基线8步）**
- [ ] **Gate0（MVE）已在本地RTX4060完成**，图表直接用于撰写 `main1.tex` 和主线汇报。

**Phase 2**：
- [ ] TopoLoss + SparseWorld-TC 在nuScenes生成任务上实现**结构坍塌显著减少**
- [ ] 拓扑损失梯度回传稳定，Betti数曲线收敛

---

# 📮 后续行动

1. **主线优先**：在云服务器上优先配置主线所需的 Docker/环境。
2. **环境隔离**：为 Phase 1 (DP) 和 Phase 2 (SparseWorld-TC) 创建独立的 conda 环境，避免与主线仿真工具链冲突。
3. **资产复用**：将本地跑通的 `mve-training-warning` 脚本直接封装为一个小工具，准备接入主线的 Validator。

你的双5880已就绪，这两把刀——**mHC与TopoLoss**——将在未来三周为你劈开两条独立的顶会通道。