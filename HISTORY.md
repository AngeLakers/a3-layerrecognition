# A3 版本历史

每节记录该版开发时要解决的问题、完成的工程改动及对应原文。历史配置和实验记录按各自的版本及适用范围阅读。当前入口见 [v1.8 实现说明](docs/V18_IMPLEMENTATION.md)。

## v1.8 · 单通道 RF、逐界面读头与配对对照

**当时的想法。** 固定单条 RF 的输入边界，把时序骨干和界面判读分开比较。GRU 比较检验已判界面的候选状态是否能改变波形记忆，M1 比较检验两臂共享声时坐标时，编码后的限窗和高斯偏置能带来什么变化。

**更新。** 接入 PatchTST 的时序 token，与 ResNet1D 共用逐界面接口。实现状态条件 GRU 和只写波形的 GRU 对照。M1 在同一个 ResNet1D 上共用相对声时打分，以全时域读取对照局部窗及高斯偏置。2026-11-26 将 v1.8 实现、历史分支和版本文档合到 `master`，并将 Python 包版本标为 `1.8.0`。[v1.8 标签](https://github.com/AngeLakers/a3-layerrecognition/tree/v1.8) · [GRU/PatchTST 比较快照](https://github.com/AngeLakers/a3-layerrecognition/tree/31682de) · [M1 读头改动](https://github.com/AngeLakers/a3-layerrecognition/commit/17bd3fd)

## v1.7.1 · 固定输入，拆开骨干、读头和组合留出

**当时的想法。** 用固定训练预算和相同面板比较各骨干的 LP 与全局读头，单独看局部读头支持范围、等复检量下的置信度分流、以及留出一种缺陷组合时的表现。

**更新。** 补齐跨骨干及全时域 Local 的实验装配、基于联合 MAP 的复检评价、组合留出的两臂抽样及逐格授权收据。[原执行计划](docs/V171_EXECUTION_PLAN.md) · [代码快照](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.7.1-fixed40)

## v1.7 · 全时域 G1D 参照

**当时的想法。** 给逐界面 LP 增加同系列时序编码的全时域参照，比较整组状态的判读表现，为后续读头范围和置信度实验提供基础。

**更新。** 登记 G1D 架构和配套 runner，固定源码字节以追溯当时所用实现。[G1D 架构](scripts/a3_v17_g1d_model.py) · [运行器](scripts/a3_v17_g1d_runner_full.py) · [代码快照](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.7-baselines)

## v1.6 · 深层波形分离、训练接入与联合评价

**当时的想法。** 先验证浅层回波混响分解能否为深层界面提供可辨识的附加信号。用训练折拟合固定变换，在与原 RF 对齐的第二通道上比较，保留单通道参照与独立的启用条件。

**更新。** R3 加入 M3 残差和深层成分、D2 分组判据、按板评估及条件后验温度缩放。后续 runner 接入冻结时基、M6 校验与折绑定的在线 M3，加入学习率 warmup/cosine 和 guarded 验证导出。合并到主线时延续较新的数据行视图、v1.8 单 RF 时基、公共学习率函数及配置校验，历史 runner 快照仍可检出。[原实现说明及补记](docs/V16_R3_IMPLEMENTATION.md) · [R3 开发快照](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.6-dev) · [runner 快照](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.6-runner) · [PR #1](https://github.com/AngeLakers/a3-layerrecognition/pull/1)

## v1.5 系列 · 稳定训练输入和执行条件

| 版本 | 当时的想法和实现更新 | 原始记录 |
| --- | --- | --- |
| v1.5 | 在 v1.4 冻结协议基础上检验抑制策略、前缀扰动及固定物理特征，分别登记执行合同。 | [冻结快照](https://github.com/AngeLakers/a3-layerrecognition/tree/v1.5-freeze) |
| v1.5.1 | 在计算效率和填充隔离之间取得一致性，按有效长度分组执行 GroupNorm 骨干。 | [改动提交](https://github.com/AngeLakers/a3-layerrecognition/commit/27fa6e5) |
| v1.5.2 | 用始波与底波标定逐板时基，将局部窗口和薄板输入前缀固定到可复现配置。 | [时基校准报告](reports/v152_timebase_calibration.md) · [代码快照](https://github.com/AngeLakers/a3-layerrecognition/tree/v1.5.2) |
| v1.5.3 | 在确定性约束下比较执行加速，固定 TF32、batch16 和 FP32 精度配置，保留 v1.5.2 兼容入口。 | [速度补测报告](reports/v153_speed_final.md) · [代码快照](https://github.com/AngeLakers/a3-layerrecognition/tree/v1.5.3) |

## v1.1 至 v1.4 · 从结构化原型到冻结训练

| 版本 | 当时的想法和实现更新 | 原始记录 |
| --- | --- | --- |
| v1.1 | 利用已知层序和界面位置，从原始 RF 读取各界面证据，通过条件 Bernoulli 和精确枚举得到联合状态。 | [原型说明](https://github.com/AngeLakers/a3-layerrecognition/blob/dc691d2/README.md) |
| v1.2 | 通过保留时间的骨干和统一通道适配器装配可变界面的结构化读头，明确 Local、Pre、条件更新和推断的接口。 | [架构原文](docs/ARCHITECTURE_V1_2.md) · [当时的验证](docs/VALIDATION.md) |
| v1.3 | 让四类读头共享数据、损失、评估与训练入口，补齐真实 RF 流式加载、标签工具、权重与 OOF 汇总。 | [发布快照](https://github.com/AngeLakers/a3-layerrecognition/tree/113a02d) · [架构原文](https://github.com/AngeLakers/a3-layerrecognition/blob/113a02d/docs/ARCHITECTURE_V1_3.md) · [当时的验证](docs/VALIDATION_V1_3.md) |
| v1.4 | 将工作点从最深界面指标转向整组判对，加入完整正常点的误报护栏、逐轮正常点重抽及 checkpoint 诊断。 | [冻结协议](docs/V1_4_FREEZE.md) · [代码快照](https://github.com/AngeLakers/a3-layerrecognition/tree/v1.4-freeze) |

## 已合并分支的恢复点

`master` 是当前唯一开发分支。下列标签固定合并前各远端分支的最后一个提交，可用于查看当时的 README、配置和代码。

| 原分支 | 提交 | 归档标签 |
| --- | --- | --- |
| `v1.5`、`v1.5.3` | `d72dc25` | [v1.5.3](https://github.com/AngeLakers/a3-layerrecognition/tree/v1.5.3) |
| `v1.6-dev` | `f93b796` | [archive/v1.6-dev](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.6-dev) |
| `claude/project-thread-lp54lp` | `9a246d7` | [archive/v1.6-runner](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.6-runner) |
| `v1.7-baselines` | `7c90714` | [archive/v1.7-baselines](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.7-baselines) |
| `codex/v171-fixed40` | `ee005bd` | [archive/v1.7.1-fixed40](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.7.1-fixed40) |
| `codex/v18-state-gru` | `ad25db0` | [archive/v1.8-state-gru](https://github.com/AngeLakers/a3-layerrecognition/tree/archive/v1.8-state-gru) |
