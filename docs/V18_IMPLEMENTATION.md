# A3 v1.8 实现说明

v1.8 的时序骨干接收单通道原始 RF 和有效长度掩码，输出保留原采样坐标的时序特征。读头使用已知界面位置和标定声时。推断按有效界面枚举候选状态，输出联合概率、MAP 和边缘概率。[模型工厂](../src/a3_ultrasound/training.py) · [集成模型](../src/a3_ultrasound/models/integrated.py)

## 读头与对照

| 用途 | 配置 | 实现 |
| --- | --- | --- |
| LP 参照 | `head_type=local_pre_independent` | 结合 Local 和 Pre 读取，逐界面估计状态概率 |
| 状态条件 GRU | `head_type=state_conditioned_gru` | 先给当前界面打分，再按该界面的候选状态选择波形写入，供后续界面读取 |
| 波形 GRU 对照 | `head_type=waveform_only_gru` | 共享同一种波形写入，用于单独比较状态条件写入 |
| M1 全时域与局部窗对照 | `backbone=resnet1d`、`head_type=local_independent`、`relative_time_attention=true` | 两臂以同一界面声时作为相对坐标。全时域用 `readout_scope=full_waveform`，局部窗用 `readout_scope=indexed` 并传入高斯先验参数 |

LP 和两种 GRU 可结合 ResNet1D 或 PatchTST 使用。PatchTST 提取分类头之前的 patch token，使用 16 点 patch、4 点步长并保留 patch 中心坐标。M1 使用独立界面读头，比较编码后限窗与高斯偏置的合用效果。[LP 实现](../src/a3_ultrasound/models/baselines.py) · [GRU 实现](../src/a3_ultrasound/models/state_gru.py) · [PatchTST 实现](../src/a3_ultrasound/models/patchtst.py) · [索引读取](../src/a3_ultrasound/models/indexed_readout.py)

## 运行与检查点

- v1.8 模型在 `model` 中显式设置 `architecture_version: a3-v1.8`、`backbone_norm: group8`、`query_view: position2`、`input_channels: 1`。用仓库内的合成配置检查端到端装配和检查点重载。[示例配置](../configs/experiment/v1_8_gru_synthetic.yaml) · [训练入口](../scripts/run_a3.py) · [GRU/PatchTST 测试](../tests/test_v18_state_gru.py) · [M1 测试](../tests/test_v18_m1.py)
- M1 使用与已存检查点一致的 `readout_scope`、`relative_time_attention` 和先验配置。历史模型按各自的架构标识、配置及提交读取。[版本历史](../HISTORY.md)
- 复现真实实验时，依据 A3 研究资料库的实际运行清单匹配源码、输入、配置和产物。归档的 GRU/PatchTST 比较对应提交 `31682de`，M1 在此基础上增加相对声时读头。版本标签标识本次工程整合。[GRU/PatchTST 源码快照](https://github.com/AngeLakers/a3-layerrecognition/tree/31682de) · [M1 模型改动](https://github.com/AngeLakers/a3-layerrecognition/commit/17bd3fd)
