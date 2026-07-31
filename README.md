# A3 Layer Recognition

A3 V1.1 的固定界面、多界面结构化状态推断原型。任务从 raw RF 和固定界面结构量出发，经 indexed Local/Pre-interface Readout、scalar Bernoulli PCC 与 exact enumeration，输出联合状态分布、MAP path 和逐界面边缘概率。

## 当前实现范围

- packed 五项 `a3-interface-meta-v3`，内部拆为三项 Query 与三项 Prior view；
- metadata → network API 的正式内存 collate，按每条有效长度派生
  `tau/T` 与 `Delta/T`，按每批试块总厚度派生 `z/zmax`；
- 基于 `pytorch-tcn==1.2.3` 的 noncausal、same-length Backbone adapter，
  按有效前缀长度分组，阻断 padded tail 对短序列边界的卷积泄漏；
- `tsai==1.0.1` 已作为 PyTorch 时序结构候选库加入同一 Conda 环境；
  当前尚未把其 InceptionTime/ResNet trunk 接入正式 Backbone API；
- Local query-conditioned / Pre query-free indexed masked-attention Readout；
- 七套来源投影的 ResponsibilityTokenProjector；
- ordered masked Prefix MLP、path-independent Gate 与 scalar Bernoulli PCC；
- K=2/3 deterministic exact enumeration utility；
- synthetic mixed-`K` integrated conditional model，统一输出 support、attention、
  三职责 token 与 PCC conditional logits；
- matched-joint 基线的冻结结构规格与解析容量匹配工具：参考配置选择
  `joint_hidden_dim=103`，PCC/Joint head 分别为 28,579/28,527 参数，
  差异 0.182%，且宽度搜索不读取标签或验证分数；
- 94 项合成结构/数值单元测试，以及 RTX 4060 上单个 mixed-`K`
  integrated forward/backward CUDA smoke；
- M1-01..13 的确定性软件验收记录与 M1-GATE。

真实 RF 文件 loader、真实 specimen-level split、matched-joint 训练实现、
主训练、指标结果和任何研究结论均不在当前范围内。M1-GATE 只放行
M2-01 的 label-free batch construction，不放行监督训练或性能主张。

## Environment

使用项目的 Conda 解释器并安装锁定依赖：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

同一环境可由 `environment.yml` 重建。PyTorch 必须使用清单指定的 CUDA 轮子；CPU-only Torch 只可用于 API 测试，不能替代 GPU smoke。

## Layout

```text
src/a3_ultrasound/
  schema.py              # network-facing tensor whitelist
  data/collate.py         # validated metadata/in-memory trace -> InterfaceBatch
  models/                # TCN Backbone, Query/Readout/PCC and integrated forward
  inference/             # exact paths and order controls
configs/                 # schema only; no unfounded numeric freeze
tests/                   # synthetic structural checks
```

## M1 验证入口

```powershell
python -m pytest -q
python scripts/compute_capacity_match.py --output <RESULTS>/m1_11_12_capacity_match.json
python scripts/m1_integrated_gpu_smoke.py --output <RESULTS>/m1_13_integrated_gpu_smoke.json
```

验收 JSON 由 `scripts/build_m1_acceptance.py` 从完整 JUnit、既有 M1
证据与上述两个结果确定性生成；其边界字段明确保持
`real_specimen_manifest_validated=false`、
`matched_joint_training_implemented=false` 和
`performance_claims_released=false`。
