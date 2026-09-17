# A3 Layer Recognition

## V1.2 代码审阅入口

- [审阅指南](docs/REVIEW_GUIDE.md)：阅读顺序、源码与测试对应、运行方式和待完成项。
- [完整网络架构与 API 规范](docs/ARCHITECTURE_V1_2.md)：含计算图、张量合同、三类 Head、损失、推理和容量说明。
- [本次验证记录](docs/VALIDATION.md)：此次上传对应的实际检查结果。

本仓库用于审阅当前 V1.2 工程实现。全部现有源码、配置 schema、脚本和测试一并提供；原始 RF、真实采集 manifest、模型权重和内部研究过程资料未纳入。Proposed 已有集成模型，Independent/Vanilla 当前为 Head 级实现，完整训练装配尚未完成。

A3 V1.2 的固定界面、多界面结构化状态推断原型。任务从 raw RF 和固定界面结构量出发，经成熟 pre-pooling temporal Backbone、统一 channel-only adapter、indexed Local/Pre-interface Readout、shared variable-K structured Head 与 exact enumeration，输出联合状态分布、MAP path 和逐界面边缘概率。

## V1.2 当前实现范围

- 正式 Backbone 候选为 tsai ResNet1D、InceptionTime、FCN-1D；当前 `pytorch-tcn` wrapper 明确为 smoke/debug-only，不用于后续正式训练；
- 统一 `TemporalBackboneOutput` 与 `CommonTemporalBackboneAdapter`：将不同 `D_backbone` 映射到 `D_h=64`，保持原生 temporal length、`feature_centers` 和 `feature_mask`；
- ResNet1D/FCN-1D 可选择 pre-pooling stage；InceptionTime 当前暴露 final-pre-pool；所有现有正式 wrappers 都是 time-preserving；
- Local-token diagnostics 记录有效界面窗口的 token 数分布，空 Local support fail-fast；
- packed 五项 `a3-interface-meta-v3` 和 Query/Prior 分流不变；`K_cap` 不进入 Query；
- Local query-conditioned / Pre query-free Readout、三职责 token、path-independent Gate 和 Correction 输入隔离继续保持；
- Prefix 使用共享 GRUCell，不含固定 `Kmax×feature_dim` 权重；工程 guard 为 `K_cap=6`，真实试样声明仍限 K=2/3；
- Local Base 默认 `32→32→1` MLP，Linear 保留为消融；
- Head baseline 为 Independent、Vanilla Classifier Chain、Proposed；Vanilla classifier 只读 `r_loc/r_ctx/prefix_hidden`，不直接读 Query 或 `r_gate`；matched joint 只保留为 optional historical small-K spec；
- 三类 Head 均采用同构但独立参数的 Local Base；Independent 以它作为主输出，Vanilla/Proposed 以同一 `lambda_local` 规则施加 Local auxiliary loss；
- 监督 loss 先按每样本可靠标签数 `N_i^sup` 归一，再在有监督样本间等权平均；
- inference enumerator 已解除 fresh-import 循环，并可对 K≤6 做 exact path enumeration。

V1.2 当前只有结构/API 与本地测试证据。V1.1 的 M1/M2-LF JSON、JUnit、日志和 source hash 原样保留为历史结果；受影响 gate 必须后续重跑，不能据此声称 V1.2 当前门禁通过。真实监督训练、性能与机制结论继续关闭。

2026-08-07 外部审核反馈修订后的 V1.2 本地完整回归：`157 passed`，0 failure/error/skip；4 个 warning 来自 tsai/PyTorch JIT deprecation。该结果不是新的 M1 acceptance、GPU smoke 或 real-RF gate。

## V1.1 历史实现范围（待归档）

- packed 五项 `a3-interface-meta-v3`，内部拆为三项 Query 与三项 Prior view；
- metadata → network API 的正式内存 collate，按每条有效长度派生
  `tau/T` 与 `Delta/T`，按每批试块总厚度派生 `z/zmax`；
- 严格 signed-integer TXT 的真实 RF label-free loader：只读取确定性的
  五点空间抽样，使用一项批级固定幅值分母，拒绝逐条归一化、路径越界、
  非整数、缺文件与长度不符，并且不把坐标或标签写入网络 batch；
- 基于 `pytorch-tcn==1.2.3` 的 noncausal、same-length Backbone adapter，
  按有效前缀长度分组，阻断 padded tail 对短序列边界的卷积泄漏；
- `tsai==1.0.1` 已作为 PyTorch 时序结构候选库加入同一 Conda 环境；
  当前尚未把其 InceptionTime/ResNet trunk 接入正式 Backbone API；
- THUML `Time-Series-Library` 已作为固定 commit 的本地源码依赖加入
  `third_party/Time-Series-Library`，后续用于 TimesNet 完整分类器基线；
  不安装其整套上游 requirements，也不把 TimesNet 混入 PCC 运行时依赖；
- Local query-conditioned / Pre query-free indexed masked-attention Readout；
- 七套来源投影的 ResponsibilityTokenProjector；
- ordered masked Prefix MLP、path-independent Gate 与 scalar Bernoulli PCC；
- K=2/3 deterministic exact enumeration utility；
- synthetic mixed-`K` integrated conditional model，统一输出 support、attention、
  三职责 token 与 PCC conditional logits；
- matched-joint 基线的冻结结构规格与解析容量匹配工具：参考配置选择
  `joint_hidden_dim=103`，PCC/Joint head 分别为 28,579/28,527 参数，
  差异 0.182%，且宽度搜索不读取标签或验证分数；
- 136 项数据/结构/数值/CLI 单元测试，以及 RTX 4060 上 synthetic mixed-`K`
  integrated smoke、真实 RF label-free forward/support 和 target-free autograd；
- M1-01..13 的确定性软件验收记录与 M1-GATE。

真实 RF label-free forward 与 target-free autograd 已闭合工程链路；真实
specimen-level split、matched-joint 训练实现、主训练、指标结果和任何研究结论
仍不在当前范围内。现有 M2 证据不放行监督训练或性能主张。

## Environment

使用项目的 Conda 解释器并安装锁定依赖：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

同一环境可由 `environment.yml` 重建。PyTorch 必须使用清单指定的 CUDA 轮子；CPU-only Torch 只可用于 API 测试，不能替代 GPU smoke。

PyCharm Run/Debug 选择已有 `aislee-a3-layercongition` Conda interpreter。
各 runner 会从项目 `src/` 解析 package，无需额外设置 `PYTHONPATH`；工作目录、
manifest 与输出路径必须作为显式参数提供，和 Codex CLI 使用同一合同。

THUML 源码依赖使用 Git submodule 固定：

```powershell
git submodule update --init --recursive
```

固定 revision 记录在 `pyproject.toml` 的
`tool.a3.third-party.time-series-library`。TimesNet 接入前仍须单独完成 import、
分类接口、split、资源与许可证验收；本次仅加入源码，不声称兼容或性能已验证。

## Layout

```text
src/a3_ultrasound/
  schema.py              # network-facing tensor whitelist
  data/collate.py         # validated metadata/in-memory trace -> InterfaceBatch
  data/real_rf.py         # strict selected TXT -> label-free InterfaceBatch
  diagnostics.py          # target-free autograd probe and gradient audit
  models/                # TCN Backbone, Query/Readout/PCC and integrated forward
  inference/             # exact paths and order controls
configs/                 # schema only; no unfounded numeric freeze
tests/                   # synthetic structural checks
```

## V1.1 M1 历史验证入口

以下 runner 绑定旧 evidence contract，当前不应生成或覆盖 V1.2 acceptance。V1.2 M1 重验证需使用新的 schema/version/output 文件名。

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

## M2-LF-01 无标签真实 RF 反向诊断

在 PyCharm 中以 `scripts/m2_label_free_real_rf_backward_diagnostic.py` 为入口，
或在同一 Conda 环境用 CLI 运行：

```powershell
python scripts/m2_label_free_real_rf_backward_diagnostic.py `
  --raw-root <RAW_ROOT> `
  --manifest <ACQUISITION_MANIFEST> `
  --metadata-schema <METADATA_SCHEMA> `
  --output <RESULT_JSON> `
  --log <RUN_LOG>
```

该入口只接受 label-free batch，用固定无 target 数值 probe 做一次 backward；
不会创建 optimizer、更新参数或保存 checkpoint。它只证明真实输入上的 autograd
链路可达，不是训练 loss、性能指标或传播机制证据。
