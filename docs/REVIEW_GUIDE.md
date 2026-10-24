# A3 V1.3 代码审阅指南

更新：2026-10-08。四Head已共用训练与exact评估；软件证据见[V1.3验证记录](VALIDATION_V1_3.md)，当前规范见[V1.3架构](ARCHITECTURE_V1_3.md)。本次仅清理旧状态，未重跑测试或放行真实训练。

## 建议阅读顺序

1. 阅读 [完整架构](ARCHITECTURE_V1_3.md)，重点核对第4–10节的信息隔离、四Head、完整标签主训练与概率一致性，以及第11–15节的运行和未完成项。
2. 从 [`A3PCCModel`](../src/a3_ultrasound/models/integrated.py) 跟踪 forward 和 infer_exact。
3. 对照下表检查局部实现与对应测试。
4. 阅读 [2026-10-01验证记录](VALIDATION_V1_3.md)，运行所需测试；反馈请注明 commit、文件、函数、复现条件及预期/实际行为。

## 源码和测试地图

| 环节 | 源码 | 主要测试 |
|---|---|---|
| Batch / metadata / mask | [schema.py](../src/a3_ultrasound/schema.py)、[collate.py](../src/a3_ultrasound/data/collate.py)、[metadata.py](../src/a3_ultrasound/data/metadata.py) | [test_collate.py](../tests/test_collate.py)、[test_metadata.py](../tests/test_metadata.py) |
| ResNet1D / InceptionTime / FCN-1D、temporal adapter | [backbone.py](../src/a3_ultrasound/models/backbone.py) | [test_backbone.py](../tests/test_backbone.py) |
| Query / Prior / indexed Local 与 Pre Readout | [query.py](../src/a3_ultrasound/models/query.py)、[indexed_readout.py](../src/a3_ultrasound/models/indexed_readout.py) | [test_schema_and_readout.py](../tests/test_schema_and_readout.py) |
| 三职责 token | [responsibility_tokens.py](../src/a3_ultrasound/models/responsibility_tokens.py) | [test_responsibility_tokens.py](../tests/test_responsibility_tokens.py) |
| Shared GRU / Local Base / Gate / Correction | [pcc.py](../src/a3_ultrasound/models/pcc.py) | [test_pcc_enumerator.py](../tests/test_pcc_enumerator.py) |
| Local / Local＋Pre / Vanilla Head | [baselines.py](../src/a3_ultrasound/models/baselines.py) | [test_v1_2_architecture.py](../tests/test_v1_2_architecture.py) |
| 四Head集成 / exact enumeration | [integrated.py](../src/a3_ultrasound/models/integrated.py)、[enumerator.py](../src/a3_ultrasound/inference/enumerator.py) | [test_integrated_model.py](../tests/test_integrated_model.py) |
| 等样本监督权重 / local auxiliary loss | [losses.py](../src/a3_ultrasound/models/losses.py) | [test_losses.py](../tests/test_losses.py) |
| 模型配置 / 容量透明度 | [schema.yaml](../configs/model/schema.yaml)、[capacity.py](../src/a3_ultrasound/models/capacity.py) | [test_v1_2_architecture.py](../tests/test_v1_2_architecture.py) |
| 公共训练、完整标签、split、checkpoint、exact评估 | [training.py](../src/a3_ultrasound/training.py)、[run_a3.py](../scripts/run_a3.py) | [test_v1_3_training.py](../tests/test_v1_3_training.py) |

## 网络数据流

```text
raw RF + length_mask
  -> temporal Backbone (pre-pooling)
  -> channel-only Adapter (D_h=64, 保留时间坐标和 mask)
  -> indexed Local / Pre Readout <- metadata 的 Query/Prior 分流
  -> role-specific r_loc / r_ctx / r_gate
  -> 四种可替换 Head：Local | Local＋Pre | Vanilla Chain | Proposed
  -> 同一 conditional-step / exact-enumeration 协议
  -> joint posterior、MAP、逐界面 marginal
```

四种Head是并行比较方案，通过`head_type`选择并共用编码/条件推断/exact接口；`A3PCCModel`默认Proposed，公共工厂支持四种Head。主训练使用真实完整前缀，部署评估使用候选路径枚举，预测不读取标签。

Proposed 的条件 logit 为 `base(r_loc) + sigmoid(gate(r_gate)) * rho * tanh(correction(r_ctx, h_prefix))`。共享 GRU 只编码当前界面之前的 Query/state；Gate 不读 prefix；第一个界面的 Correction 为零。完整公式、默认维度及参数量见架构文档。

## 运行测试

在仓库根目录、项目依赖已安装的 Python 环境中运行：

```powershell
python -m pytest -q
```

聚焦架构检查可运行：

```powershell
python -m pytest -q tests/test_v1_3_training.py tests/test_v1_2_architecture.py tests/test_backbone.py tests/test_integrated_model.py tests/test_pcc_enumerator.py tests/test_losses.py tests/test_responsibility_tokens.py tests/test_schema_and_readout.py
```

环境参考 [`requirements.txt`](../requirements.txt)、[`environment.yml`](../environment.yml) 和 [`pyproject.toml`](../pyproject.toml)。前两者是原登记的 Windows/CUDA 环境清单，未在此次整理中升级或重新验证远端依赖下载；在另一台机器全新安装仍需验证。PyCharm 使用同一解释器，以仓库根目录为 working directory，配置 pytest target 为 `tests`。命令行测试通过不等于已验证 PyCharm GUI。

测试的输入是合成/临时 fixture，运行测试不需要原始实验 RF，也不执行真实监督训练。历史 scripts 中需 `--raw-root`、manifest 或既有证据文件的入口需要另备输入，不能视为下载仓库即可完成的真实实验。

`third_party/Time-Series-Library` 通过 gitlink 固定 revision，来源见 [`.gitmodules`](../.gitmodules)。它是后续 TimesNet baseline 的源码依赖，不属于已完成的 A3 网络集成；核心单元测试不依赖初始化该子模块。需要阅读它时使用 `git submodule update --init --recursive`，不要为此安装整套上游依赖。

## 审阅重点与未完成项

- 检查 padding、有效长度、feature-center 几何和空 Local support 行为。
- 检查 Query/Prior 输入隔离、三职责投影、prefix 是否泄漏当前/未来状态、Gate 是否独立于候选路径。
- 检查 K=1…6 的参数共享、概率归一、mixed-K mask 和 teacher-prefix/exact 一致性。
- 检查四Head的公平输入、完整标签共同cohort、Local Base辅助分派和每样本等权；Local独立Head不重复加Base损失。
- 四Head训练装配已有实现；跨采集manifest的真实数据准备、正式监督split与outer evaluation尚未闭合。
- V1.3完整M1与受影响真实RF门禁仍待专门重验证；2026-10-01正式Backbone×四Head合成smoke仅是已记录的软件检查；旧 scripts/验收合同及 matched-joint 配置保留作历史参考，不能套用旧 PASS 证明当前版本。
- K=4–6 仅是工程支持范围；noncausal Backbone 下的 Pre support 不构成原始波形严格因果裁剪。

## 内容边界

纳入完整 `src/`、`tests/`、`scripts/`、实现配置、依赖清单、架构规范和本指南。真实 RF、真实采集 manifest、研究日志、账号凭据和本机运行产物保留在本地。仓库默认私有，审阅者需由仓库所有者授予访问权限。

没有新增开源许可证；第三方代码继续受各自上游许可证约束。本次整理不作性能、机制有效性或论文创新的肯定裁决。
