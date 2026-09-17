# A3 V1.2 架构与 API 技术规范

> 版本：V1.2，2026-08-07
> 状态：用户批准的结构迁移与本地完整回归已完成；M1/M2-LF 受影响门禁待重跑；真实监督训练仍为 `GATED`

> 审阅副本：2026-09-17 从研究工作区 `a3/docs/ARCHITECTURE_V1_2.md` 同步完整正文；仅调整顶部导航，章节正文保持不变。研究合同、原始数据和内部评审记录留在研究工作区。审阅入口见 [REVIEW_GUIDE.md](REVIEW_GUIDE.md)。

## 1. 定位与声明边界

A3 V1.2 采用“成熟可替换 temporal Backbone + 统一 temporal-feature interface + interface-aware indexed Readout + variable-K propagation-conditioned structured Head”。Backbone、classifier-chain 分解和 exact enumeration 都是已有工具；当前任务专用贡献候选集中在界面证据读取、传播前上下文、共享前缀状态编码以及受限条件修正的职责组合。

这里的 `propagation-conditioned` 只表示：按已知浅到深传播顺序，使用前序 observation context 与候选上游界面状态条件化下游界面推断。它是 directed inductive bias，不是 wave equation、transfer matrix、differentiable forward model，也不构成严格物理因果建模主张。

当前代码与单元测试只构成结构/API 证据。真实试样尚未形成监督训练与 outer evaluation 闭环，不能据此声称性能、泛化、传播机制或论文创新成立。

## 2. 主计算图

~~~text
raw RF + length_mask
→ mature temporal Backbone（pre-pooling feature stage）
→ TemporalBackboneOutput(features, feature_centers, feature_mask)
→ CommonTemporalBackboneAdapter(D_backbone → D_h=64；不改变 L/坐标/mask)
→ Query/Prior 分流
→ query-conditioned Local Readout + query-free Pre Readout
→ r_loc / r_ctx / r_gate
→ local base + path-independent gate × bounded(context + prefix correction)
→ p(S_k | x, S_<k)
→ K≤6 exact enumeration
→ joint path posterior / MAP / interface marginals
~~~

## 3. Backbone 与统一 temporal interface

### 3.1 角色

- 正式训练候选：ResNet1D、InceptionTime、FCN-1D。
- 当前 TCN：只用于 smoke、debug 和轻量 reference，不进入后续正式训练。
- A3 只读取 global pooling/classifier 之前的 temporal feature map。
- 当前正式 wrapper 均为 time-preserving；ResNet1D/FCN-1D 可选 `stage1/stage2/stage3/final_pre_pool`，InceptionTime 当前只暴露 `final_pre_pool`。
- 后续若接入 downsampled/asymmetric wrapper，必须逐层计算真实 `feature_centers`；不得用插值伪造原始采样级分辨率。

### 3.2 输出合同

~~~text
TemporalBackboneOutput:
  features            float [B,L,D_backbone]
  feature_centers     float [L] or [B,L]
  feature_mask        bool  [B,L]
  effective_stride    optional
  theoretical_rf_size optional
~~~

`CommonTemporalBackboneAdapter` 只做通道投影：

~~~text
[B,L,D_backbone] → [B,L,64]
~~~

它必须保持 `L`、`feature_centers`、`feature_mask` 和可用几何诊断不变。不同 Backbone 可以有不同原生 `L`，但后端维度固定为 `D_h=64`。主配置不把不同 Backbone 插值到统一 canonical `T'` 网格；跨 Backbone 公平性依靠各 wrapper 真实传播的 `feature_centers`、同一 Readout 代码与逐 Backbone Local-token resolution 审计闭合。

### 3.3 Local-token resolution 诊断

对每个有效界面记录 Local support 的实际 feature-token 数，并汇总：

- min / mean / median / max；
- `fraction_local_tokens_le_1`；
- `fraction_local_tokens_le_2`。

`N_loc=0` 继续 fail-fast；`N_loc≤2` 当前只作为诊断，不冻结人为阈值。优先选择更早 stage、降低 downsampling 或更换 time-preserving Backbone，不默认插值。

## 4. Query、Prior 与 Readout

packed `a3-interface-meta-v3` 保持五字段：

~~~text
[k/K_i, tau/T_i, z/zmax, Delta_minus/T_i, Delta_plus/T_i]
~~~

- Query 只读 `[k/K_i, tau/T_i, z/zmax]`；不使用 LayerNorm、absolute-interface-ID、`Kmax` 或 `K_cap`。
- Prior 只读 `[tau/T_i, Delta_minus/T_i, Delta_plus/T_i]`，用于 Local/Pre support 与 Gaussian local prior。
- Local Readout 为 query-conditioned；空 support fail-fast，不拥有 `empty_context` 参数。
- Pre Readout 的 attention scorer 为 query-free；空 support 使用独立可学习 `empty_context`。
- noncausal Backbone 下的 support 是 feature-center indexed support，不是 raw-RF strict causal crop。

## 5. Role-Specific Tokenization（三职责 token）

V1.1 已确认的职责隔离继续有效：

- `r_loc` 只读 Local evidence + Query，只服务 Local Base；
- `r_ctx` 只读 Pre context + Query，不直接读取 Local evidence；
- `r_gate` 可读 Local/Pre/Query，但只服务 gate；
- 七套 source projection、三套 bias 和三套 LayerNorm 相互独立；
- invalid interface slot 在投影与归一化后归零。

论文、框架图与项目说明统一使用 `Role-Specific Tokenization`、`Role-Specific Feature Projection` 或“三职责表征”。实现类名 `ResponsibilityTokenProjector` 为兼容既有代码与测试保留；当前模块没有竞争权重、expert assignment 或 token dispatch，因此不得称为 routing/mixture 机制。

## 6. Variable-K propagation-conditioned Head

### 6.1 Shared Prefix GRUCell

V1.1 固定 `Kmax=3` 的 flatten Prefix MLP 已移除。V1.2 使用一套共享参数：

\[
h_0=0,\qquad h_j=\operatorname{GRUCell}([q_j;E_s(s_j)],h_{j-1}).
\]

预测第 `k` 个界面时只使用 `j<k` 的有效 Query/state。padding 不更新 hidden；第一界面 `prefix_hidden=0`。

### 6.2 条件 logit

\[
\eta_k^\pi=b(r_k^{loc})+
\sigma(G(r_k^{gate}))\cdot
\rho\tanh C(r_k^{ctx},h_{<k}^{\pi}).
\]

- Local Base 默认 `32→32→1` MLP，线性模式保留为消融。
- Gate 不读取 prefix，因此同一样本/界面的所有候选路径共享 gate。
- Correction 只直接读取 `r_ctx` 与 prefix hidden，不读取 `r_loc`。
- 第一界面 correction 强制为 0；Correction 最后一层零初始化。
- `rho` 是 Validation 候选 `{1,2,3}`，不是已冻结性能最优值。

所有 Head 参数跨样本、界面位置和 K 共享；`K_cap=6` 只用于 padding/schema/validation/exact-enumeration guard，不进入 Query 或权重维度。

## 7. 推理与实证范围

- 二分类 exact enumeration 路径数为 `2^K`；当前工程支持 `1≤K≤6`，K=6 为 64 条路径。
- Exact enumeration 是 0 参数确定性推理工具，不是 flat joint Head。
- 当前真实试样只允许验证 K=2/3；K=4–6 仅能证明实现可运行，不能宣称 unseen-K 泛化。

## 8. Head baseline 层级

正式主对照顺序改为：

1. Independent Head：以 Local Base `b_k=B(r_k^{loc})` 作为主输出，不读 prefix/context；
2. Vanilla Classifier Chain：用共享 Prefix GRU 将前缀 `[q_j;E(s_j)]_{j<k}` 编为 `h_{<k}`；classifier **只**读取 `concat(r_loc,r_ctx,h_{<k})`，不再直接读取 Query 或 `r_gate`；
3. Proposed Propagation-Conditioned Head：使用本规范第 6 节结构。

flat 4/8-class joint 仅为有精力时的 small-K 可选 baseline。V1.1 的 exact capacity-matched joint 规格保留为历史设计，不再是主假设或必做工程。

三类模型各自持有独立训练参数、但 Local Base 架构相同；不跨模型共享已训练权重。Vanilla 和 Proposed 都显式产出 `b_k`，供同一 Local auxiliary supervision 使用；Independent 的 `b_k` 已是其主预测，不能重复叠加 auxiliary loss。

论文图与架构图必须从 shared front end 画出三个**并行可替换 Head**，不得画成 `Independent → Vanilla → Proposed` 串联数据流。三者的主推理均使用同一 exact-enumerator contract；greedy decoding 只能作为单列诊断，不得只给 Vanilla 使用 greedy、只给 Proposed 使用 exact。

## 9. 监督损失与样本权重

对第 `i` 个具有至少一个可靠标签的 A-scan：

\[
N_i^{sup}=\sum_k m^{int}_{ik}m^{label}_{ik},\qquad
L_i=\frac{\sum_k \ell_{ik}m^{int}_{ik}m^{label}_{ik}}{N_i^{sup}}.
\]

batch loss 只在有监督样本上对 `L_i` 做算术平均，因此每个有监督样本等权；部分标签不会因除以真实 `K_i` 而被低估。无任何可靠标签的 batch 不允许进入该 loss。

Independent 使用 `L_{main}=BCE(b,y)`。Vanilla 与 Proposed 使用相同的：

\[
L=L_{main}+\lambda_{local}L_{local},\qquad
L_{local}=BCE(b,y),
\]

且每个 BCE 都按同一 `N_i^{sup}` 规则归一。`lambda_local` 只能由 inner Validation 在共同候选 `{0.1,0.3,1.0}` 中选择，并且 Vanilla 与 Proposed 在同一比较中必须取相同值；当前代码默认值 `0.3` 仅为工程 reference，不是正式冻结的最优值。

## 10. 当前软件验收边界

V1.2 必须覆盖：

- fresh-process enumerator import；
- 正式 Backbone pre-pooling/stage、padding isolation、时间坐标和统一 adapter；
- Local-token diagnostics 与空 Local support fail-fast；
- K=1..6 shared-prefix、mask、第一界面 correction、参数 shape 不随 K 变化；
- K=6 exact enumeration 64 路径及归一化；
- Gate/Correction/Local Base 职责不变量；
- Independent/Vanilla/Proposed 三类 Head 可被同一枚举器调用；
- `N_i^sup` 归一和等监督样本权重。

V1.1 M1/M2-LF 结果仍可用于说明历史实现曾通过当时的精确门禁，但其 source/config hash 不代表 V1.2。受影响的 M1、real-RF forward/autograd、runtime/acceptance 必须在后续显式重跑后才能成为 V1.2 当前证据。

## 11. 代码结构与接口细节

实现真源为 `a3-layerrecognition`。下表使用仓库相对路径，以便 PyCharm 和命令行定位同一份实现；不把本机目录、数据位置或任何样本身份写入模型合同。

| 区域 | 主要实现 | 代码职责 |
|---|---|---|
| Batch schema | `src/a3_ultrasound/schema.py` | `InterfaceBatch`：`x_raw [B,T]`、`length_mask [B,T]`、`interface_meta [B,K_cap,5]`、`interface_mask [B,K_cap]`，以及可选的 `y/label_mask` |
| Collate | `src/a3_ultrasound/data/collate.py` | 从受验证 manifest 派生五字段、连续 prefix mask 与 padding；默认 `K_cap=6`，真实 manifest 仍只接受 `K∈{2,3}` |
| Temporal feature | `src/a3_ultrasound/models/backbone.py` | `TemporalBackboneOutput`、time-preserving formal wrapper、smoke-only TCN 和 channel-only adapter |
| Query / Readout | `models/query.py`、`models/indexed_readout.py` | Query/Prior 分流、window/support、Local/Pre masked attention 和 Local-token diagnostics |
| 职责与 Head | `models/responsibility_tokens.py`、`models/pcc.py` | 七投影 role-specific tokenization、Local Base、shared GRUCell、Gate 和 bounded Correction |
| 对照 Head | `models/baselines.py` | `IndependentConditionalStep` 与 `VanillaClassifierChainStep`；与 Proposed 共用 conditional-step 协议 |
| 集成与推理 | `models/integrated.py`、`inference/enumerator.py` | `A3PCCModel` 装配 Proposed 路径；泛化的 log-space exact enumeration 与 mask-aware mixed-K 输出 |
| Loss / 配置 | `models/losses.py`、`configs/model/schema.yaml` | `N_i^{sup}` 等权监督语义和 V1.2 不变量配置 |

### 11.1 Batch、metadata 与 mask

`length_mask` 和 `interface_mask` 都必须是连续有效前缀，且有效界面按浅到深排序。网络侧仅接收下列 packed metadata：

```text
[k/K_i, tau/T_i, z/zmax, Delta_minus/T_i, Delta_plus/T_i]
```

`QueryEncoder` 只读取前三项；window、hard support 与 Local Gaussian prior 只由后面定义的 Prior view 计算。`K_cap`、绝对界面 ID、样本/扫描身份、文件位置和标签来源均不进入 learned feature。`label_mask` 表示可靠监督的可用性，不能替代 `interface_mask`。

### 11.2 Backbone、时间坐标与 Local evidence

当前 wrapper 按 valid length 分组，在编码前裁去 batch padding，编码后才把 feature padding 补回。因此带 bias 的 temporal 卷积不会把短 trace 的零尾部传播到有效特征。每个 wrapper 当前强制 `L=T_i`；若未来 trunk 输出 `L<T_i`，会报错而不是伪造采样级时间轴。

由 normalized metadata 反解的每个界面窗口为：

\[
\tau_{ik}=T_i(\tau/T_i)_{ik},\qquad
a_{ik}=\tau_{ik}-T_i(\Delta^-/T_i)_{ik},\qquad
b_{ik}=\tau_{ik}+T_i(\Delta^+/T_i)_{ik}.
\]

`build_indexed_supports` 在真实 `feature_centers` 上构造 `M_loc` 与 `M_pre`。Local 的 hard support 为空立即失败；Pre 为空时才使用其独有的 learnable empty context。`summarize_local_tokens` 会保存每个有效窗口的 token count 以及 min/mean/median/max、`≤1` 和 `≤2` 比例，用于后续选择 Backbone stage，而不是自动插值或静默降级。

### 11.3 Head protocol 与 exact inference

三种 Head 都遵守同一条件调用协议：

```python
step(r_loc, r_ctx, r_gate, query, prefix_states, interface_mask) -> conditional logits
```

`A3PCCModel` 当前只装配 Proposed Head；Independent 和 Vanilla 已实现为可替换 step，但尚没有完整的 end-to-end baseline wrapper。训练时使用 truth prefix；`infer_exact` 枚举完整路径并在 log-space 用 `logsumexp` 归一。padding 不进入路径概率；无效输出 slot 的 marginal 为零、MAP state 为 `-1`。

### 11.4 V1.2 reference 尺寸与容量

当前 reference 使用 `D_h=64`、Query `3→32→32`、attention `64`、三类 token `32`、state embedding `2×32`、Prefix GRUCell `64→32`、Local Base `32→32→1`、Correction `64→64→1`、Gate `32→64→1`。当前构造 reference 取 `rho=1.0`；正式训练只允许在 `{1,2,3}` 中以 inner Validation 选择。

| 组件 | 参数量 |
|---|---:|
| Adapter（formal `D_backbone=128`） | 8,256 |
| Query encoder | 1,184 |
| Local / Pre Readout | 6,272 / 4,288 |
| Responsibility projector | 11,552 |
| Shared interface tower（含 Adapter，不含 Backbone/Head） | 31,552 |
| Local Base | 1,089 |
| Prefix state embedding + GRUCell | 9,472 |
| Vanilla classifier | 6,273 |
| Proposed Correction / Gate | 4,225 / 2,177 |
| Independent / Vanilla / Proposed Head | 1,089 / 16,834 / 16,963 |

注册环境中以默认 `final_pre_pool` wrapper 构造的总参数量如下。它们是透明度记录，不是参数量匹配约束，也不是性能证据：

| Backbone | Backbone | Independent total | Vanilla total | Proposed total |
|---|---:|---:|---:|---:|
| ResNet1D | 478,208 | 510,849 | 526,594 | 526,723 |
| InceptionTime | 388,352 | 420,993 | 436,738 | 436,867 |
| FCN-1D | 264,064 | 296,705 | 312,450 | 312,579 |

`head_baseline_parameter_counts` 计算三类 Head 的解析容量；单元测试将其与实际 `nn.Module` 参数数核对。独立与 Vanilla 的完整 end-to-end wrapper 尚未实现，表中的 total 由已实现的相同 Backbone/interface tower 与相应 Head 相加得到。

### 11.5 回归覆盖与已知未闭合项

V1.2 单元测试分别覆盖 backbone/adapter 几何、Query/Prior 隔离、window/support、GRU order 与 padding、K=1…6 路径枚举、三类 head protocol、loss 等权语义、fresh-process import 和配置禁止项。2026-08-07 外部审核反馈修订后的本地全量回归为 `157 passed`、0 failure/error/skip；4 个 warning 来自 tsai 路径的 PyTorch JIT deprecation。

仍未闭合：新的 V1.2 M1 structural acceptance artifact、formal Backbone CPU/GPU smoke、含 adapter/shared GRU 的 label-free real-RF forward/support/autograd、真实下采样 Backbone 的几何 wrapper、Independent/Vanilla 的完整训练装配、M0-SUP 与所有真实监督训练/outer evaluation。上述未闭合项不因本地回归而自动通过。

## 12. 禁止性表述

当前不得声称：

- classifier-chain 分解由本研究原创；
- TCN 是正式训练 Backbone；
- K=4–6 已获得真实泛化证据；
- exact inference 可无界扩展到任意 K；
- Pre branch 是 raw-RF strict causal evidence；
- `ResponsibilityTokenProjector` 实现了 routing、mixture 或 expert assignment；
- propagation-conditioned 等同于物理前向传播建模或严格物理因果；
- V1.1 历史门禁等同于 V1.2 当前门禁；
- 任何真实 RF 性能、机制或论文级结论已经成立。
