# v1.6 R3 方案审查与模块实现

状态：开发代码，未冻结。依据 2026-11-06 R3 change-spec；较早 proposal 中的可学习相消、OCTS、改变选模等建议不在本轮范围。2026-11-07 用户要求仅审查、编写代码并提交，不再继续测试。

## 已落实

- Backbone 和批次支持 `[B,C,T]`，旧单通道 `[B,T]` 路径保留。共享有效长度、窗口及抖动；第二通道从同一前缀截断并除以 131。v1.6 仅 L/LP/V、关闭物理能量比。
- `data/m3_transform.py` 保留原 Gabor/OMP、CV 次序及归一化。新残差只减同一串归一化尺度的混响副本；保留一次回波。M1 FIR 残差不用于此处。任一拟合 δ 在 ±24 返回暂停状态，变换拒绝执行。
- 族级深层正常支持不足 500 时用 TRAIN K2 池；仍不足则 r=x、d=0。保留 pooled/identity 标记。
- 用户确认保留旧每上游状态组 200 条支持门。该门只控制分组诊断可用性，变换不按真实上游状态路由。
- TRAIN 抽样按折独立：catalog ordinal、source row 升序；每完整状态组最多 5000；`SeedSequence([20261105, fold_index])`，fold_index 为 1–5；返回全部所选身份。传入记录必须已符合完整可靠 eligible 规则。
- `evaluation/v16_d2.py` 固定 r 优先、d 回退；缺组、缺类 AUC 或支持不足返回 INSUFFICIENT。保留折×板事件，重复板不伪装独立试件；K2:0/K3:00、K3:01/K3:10分别守门。
- `evaluation/v16_metrics.py` 实现 IPW 联合准确率/FPR、异常三分、归属错误、MAP 逐界面召回、NLL/Brier、15 等 IPW 质量 ECE、风险覆盖/AUGRC、HCE 与 Coverage。MAP 使用保存的 log posterior，避免 float32 exp 近并列改变结果。ECE 对置信度并列组按比例分配权重；AUGRC 使用并列组线性随机接收积分。零分母返回空值。
- `evaluation/v16_temperature.py` 从完整 log posterior 恢复各前缀条件 logit 后缩放并重枚举，避免误用 joint softmax。混合 K2/K3 共享一个 T；重复试件同时留出。平坦目标返回 T=1 和 UNIDENTIFIABLE 标记。
- 温缩 CLI 检查单 head/seed、guarded 来源、完整五折验证板事件、重复记录与可靠 mask。输出全验证集温度候选，不授权 outer。
- `v16_protocol.py` 与 `v1_6_development.json` 固定开发配置和 65/95 格矩阵，禁止 L 双通道。正式训练和 outer 入口保持关闭。

## 审查裁决

独立 Codex 审查为同家族暂定意见；DeepSeek 为跨家族审查。原始意见在独立研究资料库保存，不用测试数量代替科学通过。

采纳：M3 副本单独重建、旧支持门、D2缺项不通关、C6校准阶段 OOF 限定、条件链温缩、log空间MAP和输入来源检查。

不采纳：放宽 D1 逐位一致性、修改既定500阈值、将零退化改为0.5pp、把边界暂停自动改成取消。这些会改变已确认协议。C6按板留出只隔离温度拟合记录；其他折的模型训练和checkpoint选择可能见过该板，因此不宣称独立盲测。

## 验证记录与未完成范围

在用户停止测试指令到达前，已有全套486项和随后局部61项通过。CPU真实F1/seed23前16条TRAIN记录，L/LP/V前向与一个优化步的232个张量与tag对应commit逐位一致。仅证明该CPU诊断范围，不代替GPU/全部样本验证。

新评价模块复算历史P的40,002行预测：异常联合准确率0.7740846259364778、FPR0.014682326202723696与原记录完全一致；NLL与原0.0974233724088195差3.75e-16。其他head未完成D3。

停止指令后的入口与文档修正只做静态审查，未重跑测试。

本轮交付模块代码；全量M3缓存CLI、数据集缓存身份/折绑定、训练runner完整接入、真实D2、分组报告图导出及最终协议冻结尚未完成。M3调用方仍须检查所有折×族的BOUNDARY_LIMITED并暂停整个M3阶段。未生成真实缓存、未训练、未运行outer、未打freeze标签。

已有未提交的数据准备产物、IDE配置、原始RF/标签、凭据和研究结果不包含在本轮代码提交中。

## 2026-11-06 补充接入（Claude 审查后修改）

- **训练入口接入 v1.6 输入。** 修改前 `speed.dataset_from_config` / `stream_training.train` 只在 `protocol == V1.5` 时加载 timebase，v1.6 配置因 `board_timebase` 窗口策略缺 timebase 会在建数据集时报错（不会静默运行）。现在 `is_v16(cfg)` 时：`load_v16_timebase` 校验冻结表（sha、version、13 板、中心递增、底面在最深中心之后）；`verify_m6_inactive` 确认继承 v1.5.3 的 M6 关闭决定；`runtime_record` 把 timebase、M6、speed decision（及双通道时的 M3 清单和 D2 决定）原字节复制进输出目录并写入 resolved_config。
- **M3 第二通道改为按折参数在线计算，不写 RF 大小的缓存。** 每折只保存拟合清单（各板族 f0、带宽、α、δ、状态，以及 13 板的板族归属）。`data/v16_channels.M3Channel` 在数据管线里对同一条原始记录做固定变换，截到与 x 相同的前缀、同样除以 131，得到 `[2, n]`；窗口抖动只改 interface_meta，因此两通道逐点对齐并共用 length mask。`m3_transform.Decomposer` 按（板, 记录长度）缓存字典和 Gram 矩阵，结果与 `transform` 逐位相同；单条记录约 0.5 ms（2 核 CPU）。
- **双通道守门。** `validate_config` 要求双通道格提供 `m3_fit_manifest`、`m3_d2_decision` 及其 sha256，单通道格必须为空；`load_m3_channel` 校验 D2 状态 DECIDED 且选中的通道与配置一致、D2 使用的正是该折清单、清单为该折 TRAIN 面板与冻结 timebase 拟合、无 BOUNDARY_LIMITED。训练时模型 `input_channels` 与数据集是否带 M3 必须一致。验证缓存和 `fast_eval_assembly` 支持 `[B,2,T]`。
- **脚本。** `scripts/a3_v16_m3_fit.py`（每折拟合，任一族 δ=±24 → 清单状态 PAUSED_BOUNDARY_LIMITED，退出码 3）；`scripts/a3_v16_d2.py`（五折验证板，Hilbert 包络峰值，Mann-Whitney AUC 与 AP，每板每 K:上游|深层 类最多 5000 条，`SeedSequence([20261106, fold_index, board_ordinal])`；INSUFFICIENT 退出码 4）。λ 网格沿用 RULES_T2_R3：1e-6、1e-4、1e-2、1、100。
- **D2 数据不足规则（Claude 提议，待 Brandon 确认）。** 一行有效 = 两类都有（AUC 有定义）且拟合时该上游状态的深层正常记录 ≥200。K2:0、K2:1 必须 5 个事件全有效（4/5 计数以 5 为分母）；K3:00/01/10 有效事件 ≥3 即用有效事件取中位数，剔除的事件写入决定文件；不满足 → INSUFFICIENT，停在 E4/E5 训练之前报告，不默认取消。
- **指标。** 新增 `attribution_overcall_of_abnormal` / `attribution_undercall_of_abnormal`（同一错层事件，以全部异常样本为分母）；`pooled_reliability` 汇总 K2+K3 的 top-label ECE、HCE0.9、Coverage、风险覆盖（不是各 K 的 ECE 平均）。
- 测试：`tests/test_v16_pipeline.py` 12 项 + 原 13 项通过；全套 476 通过，19 项失败与修改前相同（本环境未安装 tsai 等）。真实数据上未运行拟合、D2 或训练。
