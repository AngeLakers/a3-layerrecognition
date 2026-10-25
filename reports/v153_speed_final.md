# V1.5.3 最终 batch16 补测与冻结

按事前规则，**分桶与 cuDNN benchmark 均不采用**。正式默认固定为 **A（不含 fused）+ TF32 + batch16 + FP32**；保留 deterministic，BF16/fused/batch64 关闭。库级 `SpeedOptions()` 仍是全关闭兼容入口，发布模板 `v1_5_3_freeze.json` 是锁定默认。训练与 outer 许可仍关闭。

## 本次唯一判定表

| 条件 | ms/训练步 | 训练条/秒 | 相对基线 | 完整验证秒 | 决定/说明 |
|---|---:|---:|---:|---:|---|
| A（不含 fused）+ TF32，FP32/batch16 | 59.197 | 270.284 | 1.0000 | 29.726 | 固定基线 |
| 同基线 + 严格长度分桶 | 55.602 | 287.758 | 1.0646 | 30.085 | 不采用：小于 1.30 倍 |
| 同基线 + benchmark | 78.942 | 202.680 | 0.7499 | 29.984 | 不采用：小于 1.10 倍 |
| 固定 FP32-after200 权重，冷输入缓存 | — | — | — | 24.007 | 0 → 40,002 条 |
| 同权重，热输入缓存 | — | — | — | 23.248 | 复用 40,002 条，metrics 完全一致 |


F1/seed23/proposed_v15，三训练条件各 200 步、同 3,200 条 RF 身份、相同初始权重，各另做一次 40,002 条冻结验证面板。判据使用完整 200 步 wall time（含 worker 启动），没有剔除前 50 步或改用稳态速度。bucket/base=1.064649812（+6.465%）；benchmark/base=0.749876571（-25.012%）。这是一次共享 GPU 条件下的有限补测，不能推断独占 GPU 的稳态吞吐。

**当时 GPU 有其他负载。** RTX 3080 Ti 12 GB；各 arm 前占用约 4.2 GB、利用率 25–54%。冷/热校正测试前也有约 6.0 GB/42% 的共享占用。未停止其他进程，没有把显存占用全部归于本进程。

## 验证缓存实际收益

两次完整验证均固定原版 FP32-after200 权重（SHA256 `fd2985ffd750e5f779c0328efff388a1aa61e5b24deb173de67e8dd14155349e`），不更新参数。冷/热分别 24.007005/23.248238 秒，省 **0.758768 秒（3.161%）**，速度比 1.032638。OS 文件缓存与 GPU shape 已预热；“冷”特指 RF/meta/能量输入缓存为空。缓存不复用 encoder 特征或 logits。

预审发现初次 cache runner 误传初始化权重。原始结果保留为诊断，不参与这张表；只重做了指定 after200 权重的一冷一热，不重训参考权重。`CACHE_REFERENCE_CORRECTION.md` 和两组原始收据保留。此次用户所说“压缩缓存”明确指会话上下文，未压缩 RF 或验证张量。

## 分桶完整性与公平比较

生产 sampler 只重排冻结记录：按有效长度分桶，桶内用局部 seed/epoch RNG 随机排列，保留不足 16 的尾批，再随机打乱所有 batch。无 Head/标签/波形幅度输入，不消耗全局 Torch/NumPy RNG。尾批也参与批序打乱，不能按板连续拼接。

真实 40 份冻结 epoch 的 CPU 排批审计均保留原 100,732 条身份、无重复/遗漏、每批长度一致；四 Head 同 seed/epoch 顺序一致（真实 epoch1 同序检查及独立多 epoch 合成核验）。这项审计没有模型前向或参数更新。本批数据的每份 plan 有 6,296 批、尾批 12 条；一般数据的逐桶尾批可能使总批数增加，不能静默丢弃。

为使 200 步的三 arm 记录完全相同，从完整 epoch1 plan 的前 200 个满批定义共同 3,200 条集合，恢复其冻结索引顺序，再只给 bucket arm 重排。每种长度计数均为 16 的倍数，三 arm 恰好各 200 个满批；这不是完整 epoch 训练，也不是均匀随机记录抽样。原 epoch/validation 身份文件未改。共同 population SHA256：`71f9aa5b0ede6e0deba6639d8295bb0bac1ab6442ffeb87152981bf3bd86d889`。

## 数值与配置检查

所有新开关关闭，200 步 loss、最终全部权重与 v1.5.2 逐位一致；同权重 2,000 条验证 MAP 逐位一致，固定零前缀 logit 最大差 0。该兼容测试沿用原冻结顺序前 3,200 条，和速度比较的共同集合不同，不把两者吞吐直接比较。

TF32 是数值选项。当前最终条件固定同一原版 FP32-after200 权重的 2,000 条零前缀 logit 最大差 0.0018215179443359375，MAP 行不一致 0%；不据此宣称完整精度/泛化已验证。历史 R4 的全部有效前缀数值比较保留在旧报告，未替换本次 batch16 判据。

R6 源码本地与 SSH 全量各 439 pytest 通过。冻结增加全部 speed 字段、后端、batching 策略和 decision JSON 绑定检查；关闭训练许可时在数据/模型前拒绝。复核还修正最终 selected-checkpoint 验证的 speed 参数遗漏及执行收据硬编码 TF32 的问题；这些不改变已测 200 步 hot path。最终确切提交测试与发布/远端验证收据由 RELEASE.json 单列记录，不把 R6 dirty 源包冒称发布提交。

默认：workers=2、prefetch_factor=2，pin_memory/non_blocking、合并有限性检查、CPU 结构校验、验证输入缓存、CPU correction、跳过小批 summarize、nominal_fixed 快路开启；fused_adam=false、tf32=true、bf16_autocast=false、cudnn_benchmark=false、batch_size=16、length_buckets=false。后端 FP32，cuDNN deterministic 与 deterministic_algorithms 均 true。协议拒绝任意开关/后端/批序漂移。所有 Head 继续共享相同的数据工厂和冻结顺序，未执行四格训练。

校准表、能量标准化、RF/标签/空间裁剪、输入长度、抖动、冻结 cohort 全部沿用 v1.5.2，主干保留按有效长度分组的整批计算。原 RF、身份数组、权重与私有配置不上传 GitHub。独立审查原文和任何失败收据保留。

## 原始证据与运行边界

私有证据目录（不入库）：`final_batch16_20261025/result_corrected_remote.json`、`target_mirror/`（72 文件 ZIP 与逐文件 SHA 核验）、`source_r6_manifest.json`、`PROTOCOL.md`、`target_mirror/sampler_audit/`、`pytest_frozen.xml`、`reviews/`。旧 R2/R3/R4/R5 的失败/对照原样保留，本表只采用 R6 最终 batch16 数据。

冒烟包含限定的真实 RF 优化器更新，所有权重只用于测速，绝不用于正式 warm start。完整原 epoch 模型执行 0、四格训练 0、outer 模型评价 0。STOP 与训练 false 保留；发布和 SSH 源码同步不构成训练启动。
