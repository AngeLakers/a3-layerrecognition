# A3 v1.7.1 执行计划

对应研究计划 `a3-v1.7.1-20261118-plan-b4-32.md`、逐格 Tracker 和论文框架 `FRAMEWORK_V171_B4_32.md`。当前为本地验收和提交阶段。推送、云端训练和新增 outer 评价各自沿用项目授权。

## 实际矩阵

| 部分 | 实验 | 新训练 |
| --- | --- | ---: |
| B1 | TCN、InceptionTime 各比较 LP 和 GLOBAL，F1–F5 × 23/41/59 | 60 |
| B2 | ResNet1D 的 L-A 全有效时域 Local，F1–F5 × 23/41/59 | 15 |
| B3 | 基于完整组合 MAP 校准置信度，对 ResNet1D LP 和 G1D 的保存预测做整条复检评价 | 0 |
| B4 | 目标 101，F1–F4 × 23/41 × LP/G1D × B/C | 32 |
| 合计 | 九个首格计入总数 | **107** |

ResNet1D 的原 LP、G1D、L-C 直接复用。B2 主比较为 L-C 与 L-A，去掉 L-B 和 90 格扩展。B3 去掉 R1，保留零置信度复检、等复检量随机参照、分组合结果和深层漏检。B4 固定 101，001 为审计阶段备用。A 仅在归档权重可按同一去目标验证协议合法重选时作为辅助。先验扰动一致性及 TRAIN 加权留待后续独立验证。

B4 的八个折×种子配对均满足当前标签、抽样和验证支持。B/C 每轮样本数按 F1–F4 依次为 98,020、92,944、92,356、104,672。C 在同一试件上等量删去其他异常，实测替代样本全部来自目标所在扫描块以外的区域。B/C 的对比可控制试件和删样规模，空间覆盖和界面标签边际差异须随结果报告。整个折×种子配对若支持不足，四格共同记录原因。基础配置、索引或载入错误按错误处理。

## 冻结输入与选择

复用 v1.7 fixed40 训练入口、Labels V2、13 试件划分、原 timebase、单通道 RF 和共享 ±24 点 jitter。模型学习率从 0.001 起按归档公式执行两轮 warmup 和逐步 cosine，最低 0.00001。40 轮、训练 batch 16、AdamW、权重衰减 0，归档 FP32/TF32 和 deterministic 设置。LP 的有效 Local 辅助权重为 0.3，GLOBAL 与 L 的有效权重为 0。

B1/B2 使用原始每轮抽样和 IPW 验证。B4 对两套 TRAIN 角色冻结独立的 40 轮清单，LP/G1D 在同一臂共用清单。B/C 共用去目标后重建的 IPW 验证。按原 guarded_joint 选择 checkpoint，完整正常记录的联合 FPR ≤2%。仅有 fallback 的格子登记为正式比较缺格。正式 outer 按原试件队列、零 jitter 评价。

归档 `g1d_code.tar` SHA-256 为 `eba8933f5ae0a7113ebee8e4a2a701feb6471f65351f15d43e558dd03856bb7a`。Labels V2 目录签名为 `7fac189e16773f3d638d664ee1c27f68dc6e7f425b8b03c88f24dcefef8497c7`。原资料位置依据 A3 项目索引解析。

## 运行顺序

1. 在目标主机上解析 RF 目录和固定工件，确认实际运行代码提交、CUDA、CPU、磁盘与分阶段授权。使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
2. 生成矩阵。reference 使用归档 `a5_configs/F1_s23_full.json`。fixed-artifacts 使用归档一格的 `timebase_calibration.json`、`m6_decision.json`、`speed_decision.json` 所在目录。saved-panels 使用 `metadata_65`，包含原 `epochs` 和 `validation_selection`。输出目录需为新目录。配置内路径可随重建目录适配主机，矩阵的 SHA-256 用于绑定实际配置。
3. 先跑 F1/s23 的 B1 四格、L-A 一格及 B4 四格。检查 train/val 总量、更新数、有效辅助损失、吞吐、显存、保存和守卫，再扩到其余格。配对优先按完整折×种子推进。失败格保存现场与原配置，按同配置和 seed 在新输出目录重跑。
4. 在取得对应授权后做 outer。保存各格配置、选择面板、checkpoint 收据、原始预测与摘要。按共同有效种子后试件等权统计，保留逐板方向、真实支持、失败及缺格。
5. B3 使用全量可靠验证预测逐格确定温度和 50% 异常复检目标的可达阈值，随后固定应用于匹配的已保存 outer。LP/G1D 实际异常复检率差超 0.5 个百分点时同预算主对比记 NA。B4 在每项对比所需的模型×臂条件上取共同种子，主要终点为 101 的 raw MAP 判对率，另有目标 FPR、精确率、已见异常、正常误报、深层和目标错判去向。B3 与 B4 的评价不增加训练格。

```sh
python -m a3_ultrasound.v171_run prepare --reference REFERENCE.json --catalog CATALOG.json --fixed-artifacts FIXED --saved-panels PANELS --output PREPARED --target 101
python -m a3_ultrasound.v171_run train --config PREPARED/configs/CELL.json --output RUNS/CELL --authorization TRAIN.json
python -m a3_ultrasound.v171_run calibration-validation --config PREPARED/configs/CELL.json --run RUNS/CELL --output VALIDATION/CELL --authorization TRAIN.json
python -m a3_ultrasound.v171_run outer --config PREPARED/configs/CELL.json --run RUNS/CELL --output OUTER/CELL --authorization OUTER.json
python -m a3_ultrasound.v171_b4 --matrix PREPARED/matrix.json --outer-root OUTER --authorization OUTER.json --output B4.json
```

训练和 outer 收据的 `protocol` 为 `A3-v1.7.1-fixed40`，`phase` 分别为 `train` 和 `outer`，`authorized` 为 `true`，`cells` 是获准 run ID 到配置 SHA-256 的映射。B3 读取保存的 outer 预测时使用独立 `b3-outer` 收据，其中 `cells` 绑定预测文件 SHA-256。

```sh
python -m a3_ultrasound.v171_b3 --reference REFERENCE.json --catalog CATALOG.json --validation VALIDATION_RAW --fold F1 --head local_pre_independent --seed 23 --run-id F1_s23_local_pre_independent_none --output B3_F1_LP_s23.json
python -m a3_ultrasound.v171_b3 compare --output B3_COMPARISON.json REPORTS/*.json
```

B3 的工作点与逐板/分层统计保存为 JSON，同一格全体试件的可达阈值、对应计数和端点保存为 `.curves.npz`。正式读取保存的 outer 时加 `--outer-predictions` 和 `--outer-authorization`。已归档的 ResNet1D 用原生 `raw.npz` 与配套收据，新增骨干可用新的 `predictions.jsonl`。

## 算力和本地验收

建议 **2 张 RTX 4090**。单卡规格为 24 GB，本地 CUDA 检查用的是 RTX 4060 Laptop，真实 RF 批的检查停在装载和抽样。云端先开每卡 2 格，首批 train/val 都跑通后再按显存、CPU、I/O 和吞吐升至每卡 3 格，即总计 5–6 格并发。首格耗时和验证开销用于更新总 GPU 小时。[NVIDIA RTX 4090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)

本地核过 107 格身份与正式 Tracker、重算了 640 份 B4 每轮索引文件的 SHA-256，验证旧采样与新 B/C 入口可以生成实际 RF 批。F1/s23 的历史 LP、G1D 验证原始后验跑通 B3。TCN、InceptionTime、ResNet1D 的七条实际模型路径以合成输入跑通 CUDA 前向、反向和 AdamW。训练与新增 outer 的实机端到端结果待云端授权执行。
