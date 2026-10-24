# A3 Layer Recognition

当前工程版本：V1.3。利用raw RF与已知固定界面先验，输出逐界面状态、joint posterior、MAP和marginals。

2026-10-24代码快照包含四Head训练与精确联合推断、流式数据及加权验证、Labels V2修订工具、训练队列/检查点镜像/成功收据校验，以及OOF汇总入口。这是工程实现状态，不代表方法性能或科研验收通过。当前训练协议仍按最深界面F1、同分联合NLL选模；整体界面选模及训练预算的新方案尚未实施。

本仓库不包含原始RF、数据索引数组、训练权重或本机实验产物。真实数据配置和运维脚本保留登记环境的路径与SSH别名，跨机器使用须另配本地路径；克隆仓库不会自动启动训练。架构正文的链接指向独立资料库，GitHub不会直接展示该外部资料。

研究资料的统一入口：[A3实验资料](<../../实验资料/A3/README.md>)；架构与预处理正文不再维护仓库内镜像。

## 当前入口

- [完整网络与API](docs/ARCHITECTURE_V1_3.md)：四Head、输入/监督合同、训练评估与已实现消融。
- [代码审阅指南](docs/REVIEW_GUIDE.md)：源码、测试及待验证项。
- [V1.3验证记录](docs/VALIDATION_V1_3.md)：2026-10-01的181项单测、四Head短合成训练与12项CUDA合成smoke；本次文档清理未重跑。

Local Independent、Local＋Pre Independent、Vanilla Chain、Proposed通过同一前端、loss分派、optimizer、checkpoint和exact评估独立运行。主训练要求所有有效界面的可靠0/1完整；推理允许无标签/部分标签。未知前缀不能补0，当前未实现边际似然训练。

## 运行

使用已有`aislee-a3-layercongition` Conda解释器，在本项目根目录运行：

```powershell
python scripts/run_a3.py --config configs/experiment/v1_3_synthetic.yaml --head proposed --output local/new_v13_proposed
python -m pytest -q
```

`--head`支持`local_independent`、`local_pre_independent`、`vanilla_chain`、`proposed`；输出目录须为新的目录。配置、checkpoint重载及NPZ manifest字段见完整规范第11节。

PyCharm Run/Debug使用相同脚本、参数、解释器和项目根工作目录；入口自行解析`src/`，不依赖聊天状态或手动PYTHONPATH。依赖清单见[requirements.txt](requirements.txt)、[environment.yml](environment.yml)和[pyproject.toml](pyproject.toml)；本次未安装或升级环境。

## 输入和模型范围

- 保持硬件200 MHz与RF相位；不做四倍抽取。已截取记录允许起点浮动，粗参考与共享Local窗口容差须覆盖目标回波，具体范围尚待实测。
- 正式Backbone候选为ResNet1D、InceptionTime、FCN-1D；TCN仅供smoke/debug。Adapter统一为`D_h=64`并保留时间坐标。
- 链仅按浅→深；不安排反序链或传统闸门特征分类基线。`h=0`仍是尚未接入的后续候选。
- 实际试样K=2/3，`K_cap=6`仅为工程支持范围。真实RF训练须先满足已有标签、实体split和M0-SUP要求，合成配置不会放行真实训练。

## 源码布局

| 路径 | 职责 |
|---|---|
| `src/a3_ultrasound/data/`、`schema.py` | metadata、RF加载、collate、有效性与mask |
| `src/a3_ultrasound/models/` | Backbone/Adapter、Query、Local/Pre、职责表征、四Head与loss |
| `src/a3_ultrasound/inference/` | exact枚举、joint/MAP/marginals |
| `src/a3_ultrasound/training.py`、`scripts/run_a3.py` | 公共训练、split审计、checkpoint与exact评估 |
| `configs/experiment/`、`tests/` | 可执行配置及结构/数值/训练检查 |

原采集collate仍接收单manifest；公共tensor runner支持mixed-K，并提供流式catalog、实体划分检查、独立outer入口与OOF汇总。实现入口不等于已完成全部跨折/跨seed实验，实际科研结果以独立资料库中的版本化证据为准。

## 历史查证

[V1.2规范](docs/ARCHITECTURE_V1_2.md)与[V1.2验证记录](docs/VALIDATION.md)原样保留。旧M1/M2脚本、matched-joint容量配置和历史PASS只适用于对应源码/输入，不能证明V1.3验收完成。Time-Series-Library是保留的可选源码依赖，核心单元测试不要求初始化该子模块，TimesNet尚未接入。

真实数据、身份映射数组、凭据、私有配置和权重不纳入代码推送范围；历史提交及外部资料库中的原始实验记录保留。
