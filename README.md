# A3 Layer Recognition

A3 从单条超声 RF 波形及已知的界面位置推断各界面状态，输出联合状态分布、MAP 组合和逐界面边缘概率。当前工程版本为 **v1.8**，开发以 `master` 为主线，可用 `v1.8` 标签检出本次整合快照。

- 查阅 v1.8 的输入、读头、配置和源码入口。[v1.8 实现说明](docs/V18_IMPLEMENTATION.md)
- 按版本查看当时的设计动机、实现变化和原始记录。[版本历史](HISTORY.md)

## 本地验证

在仓库根目录使用 Python 3.11 或更高版本及项目依赖。下面运行 v1.8 的合成用例，输出目录应为新目录。[依赖清单](pyproject.toml)

```powershell
python -m pip install -e .
python scripts/run_a3.py --config configs/experiment/v1_8_gru_synthetic.yaml --output local/v18_gru_smoke
python -m pytest -q
```

合成检查覆盖模型装配、训练、精确枚举和检查点重载。真实实验的 RF、试件划分、运行配置、权重及研究报告由独立的 A3 研究资料库维护。
