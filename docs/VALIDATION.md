# V1.2 仓库审阅验证记录

日期：2026-09-17。适用于引入本文件的发布整理提交；提交标识以 Git 历史为准。

## 实际运行

工作目录为实现仓库根目录，使用现有登记环境：Python 3.11.15、PyTorch 2.12.1+cu126、pytest 9.1.1。

```powershell
python -m pytest -q
```

最终结果：**157 passed，4 warnings，57.97 秒**。4 个 warning 来自 tsai 路径调用 PyTorch 已弃用的 `torch.jit.script`，无 failure/error/skip。

首次运行结果为 156 passed、1 failed：`test_fresh_process_can_import_enumerator_without_models_import_order` 启动的新 Python 进程没有继承 pytest 的 `pythonpath=["src"]`，报 `ModuleNotFoundError: a3_ultrasound`。将该测试的子进程工作目录设为仓库 `src/` 后，保留其 fresh-import/circular-import 检查目的，重跑全量得到上述结果。未通过设置会话专属 `PYTHONPATH` 绕过，也未改变网络实现。

## 文档和发布内容核对

- `docs/ARCHITECTURE_V1_2.md` 的第 1–12 节与研究工作区原规范逐字一致；只调整顶部来源和导航以适应独立仓库。
- 原架构文档 SHA-256：`33eed0290eccca4f7637b4e108d43f96646b1b5096afc543ece3cb3983ce709d`。
- 已对两个既有提交的版本化文件及拟发布工作树文件做凭据、私人路径、邮箱模式检查；明确的假凭据测试 fixture 已人工区分。模式扫描不能保证识别所有敏感信息。
- 上传范围为实现仓库的源码、脚本、测试、配置和文档；不纳入原始实验数据、真实 manifest、内部研究记录及本机运行产物。
- 已存在的共享 PyCharm 项目配置保留；被忽略的个人 workspace 配置不纳入。此次未提交现有 `.idea/vcs.xml` 的本地改动。
- 第三方 Time-Series-Library 保留既有 gitlink `4e938a1767106324dd753b2a44832bf870a0252e`，未修改其内容或验证 TimesNet 集成。

## 证据范围

本次验证证明当前 checkout 的单元测试通过。未执行真实 RF 实验、正式监督训练、V1.2 M1 acceptance 重建、完整 formal-Backbone GPU 门禁、远端 CI、另一台机器全新依赖安装或 PyCharm GUI 运行。这些项目不因本次上传自动通过。

完整架构的研究状态与待完成项保持原有边界，参见 [审阅指南](REVIEW_GUIDE.md) 和 [架构规范](ARCHITECTURE_V1_2.md)。
