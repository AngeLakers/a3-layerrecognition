# A3 V1.3 验证记录

日期：2026-10-01。使用现有登记Python环境，源码未commit/push。

## 结果

- `python -m pytest -q --junitxml=local/v1_3_validation_20261001/pytest_final.xml`：**181 passed，4 warnings，55.94秒**，0 failure/error/skip。warning来自既有tsai/PyTorch JIT弃用提示。
- 公共`run_a3.py`入口：四Head各2 epochs合成训练完成；配置、history、best checkpoint和逐样本exact预测已保存。
- 通过`--checkpoint`独立重载Proposed评估，逐样本test JSONL与原结果逐字一致。
- `python scripts/v1_3_smoke.py --device cuda --output local/v1_3_validation_20261001/formal_cuda_smoke.json`：三正式Backbone × 四Head，**12/12通过**。
- 独立只读代码审查使用`gpt-6-astra/xhigh`。发现测试集被错误要求完整标签，修复为train完整、validation具备模型选择标签、test可无标签/部分标签；复核后blocking=0。

GPU为RTX 4060 Laptop GPU；smoke使用batch=2、长度2000/2300、K=2/3、float32、AMP关闭、seed=43。12例最大PyTorch allocated峰值102.954 MiB，最大`|raw logZ|=1.1921e-7`。此数值不代表设备总占用或最大可用batch。

## 新增测试重点

四Head前向/反向/optimizer覆盖；Local＋Pre不读状态前缀；非法前缀拒绝；非零条件依赖下逐路径概率、raw logZ、MAP/marginal和teacher主损失一致；mixed-K与标签泄漏；空Pre/padding有限梯度；三项显式消融；完整训练标签、部分validation/无标签test；checkpoint往返；specimen split泄漏和真实训练gate拒绝。

## 复跑与产物

```powershell
python scripts/run_a3.py --config configs/experiment/v1_3_synthetic.yaml --head proposed --output local/new_v13_proposed
python scripts/run_a3.py --config configs/experiment/v1_3_synthetic.yaml --checkpoint local/new_v13_proposed/best.pt --split test --output local/new_v13_reload
```

Head可替换为`local_independent/local_pre_independent/vanilla_chain/proposed`。每次使用新目录，runner拒绝覆盖已有run。

本次本机证据：`local/v1_3_cli_20261001/`和`local/v1_3_validation_20261001/`，Git忽略，不随仓库分发。四Head CLI之后补充了run内Git版本/dirty和wall-time记录；最终完整回归已覆盖当前代码的训练/checkpoint往返，早期CLI产物保留原状。

## 边界

没有真实RF训练或消融性能比较，没有更新M0-SUP、真实split、完整M1/M2-LF或科研晋升裁决。没有新的外部API双审，不把本地独立代码审查称作科研双通道通过。未执行PyCharm GUI、远端CI或新环境安装。V1.2原架构和验证记录保留；当前规范见[ARCHITECTURE_V1_3](ARCHITECTURE_V1_3.md)。
