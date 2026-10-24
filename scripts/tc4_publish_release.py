"""Publish the verified frozen TC4 release and update current A3 records."""
from pathlib import Path
import json
import hashlib
import re

DATA = Path(r"E:\新建文件夹\扩散焊数据集")
REL = DATA / "数据版本" / "TC4四层_冻结版_R1_20261020"
A3 = Path(r"I:\programming\实验资料\A3")


def read(path):
    return json.loads(path.read_text("utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = read(REL / "release.json")
    clean = read(REL / "cleanup_result.json")
    flow = read(REL / "flow_rerun" / "summary.json")
    verified = read(REL / "release_validation.json")
    assert clean["status"] == "completed" and len(clean["files"]) == 28
    assert flow["status"] == verified["status"] == "passed"
    assert all(not Path(r["path"]).exists() for r in clean["files"])
    assert sum(r["bytes"] for r in clean["files"]) == clean["deleted_bytes"]
    old_flow = read(Path(manifest["label_root"]) / "engineering_smoke" / "summary.json")
    assert old_flow["input_sha256"] == flow["input_sha256"]
    binding = dict(release_id=manifest["release_id"], release_sha256=verified["release_sha256"],
                   flow_summary_sha256=sha((REL / "flow_rerun" / "summary.json").read_bytes()),
                   input_identical_to_previous_run=True, all_four_heads_passed=True,
                   formal_split_created=False)
    (REL / "flow_binding.json").write_text(json.dumps(binding, indent=2), "utf-8")
    gb = clean["deleted_bytes"] / 1e9
    gib = clean["deleted_bytes"] / 2**30
    report = f"""# TC4四层冻结版 R1

版本：`{manifest['release_id']}`。状态：数据固定、标签固定、正式实体划分暂缓。

## 唯一使用入口

[release.json](release.json)记录当前波形、索引、标签和来源文件的路径、字节数、SHA-256。清单自身校验见[release.sha256](release.sha256)，完整检查见[release_validation.json](release_validation.json)。本版固定既有V3波形＋标签v1，直接引用它们，不复制第二份大波形。后续修改数据或标签规则必须另增版本，不能覆盖本版清单绑定的文件。

- 信号共998255条，959881条具有三个界面完整标签；000正常背景955371条、含设计缺陷4510条，38374条保留mask。
- 标签文件在[标签目录](../../处理结果/TC4四层_图纸标签与小流程_v1_20261020/)下的`batch1`至`batch4/labels.npz`。`y`顺序为I1/I2/I3，未知为-1；须结合`label_mask`。主训练候选采用`complete_label_mask`，不要直接把`nominal_y`当可靠真值。
- 波形在[V3目录](../../处理结果/TC4四层_底波后裁剪_v3_20261020/)；用同批`index.npz`的offsets/lengths读取。原始int16与已低通float32是不同用途，均保留。完整波形仍未归一化；工程tensor固定除131，不再次滤波。
- 原始TXT/NDT保留；四块独立实体及每层3/3/4/4 mm、增益13/11/14/14 dB等事实保持。
- 正式train/validation/test等待后续批次，旧4折建议未启用。全量正常池保留，训练时在训练实体内动态抽样；9020条平衡采样示例不等于正式训练集。

## 实际标签在哪里

图按设计正面坐标显示，可能与原扫查屏幕左右方向不同。下图每行一块、三列分别为I1/I2/I3。红色是缺陷内部标签1，浅蓝是无设计缺陷参考区标签0，灰色为边缘/板边未知，黑色为1号已整体排除的两处。黑线是设计轮廓。0不代表焊后机械强度认证。

总览的正常点仅为绘图每5点显示1点，因此可见浅色条纹，**不表示只给条纹位置打标签**；保存的标签覆盖完整正常参考域。缺陷及未知点未作该显示抽稀，局部放大图显示全部点。

![四块三界面](figures/four_specimens_three_interfaces.png)

![联合状态](figures/four_specimens_joint_labels.png)

联合图只有三个界面都可靠才显示000–111，否则为灰色。统计数量和边缘余量沿用标签v1：1–4号0.55/0.80/0.75/0.60 mm，基于统一配准残差规则，不是实测制造误差界限。2 mm小槽保留的中心点少，放大图可直接查看。

局部放大：[1号](figures/batch1_label_details.png) · [2号](figures/batch2_label_details.png) · [3号](figures/batch3_label_details.png) · [4号](figures/batch4_label_details.png)。

## 本轮复跑与核验

1. 四批共1386600个原始TXT文件名/大小与来源清单一致；另随机回读128个原始TXT，哈希及原始缓存值一致。
2. 当前998255条的整数和低通保留波形均再次与V2.1父版逐值比较通过，然后才清理父版大数组。
3. 冻结清单58个文件完成全文件SHA-256验证，清理后再次核对当前数据可读性和哈希。
4. 124条真实RF、372个界面标签，FCN-1D四Head各16次参数更新；有限损失/梯度、Backbone与Head实际更新、exact概率归一化、无标签推理一致性及保存/复载均通过。输入tensor哈希与上次完全一致。详见[本轮流程](flow_rerun/summary.json)及[绑定记录](flow_binding.json)。

没有正式划分、测试准确率、跨试块性能或模型优劣结论。检查点仅工程用途，不作为未来预训练权重；本次样本使用记录保留。先前完整M1/M2LF、最终时间先验和正式研究实验要求没有被此小流程替代。

初次并行核验受Windows虚拟内存不足影响，未删除任何文件；改为串行及小缓冲后完成，未改系统设置、未安装依赖。[尝试记录](execution_attempts.json)保留。

## 已清理的内容

按用户本次明确授权，删除28个旧数组，合计**{gb:.3f} GB（{gib:.3f} GiB）**：

- 四批原始TXT转换缓存4个；
- V1早期/复核后裁剪数组8个；
- V2及V2.1整数/低通波形数组16个。

这些是可由原始TXT、保留索引和规则重建的缓存或已被V3替代的处理阶段，不声称全部逐字相同。源数据、当前V3两类波形、标签、设计、索引、哈希、规则、脚本、质量图和真实运行报告未删除。历史目录保留小型追溯材料并新增清理说明；旧脚本若依赖已清缓存，须先在新恢复目录重建，不能当作当前直接运行入口。

[清理前清单](cleanup_plan.json)记录每个目标哈希，[执行日志](cleanup_result.json)记录实际删除文件。删除前验证绝对路径范围、文件类型、大小及完整哈希；只按LiteralPath删除具体文件，没有递归删除目录。

## 复跑方法

实现脚本在`I:/programming/projects/a3-layerrecognition/scripts/`，本次版本快照在[provenance/code](provenance/code/)；模型源码未改。源码快照用于追溯，模型依赖仍由实现仓库提供。

```powershell
& 'I:\\programming\\anaconda3\\python.exe' 'I:\\programming\\projects\\a3-layerrecognition\\scripts\\tc4_freeze_release.py' verify --release '{REL}'
& 'I:\\programming\\anaconda3\\envs\\aislee-a3-layercongition\\python.exe' 'I:\\programming\\projects\\a3-layerrecognition\\scripts\\tc4_unsplit_smoke.py' --labels '{manifest["label_root"]}' --device cuda --output '<新的运行目录>'
```

已有运行目录拒绝覆盖。正式分组仍需等待后续批次，不运行旧split建议。
"""
    (REL / "README.md").write_text(report, "utf-8")
    backup = REL / "documents_before_update"
    backup.mkdir(exist_ok=False)
    note = f"""## 2026-10-20 当前：本批冻结 R1、清理完成、流程复跑通过

用户要求固定这批数据、清理重复中间产物、画实际标注位置并复跑更新状态。当前唯一入口为[TC4四层冻结版R1](<E:/新建文件夹/扩散焊数据集/数据版本/TC4四层_冻结版_R1_20261020/README.md>)，清单绑定V3波形＋标签v1，不复制大波形。998255条信号、959881条完整标签（955371正常、4510含设计缺陷）、38374条mask保持。

58个数据/来源文件已登记完整哈希；四批1386600个原TXT文件名/大小复核、128个源文件哈希抽查、998255条两类保留波形与父版全量比对通过。已按用户明确授权清理28个可重建旧数组，共{gb:.3f} GB；原始TXT/NDT、当前V3、标签和追溯资料保留。历史“旧版大数组全保留”由本次清理指令覆盖。

四块×三界面总览、联合状态及每块15槽放大图已输出。124条真实RF四Head各16次更新的CUDA流程复跑通过，exact推理/复载一致，输入与上次同哈希；没有准确率或泛化结论。正式划分仍按用户要求等待后续批次，旧4折不启用，正常池不永久删减。完整研究验收/最终时间先验与正式模型实验仍待后续，本次只完成数据冻结及小规模工程核验。

"""
    changes = []

    def edit(path, transform):
        raw = path.read_bytes()
        saved = backup / f"{len(changes)+1}_{path.name}"
        saved.write_bytes(raw)
        text = raw.decode("utf-8")
        new = transform(text)
        assert path.read_bytes() == raw, "Concurrent edit"
        path.write_bytes(new.encode("utf-8"))
        changes.append(dict(path=str(path), backup=str(saved),
                            before_sha256=sha(raw), after_sha256=sha(path.read_bytes())))

    def prepend(text):
        nl = "\r\n" if "\r\n" in text else "\n"
        at = text.find("\n") + 1
        return text[:at] + nl + note.replace("\n", nl) + text[at:]

    status = f"""# A3 当前状态

更新：2026-10-20。本页给出当前结论；历史依据在合同、实验日志和版本化报告。

|项目|当前状态|
|---|---|
|研究主线|固定界面先验＋signed RF，逐界面0/1及联合状态推断；V1.3四Head|
|本批实体|四块独立TC4四层试块；1/2号每层3 mm，3/4号每层4 mm，正面入射|
|数据固定|`TC4-4L-R1-20261020`：V3波形＋图纸标签v1，58文件完整哈希绑定|
|信号与标签|998255条信号；959881条三界面完整标签，其中000=955371、含设计缺陷=4510；38374条带mask|
|争议位置|1号I1两处及周边按此前授权从样本整体排除，不待补充确认|
|信号处理|前端裁剪、20 MHz低通、底波后截尾已冻结；保留DC与signed RF；完整波形未归一化，工程输入固定除131、不重复滤波|
|实体划分|用户明确暂缓，等后续批次统一安排；旧4折建议不启用|
|类别数量|全量保留；正常/缺陷1:1仅为训练采样起始示例，后续在确定的训练实体内轮换正常点|
|本次验证|全量父版切片一致、58文件哈希、原TXT清单及128哈希抽查；124条真实RF四Head各16次更新、exact与复载通过|
|空间标注图|四块×三界面、联合状态、每块15槽放大图已输出|
|存储清理|用户授权删除28个可重建旧数组，{gb:.3f} GB；原始及当前有效数据、追溯材料保留|
|未完成|后续批次接入、最终实体split/采样与时间先验配置、完整M1/M2LF及正式跨实体实验|
|性能结论|未报告测试准确率、泛化或模型优劣；工程检查点不作预训练模型复用|

[冻结版与图件](<E:/新建文件夹/扩散焊数据集/数据版本/TC4四层_冻结版_R1_20261020/README.md>) · [研究合同](idea-stage/docs/research_contract.md) · [Tracker](refine-logs/EXPERIMENT_TRACKER.md) · [实扫登记](a3/docs/specimen_design/TC4_FOUR_LAYER_INTAKE_20261020.md)。

其他K2/LU及图纸历史不由本批四块数量替代，见[试块资料](a3/docs/specimen_design/README.md)。研究准入与主张要求继续按[RESEARCH_GOVERNANCE](RESEARCH_GOVERNANCE.md)和合同执行，不以小流程代替正式研究验收。

更新前状态页完整备份在[本次恢复记录](<E:/新建文件夹/扩散焊数据集/数据版本/TC4四层_冻结版_R1_20261020/documents_before_update/1_PROJECT_STATUS.md>)。
"""
    edit(A3 / "PROJECT_STATUS.md", lambda _: status)
    for path in [A3 / "idea-stage/docs/research_contract.md",
                 A3 / "a3/docs/specimen_design/TC4_FOUR_LAYER_INTAKE_20261020.md",
                 A3 / "refine-logs/EXPERIMENT_TRACKER.md"]:
        edit(path, prepend)
    for path in [A3 / "findings.md", A3 / "refine-logs/EXPERIMENT_LOG.md"]:
        edit(path, lambda t: t + "\n\n" + note)
    index_note = """## 当前固定数据入口

[TC4四层冻结版R1](数据版本/TC4四层_冻结版_R1_20261020/README.md)为后续使用入口，清单绑定当前V3波形及标签v1。正式划分仍暂缓。旧缓存及被替代数组已按授权清理，旧目录保留索引、图和报告；历史“原样保留旧波形”的说明已由本次清理覆盖。详见冻结版删除清单与日志。

"""
    edit(DATA / "信号数据索引.md", lambda t: t[:t.find("\n")+1] + "\n" + index_note + t[t.find("\n")+1:])
    edit(DATA / "处理规则/统一信号处理规则.md", lambda t: t + "\n\n" + note)
    edit(DATA / "AGENTS.md", lambda t: t + "\n- 本批已固定为[TC4四层冻结版R1](数据版本/TC4四层_冻结版_R1_20261020/README.md)。按清单校验当前波形/标签，变更另增版本。旧中间大数组已按用户授权清理，原始TXT/NDT与当前数据保留；不要把历史目录当作仍包含波形的运行入口。\n")
    (REL / "document_updates.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), "utf-8")
    for dirname in ("TC4四层_四批质量与裁剪_20261020", "TC4四层_时间裁剪与LP20_v2_20261020",
                    "TC4四层_时间裁剪与LP20_v2_1_20261020"):
        p = DATA / "处理结果" / dirname / "CLEANUP_NOTICE.md"
        p.write_text(f"# 本目录旧波形已清理\n\n用户2026-10-20授权固定最终数据并清理中间产物。本目录中列入[执行日志](../../数据版本/TC4四层_冻结版_R1_20261020/cleanup_result.json)的旧波形数组已删除，索引、规则、源清单、脚本、图及报告保留。历史报告中“保留旧波形”描述的是当时状态。\n\n当前请从[冻结版R1](../../数据版本/TC4四层_冻结版_R1_20261020/README.md)进入。原始TXT/NDT未删除；若要复现历史波形，须按保留索引/规则在新恢复目录重建，不直接覆盖当前数据。\n", "utf-8")
    # Validate all local README references, including the directory entrypoints.
    links = re.findall(r"\]\(([^)]+)\)", report)
    for target in links:
        assert (REL / target).exists(), target
    print(f"Published release and updated {len(changes)} documents; {len(links)} report links checked.")


if __name__ == "__main__":
    main()
