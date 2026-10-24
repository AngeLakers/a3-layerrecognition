"""Bind the accepted label revision and diagnostic evidence; update A3 records."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tc4_label_pool import write_json

DATA = Path("E:/新建文件夹/扩散焊数据集")
A3 = Path("I:/programming/实验资料/A3")
LABELS = DATA / "处理结果/TC4四层_小缺陷边界标签_v2_20261020"
RUN = DATA / "处理结果/TC4四层_学习诊断_v1_20261020"
R1 = DATA / "数据版本/TC4四层_冻结版_R1_20261020"
R2 = DATA / "数据版本/TC4四层_冻结版_R2_20261020"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    summary = json.loads((LABELS / "summary.json").read_text("utf-8"))
    result = json.loads((RUN / "summary.json").read_text("utf-8"))
    assert summary["status"] == result["status"] == "passed"
    assert result["formal_split_created"] is False
    console = RUN.with_name(RUN.name + ".console.log")
    if console.is_file():
        shutil.copy2(console, RUN / "console.log")
    inputs = dict(np.load(RUN / "diagnostic_input.npz"))
    target = inputs["y"]
    assert len(target) == 512 and target.shape[1] == 3
    profile = json.loads((LABELS / "profile.json").read_text("utf-8"))
    for b in range(1, 5):
        source = Path(profile["signal_root"]) / f"batch{b}"
        index = dict(np.load(source / "index.npz"))
        labels = dict(np.load(LABELS / f"batch{b}" / "labels.npz"))
        rf = np.load(source / "rf_lp20_float32.npy", mmap_mode="r")
        for j in np.flatnonzero(inputs["batch_id"] == b):
            i = int(inputs["sample_id"][j])
            n, offset = int(index["lengths"][i]), int(index["offsets"][i])
            np.testing.assert_array_equal(inputs["x_raw"][j, :n],
                                          rf[offset:offset + n] / np.float32(131))
            np.testing.assert_array_equal(target[j], labels["y"][i])
            assert inputs["source_row"][j] == index["source_row"][i]
            assert inputs["length_mask"][j].sum() == n
        del rf, index, labels
    group_rows = []
    for row in result["runs"]:
        predictions = dict(np.load(RUN / row["head"] / "predictions_final.npz"))
        pred = predictions["marginals"] >= .5
        map_states = predictions["map_states"]
        assert np.isclose((map_states == target).all(1).mean(), row["final"]["exact_match"])
        for k, recorded in enumerate(row["final"]["per_interface"]):
            direct = np.zeros((2, 2), dtype=int)
            for true, estimated in zip(target[:, k], pred[:, k]):
                direct[int(true), int(estimated)] += 1
            np.testing.assert_array_equal(direct, recorded["confusion"])
        groups = {}
        for name, mask in (
            ("all_2mm", inputs["small_defect_mask"]),
            ("new_2mm_boundary", inputs["boundary_admitted_mask"]),
            ("old_2mm_core", inputs["small_defect_mask"] & ~inputs["boundary_admitted_mask"])):
            assert np.all(target[mask] == 1)
            groups[name] = dict(positive_interfaces=int(mask.sum()),
                               recalled=int(pred[mask].sum()), recall=float(pred[mask].mean()))
        group_rows.append(dict(head=row["head"], **groups))
    baseline = dict(always_000_exact_match=float((target == 0).all(1).mean()),
                    always_0_interface_accuracy=float((target == 0).mean()),
                    always_0_positive_recall=0.0)
    write_json(RUN / "diagnostic_audit.json", dict(
        status="passed", saved_rf_rows_rechecked=512,
        label_source_and_order_exact=True, independent_confusion_recount=True,
        always_normal_baseline=baseline, positive_interface_groups=group_rows,
        generalization_claim=False))
    old = json.loads((R1 / "release.json").read_text("utf-8"))
    R2.mkdir(exist_ok=False)
    files = []
    # Full parent hash recheck also verifies that the new-label operation did
    # not alter old labels, source indexes, exclusions or current RF payloads.
    for i, entry in enumerate(old["files"], 1):
        path = DATA / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert digest(path) == entry["sha256"], str(path)
        files.append(dict(entry, role="parent_" + entry["role"]))
        if i % 10 == 0:
            print(f"Parent frozen hashes verified: {i}/{len(old['files'])}", flush=True)
    metrics_rows = []
    for row in result["runs"]:
        initial, final = row["initial"], row["final"]
        recalls = "/".join(f"{100 * x['recall']:.1f}%" for x in final["per_interface"])
        metrics_rows.append(
            f"|{row['head']}|{initial['joint_nll']:.4f} → {final['joint_nll']:.4f}|"
            f"{initial['exact_match']:.2%} → {final['exact_match']:.2%}|{recalls}|")
    table = (
        "|Head|同样本联合NLL，初始→末轮|同样本三界面全部命中率|末轮I1/I2/I3召回率|\n"
        "|---|---|---|---|\n" + "\n".join(metrics_rows))
    small_table = (
        "|Head|2 mm正界面召回|新增边界正界面召回|原保守核心正界面召回|\n"
        "|---|---|---|---|\n" +
        "\n".join(f"|{g['head']}|{g['all_2mm']['recall']:.1%}|"
                  f"{g['new_2mm_boundary']['recall']:.1%}|{g['old_2mm_core']['recall']:.1%}|"
                  for g in group_rows))
    slot_rows = []
    for batch in summary["batches"]:
        for slot, counts in batch["per_small_slot"].items():
            slot_rows.append(f"|{batch['batch']}|{slot}|{counts['old_complete_positive']}|"
                             f"{counts['new_complete_positive']}|")
    slot_table = "|试块|2 mm槽位|v1完整标签点|v2完整标签点|\n|---|---|---|---|\n" + "\n".join(slot_rows)
    label_text = f"""# TC4四层图纸标签v2：仅放宽2 mm名义内部

用户确认“仅放宽2 mm小缺陷，其余保持”。以R1正面图及原配准为依据，仅将2 mm圆/方缺陷名义轮廓内及轮廓上的原未知点改为有效正标签；轮廓外缓冲区、4 mm缺陷、板边和1号I1两处争议区域排除保持。名义边界包含规则为圆距离平方≤半径平方、方形两轴绝对距离≤半边长；不外扩轮廓。mask=true代表用户接受的图纸监督假设，不新增实测边界精度保证。

998255条信号及其source_row不变。新增904条完整标签记录：完整标签960785条，正常955371条、含设计缺陷5414条，仍有37470条带未知mask。原有正常0/正1标签不修改。正常全量保留，训练阶段的1:1只为起始采样候选；正式split仍等后续批次。

{slot_table}

![边界前后对比](small_defect_boundary_comparison.png)

蓝点为v1完整正标签，橙点为v2新增完整正标签，灰点为其他位置。坐标为图纸正面；与扫查显示之间的左右对应仍使用原仿射配准，不再镜像标签数组。1号I1小缺陷位于先前排除区域，不重新纳入。

全量998255条完成旧mask重算、解析圆/方成员判定、原已知标签不变、正常集合不变、source_row/parent ID/排除集合核对。另有圆边界、圆外缓冲区、方角及大缺陷不受影响的定向测试。

运行入口：`tc4_small_defect_labels.py --profile <TC4四层_图纸标签_v2_小缺陷.json> --output <新目录>`。旧R1/v1保持可复查，当前版本入口为[冻结版R2](../../数据版本/TC4四层_冻结版_R2_20261020/README.md)。
"""
    (LABELS / "REPORT.md").write_text(label_text, encoding="utf-8")
    run_text = f"""# TC4同样本学习诊断

范围：真实RF、真实图纸监督的诊断性参数更新。正式train/validation/test未建立。训练与评分使用同一组512条样本，以下指标为回代诊断，不能解释成测试准确率、跨试块泛化、架构优劣或边界放宽带来的性能收益。

每块64条含缺陷＋64条正常，四块共256＋256；保留小缺陷和新增边界点覆盖，抽样无放回。四种Head使用相同512条、同seed 20261020和同一轮次顺序。FCN-1D，batch16，学习率0.001，12轮，每种Head384次更新；固定在0/1/3/6/12轮回代评分。无early stopping、无超参数搜索、未根据结果选择轮次。完整样本池不因本次抽样减少。

V3已滤波RF固定除131，不重复滤波、不去DC、不做逐条归一化。三界面窗口继续沿用已检查的诊断中心和±75点，按每条crop_start补偿；窗口全在记录内只说明覆盖存在，尚非最终物理TOF校准。坐标、试块ID和槽位ID仅用于选样与审计，不进入模型特征。

训练损失保留逐step total/primary/auxiliary。表中联合NLL由无标签exact推理的真实状态概率计算，四Head口径一致；界面分类用边际概率≥0.5，联合状态用MAP。初末均为eval模式在同一批固定样本上计算。推理不传真实标签，不使用teacher-forced分类成绩。

{table}

始终预测000的同样本基线：联合命中率{baseline['always_000_exact_match']:.1%}、
界面准确率{baseline['always_0_interface_accuracy']:.1%}，缺陷召回率0%。不要仅凭整体准确率判断学习。

下表只统计对应2 mm槽位的正界面，避免混入同一A扫的其他正常界面：

{small_table}

各组点数及命中数见`diagnostic_audit.json`；这些点均参与本次参数更新，不是独立验证边界的准确率。

![学习曲线](learning_curves.png)

![逐界面混淆矩阵](interface_confusion.png)

结果说明各Head在这512条上出现学习，尚不能证明泛化。应结合各界面、每块、2 mm样本和新增边界样本指标检查错误，不用单一整体准确率作为判断。`history.json`含按A扫分组的诊断；`diagnostic_audit.json`补充只统计2 mm正界面的召回；`steps.json`保存每一步损失，初末逐样本概率及MAP预测均保留。

梯度/损失有限、Backbone和Head确有更新、8路径概率归一、去掉标签后的推理不变、checkpoint重载预测完全相同均通过。工程权重标记为diagnostic_only，不作为正式预训练权重复用；全部样本ID保留，后续split应考虑本次已使用记录。无本次独立测试结果、正式M1/M2LF结论或双审晋升结论。

## 复跑

沿用已登记本机Conda环境。PowerShell执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'I:/programming/anaconda3/envs/aislee-a3-layercongition/python.exe' 'I:/programming/projects/a3-layerrecognition/scripts/tc4_learning_diagnostic.py' --config 'E:/新建文件夹/扩散焊数据集/处理规则/TC4四层_学习诊断_v1.json' --output '<新的输出目录>'
```

输出目录必须不存在，禁止覆盖本次结果。`config.json`、输入NPZ、epoch_orders、environment源码哈希和脚本副本共同确定本次运行；未安装依赖或更改环境。
"""
    (RUN / "REPORT.md").write_text(run_text, encoding="utf-8")
    release_text = f"""# TC4四层当前冻结版R2

ID：`TC4-4L-R2-small2mm-20261020`。本次只更新2 mm小缺陷标签，V3波形保持。R1及其标签v1留作对比，当前label_root指向v2。

- 信号998255条；完整标签960785条，正常955371、含设计缺陷5414，带mask37470。
- [标签规则、数量和对比图](../../处理结果/TC4四层_小缺陷边界标签_v2_20261020/REPORT.md)。
- [512条、12轮学习诊断及曲线](../../处理结果/TC4四层_学习诊断_v1_20261020/REPORT.md)。
- 设计依据：2026-10-16正面R1（用户所说的第三版）；与扫查显示的左右转换已在配准中，不再次镜像标签。
- 四块独立试块，1/2每层3 mm、3/4每层4 mm，正面入射；3号14 dB。1号两个争议区域保持排除。
- 全量正常池保留，1:1仅作起始采样候选；正式split等待后续批次。

`release.json`绑定父版全部58文件及新标签/诊断证据；父版58文件在本次发布时完整重新核对哈希。不复制大波形，不删除任何R1内容。标签放宽是用户接受的名义图纸监督，不提升边界实测可信度。诊断指标均为同样本回代，非测试性能。任何后续规则变更继续另增版本。
"""
    (R2 / "README.md").write_text(release_text, encoding="utf-8")
    code_dir = R2 / "code"
    code_dir.mkdir()
    for name in ("tc4_small_defect_labels.py", "tc4_learning_diagnostic.py",
                 "tc4_publish_learning_diagnostic.py"):
        shutil.copy2(Path(__file__).with_name(name), code_dir / name)
    for path in [R1 / "release.json", *LABELS.rglob("*"), *RUN.rglob("*"),
                 *code_dir.glob("*.py"),
                 DATA / "处理规则/TC4四层_图纸标签_v2_小缺陷.json",
                 DATA / "处理规则/TC4四层_学习诊断_v1.json"]:
        if path.is_file():
            files.append(dict(path=path.relative_to(DATA).as_posix(),
                              bytes=path.stat().st_size, sha256=digest(path),
                              role="label_v2_or_learning_evidence"))
    release = dict(release_id="TC4-4L-R2-small2mm-20261020", status="frozen_unsplit",
                   parent_release=str(R1 / "release.json"), parent_sha256=digest(R1 / "release.json"),
                   signal_root=old["signal_root"], label_root=str(LABELS),
                   diagnostic_root=str(RUN), signal_profile=old["signal_profile"],
                   label_profile=summary["version"], total_records=998255,
                   complete_labels=960785, masked_records=37470,
                   state_counts=summary["state_counts"],
                   formal_split_created=False, files=files)
    write_json(R2 / "release.json", release)
    (R2 / "release.sha256").write_text(digest(R2 / "release.json") + "  release.json\n", encoding="ascii")
    write_json(R2 / "release_validation.json", dict(status="passed",
        parent_full_hashes_rechecked=len(old["files"]), bound_files=len(files),
        records_geometry_audited=998255, new_complete_records=904,
        same_cohort_records=512, per_head_updates=384, formal_split_created=False))

    section = f"""## 2026-10-20 当前：2 mm名义内部放宽v2与学习诊断

用户确认仅放宽2 mm小缺陷，其余保持：名义轮廓内（含边界）取消内部保守mask，轮廓外缓冲区、其他尺寸和1号已排除区域保持。按图纸监督假设纳入，不把名义边界当实测精度。图纸仍为2026-10-16正面R1，扫描显示左右转换沿用原配准。

当前[冻结版R2](<{R2.as_posix()}/README.md>)绑定V3波形＋标签v2；R1保留。新增904条完整标签，当前960785条完整（955371正常、5414含设计缺陷），37470条带mask，总998255条不变。正常池全保留，后续在训练实体内轮换采样，1:1为起始候选。

512条固定样本、四Head各12轮/384次更新的[学习诊断](<{RUN.as_posix()}/REPORT.md>)已完成，曲线、逐界面混淆矩阵与每块/小缺陷/新增边界组指标已保存。训练和评分使用同一批样本，仅回代诊断，正式split继续等待后续批次，不声称测试性能、泛化、Head优劣或标签策略增益；诊断权重不复用于正式预训练。完整M1/M2LF和最终时间先验仍待完成。

"""
    status = f"""# A3 当前状态

更新：2026-10-20。本页为当前入口；历史依据保留在合同、版本报告及实验日志。

|项目|当前状态|
|---|---|
|研究主线|固定界面先验＋signed RF，逐界面0/1及联合状态推断；V1.3四Head|
|本批实体|四块独立TC4四层；1/2每层3 mm、3/4每层4 mm，正面入射，3号14 dB|
|当前数据|TC4-4L-R2-small2mm-20261020；V3波形＋标签v2，父R1保留|
|设计与显示|2026-10-16正面R1；第三版已左右镜像，扫描显示转换单独由原配准处理|
|信号/标签|998255条；960785条完整标签，正常955371、含设计缺陷5414；37470条mask|
|边界规则|只放宽2 mm名义内部及边界；外侧缓冲、4 mm与板边保持，新增904条完整标签|
|争议区域|1号I1两处仍整体排除，不重纳入、不交换制造标签|
|处理|前裁剪→20 MHz低通→底波后截尾，V3保持；输入固定除131，保留DC与signed RF|
|采样/划分|完整正常池保留；1:1为起始候选，正式实体split等后续批次|
|学习诊断|512条同样本拟合与回代，四Head各12轮/384更新；曲线、混淆矩阵和分组结果保存|
|验证|新增标签全量几何/身份检查；父R1全部58哈希重查；诊断有限梯度、无标签推理、复载通过|
|性能边界|回代指标非测试准确率，未证明泛化/Head优劣/边界放宽收益；权重不作正式预训练|
|待完成|后续批次、正式split与训练配置、最终时间先验、完整M1/M2LF和跨实体实验|
|存储|本轮无删除；此前28个可重建旧数组清理记录保持，原始TXT/NDT保留|

[当前冻结版R2](<{R2.as_posix()}/README.md>) · [诊断报告](<{RUN.as_posix()}/REPORT.md>) · [研究合同](idea-stage/docs/research_contract.md)。
"""
    replacements = {}
    replacements[A3 / "PROJECT_STATUS.md"] = status
    for path in (A3 / "idea-stage/docs/research_contract.md",
                 A3 / "refine-logs/EXPERIMENT_TRACKER.md"):
        text = path.read_text("utf-8")
        title, rest = text.split("\n", 1)
        replacements[path] = title + "\n\n" + section + rest
    for path in (A3 / "findings.md", A3 / "refine-logs/EXPERIMENT_LOG.md",
                 DATA / "处理规则/统一信号处理规则.md"):
        replacements[path] = path.read_text("utf-8").rstrip() + "\n\n" + section
    rule_path = DATA / "处理规则/统一信号处理规则.md"
    rule_text = replacements[rule_path]
    title, rest = rule_text.split("\n", 1)
    replacements[rule_path] = title + (
        "\n\n当前标签附则为v2：仅2 mm名义轮廓内及边界取消保守mask；"
        "外侧缓冲区与其他位置保持。见文末当前记录及"
        "[v2机器配置](TC4四层_图纸标签_v2_小缺陷.json)。"
        "下方v1内缩数值为继承基线，2 mm内部由v2覆盖；波形仍为V3。\n") + rest
    path = A3 / "RESEARCH_DECISIONS.md"
    text = path.read_text("utf-8")
    title, rest = text.split("\n", 1)
    replacements[path] = title + "\n\n" + section + rest
    path = DATA / "信号数据索引.md"
    text = path.read_text("utf-8")
    title, rest = text.split("\n", 1)
    replacements[path] = title + "\n\n" + section + rest.replace(
        "## 当前固定数据入口", "## 历史冻结版R1入口（标签已由v2更新）", 1).replace(
        "为后续使用入口，清单绑定当前V3波形及标签v1", "为历史对照入口，清单绑定V3波形及标签v1", 1).replace(
        "## 当前标签与未划分样本池", "## 历史标签v1与未划分样本池", 1)
    path = DATA / "AGENTS.md"
    text = path.read_text("utf-8")
    old_line = "- 本批已固定为[TC4四层冻结版R1](数据版本/TC4四层_冻结版_R1_20261020/README.md)。"
    assert old_line in text
    replacements[path] = text.replace(old_line,
        "- 当前入口为[TC4四层冻结版R2](数据版本/TC4四层_冻结版_R2_20261020/README.md)：V3波形＋仅放宽2 mm内部的标签v2；R1作为历史对照保留。")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    research_backup = A3.parent / "_整理备份" / f"A3_小缺陷标签与诊断_{stamp}"
    data_backup = R2 / "documents_before_update"
    research_backup.mkdir(parents=True)
    data_backup.mkdir()
    changes = []
    for i, (path, new_text) in enumerate(replacements.items(), 1):
        content = path.read_bytes()
        old_hash = hashlib.sha256(content).hexdigest()
        backup_dir = research_backup if path.is_relative_to(A3) else data_backup
        backup = backup_dir / f"{i}_{path.name}"
        backup.write_bytes(content)
        assert digest(path) == old_hash
        path.write_text(new_text, encoding="utf-8")
        changes.append(dict(path=str(path), backup=str(backup),
                            before_sha256=old_hash, after_sha256=digest(path)))
    write_json(R2 / "document_updates.json", changes)
    print(f"R2 bound {len(files)} files; updated {len(changes)} documents with backups.", flush=True)


if __name__ == "__main__":
    main()
