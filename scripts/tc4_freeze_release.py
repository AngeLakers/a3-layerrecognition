"""Freeze TC4 data by hashes, inspect provenance and draw actual label masks.

No deletion occurs here. A separate reviewed PowerShell step consumes the exact
cleanup plan only after the release and rerun checks succeed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import numpy as np

ROOT = Path(r"E:\新建文件夹\扩散焊数据集")
RUNS = ROOT / "处理结果"
SIGNAL = RUNS / "TC4四层_底波后裁剪_v3_20261020"
LABEL = RUNS / "TC4四层_图纸标签与小流程_v1_20261020"
V1 = RUNS / "TC4四层_四批质量与裁剪_20261020"
V2 = RUNS / "TC4四层_时间裁剪与LP20_v2_20261020"
PARENT = RUNS / "TC4四层_时间裁剪与LP20_v2_1_20261020"
REPO = Path(r"I:\programming\projects\a3-layerrecognition")


def read(path):
    return json.loads(path.read_text("utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(1024 * 1024):
            h.update(buf)
    return h.hexdigest()


def file_record(path, role):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=digest(path), role=role)


def prepare(dest):
    if (dest / "release.json").exists():
        raise FileExistsError("Release already exists; verify instead.")
    provenance = dest / "provenance"
    provenance.mkdir(exist_ok=True)
    records, source_checks, candidates = [], [], []
    for b in range(1, 5):
        audit = read(V1 / f"batch{b}" / "audit.json")
        original = Path(audit["source"])
        # Full filename/size check against retained original TXT manifest.
        actual = {f.name: f.stat().st_size for f in os.scandir(original)
                  if f.is_file() and f.name.lower().endswith(".txt")}
        manifest = V1 / f"batch{b}" / "source_manifest.csv"
        pick = set(np.random.default_rng(20261020 + b).choice(audit["files"], 32, replace=False))
        count, source_hash_checks = 0, 0
        cache = np.load(V1 / f"batch{b}" / "raw_cache.npy", mmap_mode="r")
        with manifest.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                assert actual.pop(row["filename"], None) == int(row["size"])
                if int(row["row"]) in pick:
                    data = (original / row["filename"]).read_bytes()
                    assert hashlib.sha256(data).hexdigest() == row["sha256"]
                    assert np.array_equal(np.array(data.split(), dtype=np.int16),
                                          cache[int(row["row"])])
                    source_hash_checks += 1
                count += 1
        assert not actual and count == audit["files"]
        del cache
        idx = dict(np.load(SIGNAL / f"batch{b}" / "index.npz"))
        pidx = dict(np.load(PARENT / f"batch{b}" / "index.npz"))
        # Current complete output versus parent: check every retained integer
        # and filtered sample before authorizing parent-array cleanup.
        for name in ("rf_cropped_int16.npy", "rf_lp20_float32.npy"):
            current = np.load(SIGNAL / f"batch{b}" / name, mmap_mode="r")
            parent = np.load(PARENT / f"batch{b}" / name, mmap_mode="r")
            for i, p in enumerate(idx["parent_sample_id"]):
                lo, hi = idx["offsets"][i:i+2]
                po = pidx["offsets"][p]
                assert np.array_equal(current[lo:hi], parent[po:po+hi-lo])
            assert len(current) == idx["offsets"][-1]
            del current, parent
            records.append(file_record(SIGNAL / f"batch{b}" / name, "frozen_waveform"))
        for path, role in [
            (SIGNAL / f"batch{b}" / "index.npz", "signal_index"),
            (SIGNAL / f"batch{b}" / "parent_selection.npz", "exclusion_provenance"),
            (SIGNAL / f"batch{b}" / "manifest.json", "signal_provenance"),
            (LABEL / f"batch{b}" / "labels.npz", "frozen_labels"),
            (LABEL / f"batch{b}" / "manifest.json", "label_provenance"),
            (LABEL / f"batch{b}" / "unsplit_sampling_example.npz", "sampling_example_not_split"),
            (manifest, "original_txt_sha256_manifest"),
        ]:
            records.append(file_record(path, role))
        source_checks.append(dict(batch=b, original_path=str(original), file_sizes_checked=count,
                                  source_sha256_and_cache_checks=source_hash_checks,
                                  full_parent_slice_checks=len(idx["source_row"])))
        # Explicit allowlist of obsolete arrays; keep all manifests/indexes.
        obsolete = [(V1 / f"batch{b}" / "raw_cache.npy", "TXT-derived cache; originals retained"),
                    (V1 / f"batch{b}" / "cropped" / "rf_values.npy", "superseded V1 crop"),
                    (V1 / f"batch{b}" / "cropped_initial_review" / "rf_values.npy",
                     "superseded early crop")]
        for version in (V2, PARENT):
            obsolete.extend((version / f"batch{b}" / name, "superseded intermediate waveform")
                            for name in ("rf_cropped_int16.npy", "rf_lp20_float32.npy"))
        for path, reason in obsolete:
            record = file_record(path, reason)
            record["absolute_path"] = str(path.resolve())
            record["mtime_ns"] = path.stat().st_mtime_ns
            candidates.append(record)
        print(f"batch{b}: {count} source sizes, {len(idx['source_row'])} full slices verified; hashes recorded",
              flush=True)
    snapshots = [ROOT / "处理规则" / n for n in (
        "TC4四层_质量裁剪_v1.json", "TC4四层_时间裁剪与LP20_v2_1.json",
        "TC4四层_底波后裁剪_v3.json", "TC4四层_图纸标签_v1.json")]
    snapshots += [LABEL / n for n in ("profile.json", "summary.json", "smoke_selection.json",
                                      "label_validation.json", "delivery_validation.json")]
    geom = Path(read(LABEL / "profile.json")["geometry"])
    snapshots.append(geom)
    snapshots += list((PARENT / "K3_registration").glob("batch*_registration.json"))
    for path in snapshots:
        target = provenance / path.name
        if target.exists():
            assert target.read_bytes() == path.read_bytes()
        else:
            shutil.copyfile(path, target)
        records.append(file_record(target, "frozen_provenance_snapshot"))
    for name in ("tc4_label_pool.py", "tc4_label_audit.py",
                 "tc4_unsplit_smoke.py", "tc4_freeze_release.py"):
        target = provenance / "code" / name
        shutil.copyfile(REPO / "scripts" / name, target)
        records.append(file_record(target, "code_snapshot"))
    for path in [SIGNAL / "load_signal.py", SIGNAL / "final_validation.json",
                 LABEL / "profile.json", LABEL / "smoke_selection.json"]:
        records.append(file_record(path, "runtime_or_validation_dependency"))
    summary = read(LABEL / "summary.json")
    release = dict(release_id="TC4-4L-R1-20261020", status="frozen_unsplit",
                   data_root=str(ROOT), signal_root=str(SIGNAL), label_root=str(LABEL),
                   signal_profile="tc4-four-layer-post-backwall-crop-v3",
                   label_profile="tc4-four-layer-drawing-labels-v1-20261020",
                   total_records=summary["total_records"], complete_labels=summary["complete_labels"],
                   state_counts=summary["state_counts"], masked_records=summary["masked_records"],
                   split_status="deferred_by_user_until_later_batches",
                   mutation_rule="Do not overwrite payloads; a changed rule or label needs a new release.",
                   waveform_units="original RF integer units; LP20 float32 remains unnormalized",
                   normalization="fixed divisor 131 only when constructing engineering tensors; no refilter",
                   sources=source_checks, files=records)
    write(dest / "release.json", release)
    write(dest / "cleanup_plan.json", dict(status="prepared_not_deleted", candidates=candidates,
                                          files=len(candidates), bytes=sum(r["bytes"] for r in candidates),
                                          allowed_root=str(RUNS.resolve()),
                                          excludes="all original TXT/NDT, current V3, labels, indexes, reports"))
    (dest / "release.sha256").write_text(digest(dest / "release.json") + "  release.json\n", "ascii")
    print("Freeze prepared; no files deleted.", flush=True)


def verify(dest):
    assert digest(dest / "release.json") == (dest / "release.sha256").read_text().split()[0]
    release = read(dest / "release.json")
    for r in release["files"]:
        p = ROOT / r["path"]
        assert p.stat().st_size == r["bytes"] and digest(p) == r["sha256"], str(p)
    write(dest / "release_validation.json",
          dict(status="passed", files=len(release["files"]), full_hash_verification=True,
               release_sha256=digest(dest / "release.json"), records=release["total_records"]))
    print(f"Release full hashes verified: {len(release['files'])} files", flush=True)


def plots(dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Circle, Polygon
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    out = dest / "figures"
    out.mkdir(exist_ok=False)
    geometry = read(dest / "provenance" / "geometry.json")
    slots = [s for s in geometry["slots"] if s["design"] == "K3"]
    colors = ["#c5c9ce", "#dcecf5", "#d95335"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -.5, .5, 1.5], 3)
    fig, axs = plt.subplots(4, 3, figsize=(14, 17), layout="constrained")
    jointfig, jointaxs = plt.subplots(2, 2, figsize=(12, 11), layout="constrained")
    jointcolors = ["#c5c9ce", "#dcecf5", "#009e73", "#e69f00",
                   "#56b4e9", "#cc79a7", "#0072b2", "#d55e00", "#8b5e3c"]
    jmap = ListedColormap(jointcolors)
    jnorm = BoundaryNorm(np.arange(-1.5, 8.5), 9)

    def contour(ax, slot):
        if slot["front_vertices"]:
            ax.add_patch(Polygon(slot["front_vertices"], fill=False, ec="#333333", lw=.8))
        else:
            ax.add_patch(Circle((slot["front_x"], slot["front_y"]), slot["size"]/2,
                                fill=False, ec="#333333", lw=.8))

    def scatter_values(ax, xy, value, *, joint=False):
        # Every nonzero/masked point is shown; normal background thinned only
        # for plotting. Stored labels are never resampled.
        selected = (value != 0) | (np.arange(len(value)) % 5 == 0)
        ax.scatter(xy[selected, 0], xy[selected, 1], c=value[selected], s=.6,
                   cmap=jmap if joint else cmap, norm=jnorm if joint else norm,
                   rasterized=True, linewidths=0)
        ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal",
               xlabel="设计正面 X / mm", ylabel="设计正面 Y / mm")

    for b in range(1, 5):
        lab = np.load(LABEL / f"batch{b}" / "labels.npz")
        xy, y, joint = lab["xy_front_mm"], lab["y"], lab["joint_state"]
        reg = read(dest / "provenance" / f"batch{b}_registration.json")
        sel = np.load(SIGNAL / f"batch{b}" / "parent_selection.npz")
        excluded = sel["disputed_spatial_exclusion"]
        affine = np.asarray(reg["coefficients"])
        excluded_xy = ((sel["parent_xy_local_mm"][excluded] - affine[2])
                       @ np.linalg.inv(affine[:2]))
        for k in range(3):
            ax = axs[b-1, k]
            scatter_values(ax, xy, y[:, k])
            for s in slots:
                if s["interface"] != k+1:
                    continue
                contour(ax, s)
                ax.text(s["front_x"] + 2, s["front_y"] + 2, s["id"].replace("K3-", ""),
                        fontsize=7)
            if len(excluded_xy):
                ax.scatter(excluded_xy[::5, 0], excluded_xy[::5, 1], c="#252525", s=1)
            ax.set_title(f"{b}号 / I{k+1}：1标签 {(y[:,k]==1).sum():,} 点", fontsize=12)
        ax = jointaxs.flat[b-1]
        scatter_values(ax, xy, joint, joint=True)
        for x, yy in sorted({(s["front_x"], s["front_y"]) for s in slots}):
            ax.annotate(f"({x:g},{yy:g})", (x, yy), xytext=(3, 6),
                        textcoords="offset points", fontsize=7)
        if len(excluded_xy):
            ax.scatter(excluded_xy[::4, 0], excluded_xy[::4, 1], c="#252525", s=1)
            for yy in (22, 78):
                ax.annotate("已排除", (86, yy), xytext=(-35, -18),
                            textcoords="offset points", fontsize=9)
        ax.set_title(f"{b}号：完整标签中的缺陷点 {(joint>0).sum():,}", fontsize=13)
        detail, dax = plt.subplots(3, 5, figsize=(15, 9), layout="constrained")
        for k in range(3):
            for j, s in enumerate([s for s in slots if s["interface"] == k+1]):
                ax = dax[k, j]
                cx, cy = s["front_x"], s["front_y"]
                near = (np.abs(xy - [cx, cy]) <= 4).all(1)
                ax.scatter(xy[near, 0], xy[near, 1], c=y[near, k], s=9,
                           cmap=cmap, norm=norm, linewidths=0)
                contour(ax, s)
                if len(excluded_xy):
                    e = (np.abs(excluded_xy - [cx, cy]) <= 4).all(1)
                    ax.scatter(excluded_xy[e, 0], excluded_xy[e, 1], c="#252525", s=6)
                ax.set(xlim=(cx-4, cx+4), ylim=(cy-4, cy+4), aspect="equal",
                       title=f"{s['id'].replace('K3-','')} / {s['size']:g} mm")
        detail.suptitle(f"{b}号实际标签点放大：黑线=设计轮廓，红=1，蓝=0，灰=未知，黑点=排除",
                        fontsize=15)
        detail.savefig(out / f"batch{b}_label_details.png", dpi=160)
        plt.close(detail)
    fig.suptitle("四块 × 三界面：实际标签位置（设计正面坐标）", fontsize=20)
    fig.legend([Patch(color=c) for c in colors+["#252525"]],
               ["未知 / 不确定边缘", "0：无设计缺陷参考区", "1：缺陷内部", "已排除，不在样本中"],
               loc="outside lower center", ncol=4)
    fig.savefig(out / "four_specimens_three_interfaces.png", dpi=150)
    plt.close(fig)
    jointfig.suptitle("联合状态标签：只有三个界面都可靠时才有完整状态", fontsize=17)
    jointfig.legend([Patch(color=c) for c in jointcolors+["#252525"]],
                    ["未知"]+[f"{s:03b}" for s in range(8)]+["已排除"],
                    loc="outside lower center", ncol=5)
    jointfig.savefig(out / "four_specimens_joint_labels.png", dpi=160)
    plt.close(jointfig)
    print("6 label figures written", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "plots"])
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.release)
