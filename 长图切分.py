#!/usr/bin/env python3
"""
长图切分工具

把竖向拼接的长图（作品集 / 游戏 UI 展示图等）按"自然分隔带"切成若干张。

切分逻辑：
  1. 逐行计算亮度标准差和横向梯度，找出"安静行"（纯色/近纯色的整行）。
  2. 连续的安静行组成分隔带，带宽 >= min_gap 的作为"自然切口"（切在带中间）。
  3. 两块内容无空隙直接拼接时，交界处整行像素突变，记为"拼接边界"切口。
  4. 用动态规划选切口：尽量多用自然切口，每段高宽比落在 [min_ratio, max_ratio]；
     某一段超过 max_ratio 又没有可用切口时，才在内容最少的行"强制切"。

安全性：
  - 原图只读，绝不修改、移动、删除源目录中的任何文件。
  - 每张图单独一个输出子目录，只放切片。
  - 输出目录不允许与源目录相同；已存在的子目录默认跳过（--force 才覆盖本工具生成的文件）。
  - 所有文件先写临时文件再原子改名，中途中断不会留下半截文件。

用法：
  python3 长图切分.py <图片或文件夹> [更多...] [-o 输出目录] [选项]
  python3 长图切分.py --help
"""

import argparse
import csv
import os
import re
import sys
import time

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 400_000_000

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ANALYSIS_WIDTH = 480
LOG_NAME = "切分记录.csv"


# ---------------------------------------------------------------- 分析

def analysis_array(im):
    """转为灰度并把宽度缩到 ANALYSIS_WIDTH（高度保持原样，保证切口行坐标精确）。"""
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        gray = bg.convert("L")
    else:
        gray = im.convert("RGB").convert("L")
    if gray.width > ANALYSIS_WIDTH:
        gray = gray.resize((ANALYSIS_WIDTH, gray.height), Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def find_runs(mask):
    """返回 mask 中连续 True 段的 (start, end)，end 为开区间。"""
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def plan_cuts(im, args):
    """返回切口列表 [(y, kind, gap_len)]，kind 为 'natural' 或 'forced'。"""
    W, H = im.size
    a = analysis_array(im)
    row_std = a.std(axis=1)
    row_grad = np.abs(np.diff(a, axis=1)).mean(axis=1)

    quiet = (row_std < args.std_thr) & (row_grad < args.grad_thr)
    min_gap = max(4, int(round(args.min_gap * W)))
    min_h = args.min_ratio * W
    max_h = args.max_ratio * W

    cands = []  # (y, cost, kind, gap_len)
    natural_mask = np.zeros(H, dtype=bool)
    strict_cuts = []
    for s, e in find_runs(quiet):
        if s == 0 or e == H or e - s < min_gap:
            continue
        gap = e - s
        strength = min(1.0, gap / (0.1 * W))
        cut_y = (s + e) // 2
        cands.append((cut_y, -(1.0 + strength), "natural", gap))
        strict_cuts.append(cut_y)
        natural_mask[s:e] = True

    vertical_diff = np.abs(np.diff(a, axis=0))
    edge_frac = (vertical_diff > 15).mean(axis=1)
    faint_edge_frac = (vertical_diff > 2).mean(axis=1)

    # 渐变或带轻微纹理的空白带：先把被少量噪声打断的低纹理行合并，
    # 再取带的中间作为安全切口。过宽的低纹理区通常是画面本身（天空、
    # 雾面背景等），不能当作分隔带。
    soft_quiet = (row_std < 35.0) & (row_grad < 1.25)
    join_gap = max(3, int(round(0.012 * W)))
    merged = []
    for s, e in find_runs(soft_quiet):
        if merged and s - merged[-1][1] <= join_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    soft_blocks = []
    major_gap_cuts = []
    for s, e in merged:
        gap = e - s
        if (s == 0 or e == H or gap < min_gap or gap > 0.2 * W
                or natural_mask[s:e].any()
                or any(abs((s + e) // 2 - y) < min_h for y in strict_cuts)):
            continue
        cut_y = (s + e) // 2
        snap_strength = 0.0
        if gap >= 0.12 * W and faint_edge_frac[s - 1] >= 0.1:
            # 宽章节分隔区通常有一条原生顶边线；切在线上，而不是空白中央。
            cut_y = s
            major_gap_cuts.append(cut_y)
        elif gap <= 0.06 * W:
            # 短过渡带附近若先出现一条整行突变，通常那才是两张图的
            # 真正边界。带内只接受极强突变，避免吸附到下一张图的状态栏。
            lo, hi = max(1, s - join_gap), min(H - 1, e)
            before = np.flatnonzero(
                edge_frac[lo - 1:min(hi, s + 2)] >= args.edge_thr
            )
            strong = np.flatnonzero(edge_frac[s - 1:hi] >= 0.8)
            if before.size:
                cut_y = lo + int(before[0])
                snap_strength = float(edge_frac[cut_y - 1])
            elif strong.size:
                cut_y = s + int(strong[0])
                snap_strength = float(edge_frac[cut_y - 1])
            soft_blocks.append((max(0, s - join_gap), min(H, e + join_gap)))
        strength = min(0.8, gap / (0.1 * W))
        cands.append((
            cut_y, -(0.6 + strength + 0.7 * snap_strength), "natural", gap
        ))

    # 若开头到第一条宽章节分隔线本身就没有超长，之前较窄的暗色留白
    # 多半只是标题、文字、色板等版式内部间距，不应被切开。
    intro_dividers = [y for y in major_gap_cuts if y <= max_h]
    if intro_dividers:
        intro_y = min(intro_dividers)
        cands = [
            (y, max(cost, 0.4) if kind == "natural" and y < intro_y else cost,
             kind, gap)
            for y, cost, kind, gap in cands
        ]

    # 相邻两块内容无空隙直接拼接时，交界处整行像素会突变
    win = max(2, int(0.005 * W))
    edge_peaks = []
    for i in np.flatnonzero(edge_frac >= args.edge_thr):
        y = int(i) + 1
        near_strict_gap = any(abs(y - cut) < min_h for cut in strict_cuts)
        near_soft_gap = any(s <= y <= e for s, e in soft_blocks)
        if y < min_gap or y > H - min_gap or near_strict_gap or near_soft_gap:
            continue
        lo, hi = max(0, i - win), min(len(edge_frac), i + win + 1)
        if edge_frac[i] < edge_frac[lo:hi].max():
            continue
        edge_peaks.append((y, float(edge_frac[i])))

    # 同一条拼接边界可能形成几条相距很近的横线，只保留最前面一条，
    # 置信度取这一组中的最大值。
    edge_groups = []
    cluster_gap = max(win * 2, int(round(0.015 * W)))
    for y, strength in edge_peaks:
        if edge_groups and y - edge_groups[-1][-1][0] <= cluster_gap:
            edge_groups[-1].append((y, strength))
        else:
            edge_groups.append([(y, strength)])
    for group in edge_groups:
        y = group[0][0]
        strength = max(v for _, v in group)
        # 超过 80% 的列同时突变，基本可以确认是两张图的拼接缝，必须奖励；
        # 较弱的横线仍保留为备选，避免把界面内部的状态栏误当成切口。
        cost = (-0.8 - 4.0 * (strength - 0.8)
                if strength >= 0.8 else 1.0 - 0.5 * strength)
        cands.append((y, cost, "edge", 0))

    busy = row_grad + 0.3 * row_std
    k = 9
    busy = np.convolve(busy, np.ones(k) / k, mode="same")
    step = max(8, int(max_h / 8))
    for b0 in range(step, H - step // 2, step):
        seg = busy[b0 - step // 2: b0 + step // 2]
        y = b0 - step // 2 + int(np.argmin(seg))
        cands.append((y, 3.0 + float(busy[y]) / 5.0, "forced", 0))

    cands.sort(key=lambda c: c[0])
    ys = [0] + [c[0] for c in cands] + [H]
    costs = [0.0] + [c[1] for c in cands] + [0.0]
    n = len(ys)

    INF = float("inf")
    dp = [INF] * n
    prev = [-1] * n
    dp[0] = 0.0
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            h = ys[j] - ys[i]
            if h > max_h:
                break
            if h <= 0 or dp[i] == INF:
                continue
            seg_cost = 0.0 if h >= min_h else 10.0 + (min_h - h) / W
            v = dp[i] + seg_cost + costs[j]
            if v < dp[j]:
                dp[j] = v
                prev[j] = i

    if dp[-1] == INF:
        raise RuntimeError("找不到满足约束的切分方案")

    chain = []
    j = n - 1
    while prev[j] > 0:
        j = prev[j]
        chain.append(j)
    chain.reverse()
    return [(ys[j], cands[j - 1][2], cands[j - 1][3]) for j in chain]


# ---------------------------------------------------------------- 文件读写

def save_image(img, path, fmt, info):
    kw = {}
    if info.get("icc_profile"):
        kw["icc_profile"] = info["icc_profile"]
    if fmt == "JPEG":
        if img.mode not in ("RGB", "L", "CMYK"):
            img = img.convert("RGB")
        kw.update(quality=95, subsampling=0, optimize=True)
    elif fmt == "WEBP":
        kw.update(quality=95, method=4)
    elif fmt == "PNG":
        kw.update(compress_level=6)
    tmp = path + ".part"
    img.save(tmp, format=fmt, **kw)
    os.replace(tmp, path)


CUT_MARKS = {"natural": "", "edge": "~", "forced": "*"}


# ---------------------------------------------------------------- 主流程

def collect_inputs(paths, recursive):
    files = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            if recursive:
                for root, dirs, names in os.walk(p):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and not d.endswith("切分结果")]
                    files += [os.path.join(root, n) for n in sorted(names)]
            else:
                files += [os.path.join(p, n) for n in sorted(os.listdir(p))]
        else:
            print("  ! 路径不存在：%s" % p)
    return [f for f in files
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
            and not os.path.basename(f).startswith(".")]


def default_output_root(first_input):
    base = first_input.rstrip(os.sep)
    if os.path.isfile(base):
        base = os.path.dirname(base)
    return base + "切分结果"


def unique_dirname(stem, ext, used):
    name = stem
    if name in used:
        name = "%s_%s" % (stem, ext.lstrip(".").lower())
    n = 2
    while name in used:
        name = "%s_%s_%d" % (stem, ext.lstrip(".").lower(), n)
        n += 1
    used.add(name)
    return name


def process_one(src, out_dir, args):
    with Image.open(src) as raw:
        raw.load()
        fmt = raw.format
        info = dict(raw.info)
        im = ImageOps.exif_transpose(raw)
        if im is raw:
            im = raw.copy()
    W, H = im.size
    ratio = H / W

    if ratio <= args.max_ratio:
        return {"status": "跳过(不够长)", "W": W, "H": H, "cuts": []}

    cuts = plan_cuts(im, args)
    if args.dry_run:
        return {"status": "预演", "W": W, "H": H, "cuts": cuts}

    os.makedirs(out_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(src))
    generated = re.compile(r"^%s_(?:\d{2}|00原图|对照预览)\.[^.]+$" % re.escape(stem))
    for old_name in os.listdir(out_dir):
        old_path = os.path.join(out_dir, old_name)
        if generated.match(old_name) and os.path.isfile(old_path):
            os.remove(old_path)

    bounds = [0] + [c[0] for c in cuts] + [H]
    total = 0
    for idx in range(len(bounds) - 1):
        piece = im.crop((0, bounds[idx], W, bounds[idx + 1]))
        piece_path = os.path.join(out_dir, "%s_%02d%s" % (stem, idx + 1, ext))
        save_image(piece, piece_path, fmt, info)
        with Image.open(piece_path) as chk:
            if chk.size != (W, bounds[idx + 1] - bounds[idx]):
                raise RuntimeError("切片尺寸校验失败：%s" % piece_path)
            total += chk.height
    if total != H:
        raise RuntimeError("切片总高度 %d 与原图高度 %d 不一致" % (total, H))

    return {"status": "完成", "W": W, "H": H, "cuts": cuts}


def main():
    ap = argparse.ArgumentParser(description="长图切分工具（原图只读，输出到独立目录）")
    ap.add_argument("inputs", nargs="+", help="图片文件或文件夹")
    ap.add_argument("-o", "--output", help="输出根目录（默认：<输入文件夹>切分结果）")
    ap.add_argument("-r", "--recursive", action="store_true", help="递归处理子文件夹")
    ap.add_argument("--max-ratio", type=float, default=1.5,
                    help="单片最大高宽比，超过才切；原图高宽比不超过它则跳过（默认 1.5）")
    ap.add_argument("--min-ratio", type=float, default=0.3,
                    help="单片最小高宽比，防止切出细条（默认 0.3）")
    ap.add_argument("--min-gap", type=float, default=0.012,
                    help="自然分隔带最小高度，占图宽比例（默认 0.012，1920 宽约 23px）")
    ap.add_argument("--std-thr", type=float, default=6.0, help="安静行亮度标准差阈值（默认 6）")
    ap.add_argument("--grad-thr", type=float, default=2.0, help="安静行横向梯度阈值（默认 2）")
    ap.add_argument("--edge-thr", type=float, default=0.45,
                    help="拼接边界判定：整行发生突变的列占比（默认 0.45）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（试跑用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印切分方案，不写任何文件")
    ap.add_argument("--force", action="store_true", help="输出子目录已存在时覆盖（默认跳过）")
    args = ap.parse_args()

    if args.min_ratio >= args.max_ratio:
        sys.exit("min-ratio 必须小于 max-ratio")

    files = collect_inputs(args.inputs, args.recursive)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        sys.exit("没有找到可处理的图片")

    out_root = os.path.abspath(args.output or default_output_root(os.path.abspath(args.inputs[0])))
    src_dirs = {os.path.dirname(f) for f in files}
    if out_root in src_dirs:
        sys.exit("输出目录不能和原图所在目录相同：%s" % out_root)

    print("共 %d 张图，输出到：%s%s" % (len(files), out_root, "（预演，不写文件）" if args.dry_run else ""))
    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)

    log_path = os.path.join(out_root, LOG_NAME)
    log_rows = []
    used = set()
    stats = {}
    t0 = time.time()
    for i, src in enumerate(files, 1):
        name = os.path.basename(src)
        stem, ext = os.path.splitext(name)
        sub = unique_dirname(stem, ext, used)
        out_dir = os.path.join(out_root, sub)
        prefix = "[%d/%d] %s" % (i, len(files), name)

        if os.path.exists(out_dir) and not args.force and not args.dry_run:
            print("%s  已存在输出，跳过（--force 可覆盖）" % prefix)
            stats["跳过(已存在)"] = stats.get("跳过(已存在)", 0) + 1
            continue
        try:
            r = process_one(src, out_dir, args)
        except Exception as e:  # noqa: BLE001 单张失败不影响其它
            print("%s  失败：%s" % (prefix, e))
            stats["失败"] = stats.get("失败", 0) + 1
            log_rows.append([name, "", "", "失败: %s" % e, "", "", "", src])
            continue

        cuts = r["cuts"]
        n_kind = {k: sum(1 for c in cuts if c[1] == k) for k in CUT_MARKS}
        stats[r["status"]] = stats.get(r["status"], 0) + 1
        if r["status"].startswith("跳过"):
            print("%s  %dx%d  高宽比 %.2f，不够长，跳过" % (prefix, r["W"], r["H"], r["H"] / r["W"]))
        else:
            print("%s  %dx%d → %d 片（分隔带 %d，拼接边界 %d，强制切 %d）"
                  % (prefix, r["W"], r["H"], len(cuts) + 1,
                     n_kind["natural"], n_kind["edge"], n_kind["forced"]))
        log_rows.append([
            name, "%dx%d" % (r["W"], r["H"]), "%.2f" % (r["H"] / r["W"]), r["status"],
            len(cuts) + 1 if cuts else "", n_kind["forced"],
            " ".join("%d%s" % (c[0], CUT_MARKS[c[1]]) for c in cuts),
            src,
        ])

    if not args.dry_run and log_rows:
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["文件名", "尺寸", "高宽比", "状态", "切片数", "强制切数",
                            "切口y坐标(~=拼接边界 *=强制切)", "原图路径"])
            w.writerows(log_rows)

    print("\n完成，用时 %.1fs：%s" % (time.time() - t0,
                                  "，".join("%s %d" % kv for kv in stats.items())))
    if not args.dry_run:
        print("切分记录：%s" % log_path)


if __name__ == "__main__":
    main()
