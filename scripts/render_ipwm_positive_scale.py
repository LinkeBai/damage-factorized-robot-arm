"""Create paper-ready tables and figures for the frozen positive replication."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/ipwm_positive_scale_20260911"
PAPER = ROOT / "paper"
PRIMARY = {"high_damping", "mixed_composition", "mixed_unseen"}


def md_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def main() -> None:
    summary = json.loads((OUT / "results-summary.json").read_text(encoding="utf-8"))
    with (OUT / "all-cells.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    h50 = [r for r in rows if int(r["horizon"]) == 50]
    labels = sorted({r["physics"] for r in h50})
    for ax, metric, ylabel in [
        (axes[0], "object_state_improvement_pct", "Object-state RMSE reduction (%)"),
        (axes[1], "selective_xy_rmse_m", "Selective object xy RMSE (mm)"),
    ]:
        for x, condition in enumerate(labels):
            values = [float(r[metric]) * (1000 if metric.endswith("_m") else 1)
                      for r in h50 if r["physics"] == condition]
            ax.scatter(np.full(len(values), x) + np.linspace(-.08, .08, len(values)), values, s=34)
            ax.plot([x - .18, x + .18], [np.mean(values)] * 2, color="black", lw=1.8)
        ax.set_xticks(range(len(labels)), [x.replace("_", "\n") for x in labels], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="#8993a4", lw=1) if metric.endswith("pct") else None
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("H=50, all fixed seeds and conditions")
    axes[1].set_title("Absolute position error (separate units)")
    fig.savefig(OUT / "positive-replication.png", dpi=200)
    fig.savefig(OUT / "positive-replication.svg")
    plt.close(fig)

    primary = [r for r in rows if r["physics"] in PRIMARY]
    lines = [
        "# IPWM 历史正收益大规模独立复现", "",
        "本实验固定历史检查点、D3 拓扑和固定侧接触邻域策略，在新生成且互不重复的轨迹上复现选择性 IPWM 相对机制匹配载体的预测收益。窗口、预测方法和模型种子不重复计入数据量。", "",
        "## 可比较的数据规模", "",
        md_table(["项目", "ActivePusher 原文", "本次 IPWM 复现"], [
            ["独立仿真测试轨迹/问题", "4,000（4×1,000）", "8,400（7×1,200）"],
            ["主要条件", "4 个任务设置", "3 个预注册物理条件"],
            ["完整条件覆盖", "4 个任务设置", "7 个物理条件"],
            ["训练重复", "5", "固定 3 个已有检查点；没有重训"],
        ]), "",
        "以上仅比较可核验的独立样本计数。任务、物体多样性和研究问题不同，样本更多不代表性能更强。", "",
        "## 预注册主要分析", "",
        f"主要条件共 {summary['primary_cells']} 个种子–条件–预测长度格子，其中 {summary['positive_primary_cells']} 个组合状态误差改善；平均改善 {summary['mean_primary_improvement_pct']:.2f}%。H=50 为 {summary['positive_primary_h50_cells']}/9 个改善，平均改善 {summary['mean_primary_h50_improvement_pct']:.2f}%。", "",
        md_table(["种子", "条件", "H", "载体组合RMSE", "IPWM组合RMSE", "改善", "位置RMSE (mm)"] , [
            [r["seed"], r["physics"], r["horizon"], f"{float(r['carrier_object_state_rmse']):.5f}",
             f"{float(r['selective_object_state_rmse']):.5f}", f"{float(r['object_state_improvement_pct']):.2f}%",
             f"{float(r['selective_xy_rmse_m']) * 1000:.3f}"] for r in primary
        ]), "",
        "## 机制与边界", "",
        f"选择性包装器相对载体的机器人预测最大改动为 {summary['max_robot_change_from_carrier']:.3e}，锁定约束最大违反为 {summary['max_lock_violation']:.3e}。组合对象状态 RMSE 只用于复现历史口径；论文正文必须将位置（m）和速度（m/s）分开报告。", "",
        "该结果支持的主张限于固定侧接触邻域、D3 锁定和既有检查点下的预测复现性。它不构成新任务泛化、物体多样性或最终真机权重收益证据。三个种子是固定模型检查点，不应表述为三次新训练。", "",
        "所有 63 个格子见 `runs/ipwm_positive_scale_20260911/all-cells.csv`，包括非主要条件和任何负收益。", "",
    ]
    md = "\n".join(lines)
    md_path = PAPER / "ipwm-positive-scale-evidence-20260911.md"
    md_path.write_text(md, encoding="utf-8")

    body = ["<img src='../runs/ipwm_positive_scale_20260911/positive-replication.png'>"]
    in_table = False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(c == "---" for c in cells):
                continue
            if not in_table:
                body.append("<table>"); in_table = True
            body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                body.append("</table>"); in_table = False
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                body.append(f"<h{level}>{html.escape(line[level:].strip())}</h{level}>")
            elif line:
                body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table>")
    (PAPER / "ipwm-positive-scale-evidence-20260911.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>IPWM 正收益复现</title>"
        "<style>body{max-width:1180px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;color:#172337}"
        "table{border-collapse:collapse;width:100%;font-size:14px}td{border-bottom:1px solid #d7dee7;padding:7px}"
        "tr:first-child{font-weight:bold;background:#edf2f8}img{max-width:100%}h2{margin-top:38px}</style>" + "".join(body),
        encoding="utf-8")

    tex = [
        r"\subsection{Large-scale replication of selective prediction}",
        ("We freeze the three historical checkpoints and evaluate 8,400 newly generated, independently reset "
         "trajectories across seven physical conditions. Evaluation windows, methods, and checkpoints do not "
         "multiply the independent sample count. We retain every seed, condition, and horizon."),
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pre-registered primary-stratum replication. The historical mixed-unit metric is retained only for comparability; position and velocity are reported separately in the full table.}",
        r"\begin{tabular}{lrr}", r"\hline",
        "Horizon & Positive cells & Mean reduction (\\%) \\\\", r"\hline",
    ]
    for horizon in [10, 25, 50]:
        vals = [float(r["object_state_improvement_pct"]) for r in primary if int(r["horizon"]) == horizon]
        tex.append(f"{horizon} & {sum(v > 0 for v in vals)}/9 & {np.mean(vals):.2f} \\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (PAPER / "ipwm-positive-scale-experiments-20260911.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
