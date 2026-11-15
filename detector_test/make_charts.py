"""Two post-ready charts from the saved results (no re-running of models).

    python make_charts.py   ->  results/chart_speed_vs_accuracy.png, results/chart_recall.png
Colours: Okabe-Ito palette (colour-blind safe); family is also written in the labels, so colour is never the only cue.
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
ev = json.load(open(R + "/coco128_eval.json"))["models"]
speed = json.load(open(R + "/speed.json"))
FAMILY = {"R-CNN": "#D55E00", "YOLO": "#0072B2", "RT-DETR (transformer)": "#009E73"}
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7ec"


def family(name):
    return "R-CNN" if "R-CNN" in name else "RT-DETR (transformer)" if name.startswith("rtdetr") else "YOLO"


def pretty(name):
    return name.replace("yolo", "YOLO").replace("rtdetr", "RT-DETR")


plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

# ---- chart 1: accuracy vs speed ------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
for name, e in ev.items():
    f = family(name)
    ax.scatter(speed[name]["median_ms"], e["mAP50_95"], s=70, color=FAMILY[f], marker="s" if e["kind"] == "mask" else "o",
               edgecolor="white", linewidth=0.8, zorder=3)
label = {"Mask R-CNN R50-FPN v2": (-8, 9), "Faster R-CNN R50-FPN v2": (-8, -16), "Faster R-CNN MobileNetV3": (-8, -4), "rtdetr-l": (8, 6),
         "yolo11x": (8, 4), "yolo11m": (6, -15), "yolo26n": (-4, -17), "yolo11n-seg": (8, -12), "yolo11m-seg": (6, 6)}
for name, (dx, dy) in label.items():
    ax.annotate(pretty(name), (speed[name]["median_ms"], ev[name]["mAP50_95"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color=INK, ha="right" if dx < 0 else "left")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "500", "1000"])
ax.set_xlabel("Time per image on a MacBook (ms, log scale)  -  further left is faster", fontsize=10)
ax.set_ylabel("Accuracy (mAP50-95)  -  higher is better", fontsize=10)
ax.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in FAMILY.items()] + \
          [plt.Line2D([], [], marker="o", ls="", color=MUTED, label="boxes only"), plt.Line2D([], [], marker="s", ls="", color=MUTED, label="boxes + masks")]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
fig.suptitle("20 object detectors: accuracy vs speed", x=0.06, ha="left", fontsize=14, color=INK, fontweight="bold")
fig.text(0.06, 0.915, "At similar accuracy, Mask R-CNN is ~20x slower than YOLO11m; the one transformer tested (RT-DETR-L) scored highest", fontsize=10, color=MUTED)
fig.text(0.06, 0.012, "Apple M5 laptop, 128 labelled COCO images (the models trained on them, so compare the order, not the absolute scores).",
         fontsize=7.5, color=MUTED)
fig.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.97)
fig.savefig(R + "/chart_speed_vs_accuracy.png"); plt.close(fig)

# ---- chart 2: who finds the objects --------------------------------------------------------------------------------
pick = ["Mask R-CNN R50-FPN v2", "Faster R-CNN R50-FPN v2", "rtdetr-l", "yolo11x", "yolo11m", "yolo11n"]
metrics = [("recall_at_0_5", "All objects"), ("recall_small", "Small objects"), ("crowded_recall_at_0_5", "Crowded scenes")]
fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), dpi=160, sharey=True)
for ax, (key, title) in zip(axes, metrics):
    vals = [ev[n][key] for n in pick]
    ypos = list(range(len(pick)))[::-1]
    ax.barh(ypos, vals, color=[FAMILY[family(n)] for n in pick], height=0.62, zorder=3)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.015, y, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1); ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0%", "25%", "50%", "75%"])
    ax.grid(True, axis="x", color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(list(range(len(pick)))[::-1]); axes[0].set_yticklabels([pretty(n) for n in pick], fontsize=9.5, color=INK)
fig.suptitle("Share of real objects each model found (confidence 0.5)", x=0.01, ha="left", fontsize=14, color=INK, fontweight="bold")
fig.text(0.01, 0.015, "R-CNN (orange) finds the most, especially small objects and crowded scenes. Same 128 labelled COCO images as the other chart.",
         fontsize=8, color=MUTED)
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.2, right=0.98, wspace=0.12)
fig.savefig(R + "/chart_recall.png"); plt.close(fig)
print("saved charts")
