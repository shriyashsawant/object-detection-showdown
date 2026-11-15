"""Build results/tables.md from results/summary.json (speed, crowded scenes) and results/coco128_eval.json (accuracy vs ground truth)."""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
summ = json.load(open(R + "/summary.json"))
ev = json.load(open(R + "/coco128_eval.json"))
speed = json.load(open(R + "/speed.json")) if os.path.exists(R + "/speed.json") else {}      # one-model-at-a-time timings (preferred)
GT = {"000000000196.jpg": 42, "000000000164.jpg": 40, "000000000315.jpg": 38}          # ground-truth objects in the 3 crowded images
crowded = [k for k, v in summ.items() if v["image"] in GT]
lines = []

# speed (median over the 5 images of each model's median ms)
rows = {}
for stem, s in summ.items():
    for m in s["models"]:
        rows.setdefault(m["model"], dict(ms=[], kind=m["kind"], params=m["params_M"], mb=m["weights_MB"]))["ms"].append(m["median_ms"])
lines += ["### Accuracy vs ground truth (coco128, 128 images, 929 objects; contaminated, see caveat) + speed (median ms over 5 images, M5 / MPS, one model loaded at a time)", "",
          "| Model | Task | Params (M) | Weights (MB) | Median ms | mAP50 | mAP50-95 | Precision@0.5 | Recall@0.5 | Recall small objects | Recall on crowded images |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
order = sorted(ev["models"], key=lambda n: -ev["models"][n]["mAP50_95"])
for n in order:
    e, r = ev["models"][n], rows[n]
    ms = speed[n]["median_ms"] if n in speed else statistics.median(r["ms"])
    lines.append(f"| {n} | {e['kind']} | {e['params_M']} | {e['weights_MB']} | {ms:.0f} | {e['mAP50']:.3f} | {e['mAP50_95']:.3f} | "
                 f"{e['precision_at_0_5']:.3f} | {e['recall_at_0_5']:.3f} | {e['recall_small']:.2f} | {e['crowded_recall_at_0_5']:.2f} |")

# crowded scenes: objects found / masks produced vs ground truth
lines += ["", "### The three most crowded images (ground truth: 42, 40, 38 objects): detections / masks at confidence 0.5", "",
          "| Model | " + " | ".join(f"{summ[s]['image'][-8:-4]} (GT {GT[summ[s]['image']]})" for s in crowded) + " | Masks produced (same images) |", "| --- | ---: | ---: | ---: | --- |"]
for n in order:
    cells, masks = [], []
    for s in crowded:
        m = next(x for x in summ[s]["models"] if x["model"] == n)
        cells.append(str(m["detections"])); masks.append("-" if m["masks"] is None else str(m["masks"]))
    lines.append(f"| {n} | " + " | ".join(cells) + " | " + ("/".join(masks) if masks[0] != "-" else "no masks") + " |")
open(R + "/tables.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
