"""Score every available model against ground-truth boxes on coco128 (128 images, 929 objects).

    python eval_coco128.py

CAVEAT: coco128 images are a slice of COCO *train2017*, which all these models were trained on, so scores are inflated and must not be
quoted as accuracy. They are useful for the ORDER of models and for crowded-scene behaviour, not for absolute numbers.

Per model:
  mAP50 and mAP50-95  threshold-free accuracy (COCO-style 101-point AP, class-averaged) from all detections with confidence >= 0.05
  precision / recall  at confidence 0.5, class-aware match at IoU >= 0.5   (NOT comparable across families: score scales differ)
  recall by object size (COCO: small < 32^2 px, medium < 96^2, large) and on the CROWDED subset (images with >= 10 objects) at confidence 0.5
Speed is NOT measured here (every image is a new input size, which distorts timing); see compare_models.py for repeated-run timings.
"""
import json, os, sys

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_models as cm  # noqa: E402

DATA = os.path.join(HERE, "data", "coco128")
CROWDED = 10
THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def load_gt(names):
    gt, img_dir, lab_dir = {}, DATA + "/images/train2017", DATA + "/labels/train2017"
    for f in sorted(os.listdir(img_dir)):
        w, h = Image.open(os.path.join(img_dir, f)).size
        boxes, labels = [], []
        p = os.path.join(lab_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(p):
            for line in open(p):
                c, x, y, bw, bh = line.split()[:5]
                x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
                boxes.append([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]); labels.append(names[int(c)])
        gt[f] = dict(boxes=torch.tensor(boxes).reshape(-1, 4), labels=labels)
    return gt


def bucket(box):
    a = float((box[2] - box[0]) * (box[3] - box[1]))
    return "small" if a < 32 ** 2 else "medium" if a < 96 ** 2 else "large"


def average_precision(tp_flags, n_gt):
    if n_gt == 0:
        return None
    if not tp_flags:
        return 0.0
    tp = np.cumsum(tp_flags); fp = np.cumsum(1 - np.array(tp_flags))
    rec, prec = tp / n_gt, tp / (tp + fp)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])                              # monotone precision envelope
    return float(np.mean([prec[np.searchsorted(rec, r, side="left")] if r <= rec[-1] else 0.0 for r in np.linspace(0, 1, 101)]))


def mean_ap(preds, gt, ious):
    """preds: {file: dict(boxes, labels, scores)}; ious: {file: IoU matrix pred x gt}. Returns (mAP50, mAP50-95)."""
    classes = sorted({l for g in gt.values() for l in g["labels"]})
    per_t = []
    for t in THRESHOLDS:
        aps = []
        for c in classes:
            n_gt = sum(g["labels"].count(c) for g in gt.values())
            dets = [(preds[f]["scores"][i], f, i) for f in preds for i, l in enumerate(preds[f]["labels"]) if l == c]
            dets.sort(key=lambda d: -d[0])
            used = {f: np.zeros(len(gt[f]["labels"]), bool) for f in gt}
            flags = []
            for _, f, i in dets:
                best, bj = t, -1
                for j, l in enumerate(gt[f]["labels"]):
                    if l == c and not used[f][j] and ious[f][i, j] >= best:
                        best, bj = float(ious[f][i, j]), j
                if bj >= 0:
                    used[f][bj] = True; flags.append(1)
                else:
                    flags.append(0)
            ap = average_precision(flags, n_gt)
            if ap is not None:
                aps.append(ap)
        per_t.append(float(np.mean(aps)))
    return per_t[0], float(np.mean(per_t))


def main():
    from ultralytics import YOLO
    names = YOLO(cm.find_weight("yolo11n")).names
    gt = load_gt(names)
    counts = {f: len(g["labels"]) for f, g in gt.items()}
    crowded = [f for f, n in counts.items() if n >= CROWDED]
    print(f"{len(gt)} images, {sum(counts.values())} objects; crowded (>= {CROWDED} objects): {len(crowded)} images, {sum(counts[f] for f in crowded)} objects")
    cm.CONF, cm.WARM, cm.RUNS = 0.05, 0, 1
    registry = list(cm.torchvision_models()) + list(cm.ultralytics_models())
    out = {}
    first = os.path.join(DATA, "images/train2017", sorted(gt)[0])
    for name, params, mb, kind, run in registry:
        run(Image.open(first).convert("RGB"))                                           # warm-up, discarded
        preds, ious = {}, {}
        for f, g in gt.items():
            r = run(Image.open(os.path.join(DATA, "images/train2017", f)).convert("RGB"))
            preds[f] = r
            ious[f] = box_iou(r["boxes"], g["boxes"]).numpy() if len(r["labels"]) and len(g["labels"]) else np.zeros((len(r["labels"]), len(g["labels"])))
        m50, m5095 = mean_ap(preds, gt, ious)
        tp = fp = n_gt = ctp = cn = cdet = cmask = 0
        found = {"small": 0, "medium": 0, "large": 0}; total = dict(found)
        for f, g in gt.items():
            r = preds[f]
            keep = [i for i, s in enumerate(r["scores"]) if s >= 0.5]
            used = np.zeros(len(g["labels"]), bool); t = 0
            for i in sorted(keep, key=lambda i: -r["scores"][i]):
                cand = [(ious[f][i, j], j) for j in range(len(used)) if not used[j] and g["labels"][j] == r["labels"][i] and ious[f][i, j] >= 0.5]
                if cand:
                    used[max(cand)[1]] = True; t += 1
            tp += t; fp += len(keep) - t; n_gt += len(g["labels"])
            for j, u in enumerate(used):
                b = bucket(g["boxes"][j]); total[b] += 1; found[b] += int(u)
            if f in crowded:
                ctp += t; cn += len(g["labels"]); cdet += len(keep)
                cmask += 0 if r["masks"] is None else len(keep)
        prec, rec = tp / max(tp + fp, 1), tp / max(n_gt, 1)
        out[name] = dict(kind=kind, params_M=round(params, 1), weights_MB=round(mb, 1), mAP50=round(m50, 3), mAP50_95=round(m5095, 3),
                         precision_at_0_5=round(prec, 3), recall_at_0_5=round(rec, 3),
                         recall_small=round(found["small"] / max(total["small"], 1), 3), recall_medium=round(found["medium"] / max(total["medium"], 1), 3),
                         recall_large=round(found["large"] / max(total["large"], 1), 3), crowded_recall_at_0_5=round(ctp / max(cn, 1), 3),
                         crowded_detections_at_0_5=cdet, crowded_masks_at_0_5=cmask if kind == "mask" else None, crowded_gt_objects=cn)
        o = out[name]
        print(f"{name:<26}mAP50 {o['mAP50']:.3f}  mAP50-95 {o['mAP50_95']:.3f}  P@.5 {o['precision_at_0_5']:.3f}  R@.5 {o['recall_at_0_5']:.3f}  small {o['recall_small']:.2f}  crowdedR {o['crowded_recall_at_0_5']:.2f}", flush=True)
    json.dump(dict(images=len(gt), objects=sum(counts.values()), crowded_images=len(crowded), crowded_objects=sum(counts[f] for f in crowded),
                   most_crowded=sorted(counts, key=counts.get, reverse=True)[:6], models=out),
              open(os.path.join(HERE, "results", "coco128_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
