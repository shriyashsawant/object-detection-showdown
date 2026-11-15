"""Run every available detector / segmenter on the same images and put input + every output side by side.

    python compare_models.py            # uses weights/ and images/ next to this file

Models whose weight file is missing are skipped (nothing is downloaded here), so the grid grows as you add weights.
There is no ground truth for these images, so besides speed / size we report a CONSENSUS check:
an object counts as 'consensus' when at least half of the models found it (same class, box IoU >= 0.5).
'consensus recall' = share of those objects a model found; 'only this model' = detections nobody else made
(possible false positives, or possibly objects only this model is sharp enough to see). It is a proxy, not accuracy.
"""
import json, os, statistics, time, zlib

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.ops import box_iou

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("WEIGHTS_DIR", os.path.join(HERE, "weights"))
IMAGES = os.environ.get("IMAGES_DIR", os.path.join(HERE, "images"))
OUT = os.path.join(HERE, "results")
CONF, WARM, RUNS, PANEL_H, COLS = 0.5, 2, 5, 420, 4
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()


def timed(fn):
    for _ in range(WARM):
        fn(); sync()
    ts = []
    for _ in range(RUNS):
        t = time.perf_counter(); out = fn(); sync(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), min(ts), max(ts), out


def find_weight(name):
    for folder in (WEIGHTS, os.path.expanduser("~/Desktop")):
        p = os.path.join(folder, name + ".pt")
        if os.path.exists(p):
            return p
    return None


# ---- model registry ------------------------------------------------------------------------------------------------
def torchvision_models():
    from torchvision.models import detection as D
    spec = [("Faster R-CNN MobileNetV3", D.fasterrcnn_mobilenet_v3_large_fpn, D.FasterRCNN_MobileNet_V3_Large_FPN_Weights),
            ("Faster R-CNN R50-FPN v2", D.fasterrcnn_resnet50_fpn_v2, D.FasterRCNN_ResNet50_FPN_V2_Weights),
            ("Mask R-CNN R50-FPN v2", D.maskrcnn_resnet50_fpn_v2, D.MaskRCNN_ResNet50_FPN_V2_Weights)]
    for name, builder, weights in spec:
        ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
        w = weights.DEFAULT
        if not os.path.exists(ckpt + os.path.basename(w.url)):      # never download here
            print("skip (weights not cached):", name); continue
        model = builder(weights=w).to(dev).eval()
        names = w.meta["categories"]
        params = sum(p.numel() for p in model.parameters()) / 1e6

        def run(img, model=model, names=names):
            t = torchvision.transforms.functional.to_tensor(img).to(dev)

            def f():
                with torch.no_grad():
                    return model([t])[0]
            med, lo, hi, out = timed(f)
            keep = out["scores"] >= CONF
            masks = (out["masks"][keep, 0] > 0.5).cpu().numpy() if "masks" in out else None
            return dict(ms=(med, lo, hi), boxes=out["boxes"][keep].cpu(), scores=out["scores"][keep].cpu().tolist(),
                        labels=[names[int(i)] for i in out["labels"][keep].cpu()], masks=masks)
        yield name, params, os.path.getsize(ckpt + os.path.basename(w.url)) / 1e6, "mask" if "mask" in name.lower() else "box", run


YOLO_FILES = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo12n", "yolo12m", "yolo26n", "yolo26m", "yolo26l", "rtdetr-l",
              "yolo11n-seg", "yolo11m-seg", "yolo11l-seg", "yolo26n-seg", "yolo26m-seg", "yolo26l-seg"]


def ultralytics_models():
    from ultralytics import RTDETR, YOLO
    for stem in YOLO_FILES:
        path = find_weight(stem)
        if path is None:
            print("skip (no weights file):", stem); continue
        seg = stem.endswith("-seg")
        model = (RTDETR if stem.startswith("rtdetr") else YOLO)(path)
        params = sum(p.numel() for p in model.model.parameters()) / 1e6

        def run(img, model=model, seg=seg):
            frame = np.array(img)[:, :, ::-1].copy()
            med, lo, hi, out = timed(lambda: model.predict(frame, device=dev.type, conf=CONF, verbose=False, retina_masks=seg))
            r = out[0]
            masks = (r.masks.data.cpu().numpy() > 0.5) if seg and r.masks is not None else None
            return dict(ms=(med, lo, hi), boxes=r.boxes.xyxy.cpu(), scores=r.boxes.conf.cpu().tolist(),
                        labels=[r.names[int(c)] for c in r.boxes.cls.cpu()], masks=masks)
        yield stem, params, os.path.getsize(path) / 1e6, "mask" if seg else "box", run


# ---- analysis ------------------------------------------------------------------------------------------------------
def consensus(res):
    """Group same-class boxes (IoU>=0.5) from different models; a group backed by >= half of the models is 'consensus'."""
    items = [(m, i) for m, r in res.items() for i in range(len(r["labels"]))]
    if not items:
        return [], {}
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    boxes = torch.stack([res[m]["boxes"][i] for m, i in items])
    iou = box_iou(boxes, boxes).numpy()
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if items[a][0] != items[b][0] and iou[a, b] >= 0.5 and res[items[a][0]]["labels"][items[a][1]] == res[items[b][0]]["labels"][items[b][1]]:
                parent[find(a)] = find(b)
    groups = {}
    for k, item in enumerate(items):
        groups.setdefault(find(k), []).append(item)
    need = int(np.ceil(len(res) / 2))
    cons = [g for g in groups.values() if len({m for m, _ in g}) >= need]
    only = {m: sum(1 for g in groups.values() if len(g) == 1 and g[0][0] == m) for m in res}
    return cons, only


def size_buckets(boxes):
    a = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).numpy() if len(boxes) else np.array([])
    return dict(small=int((a < 32 ** 2).sum()), medium=int(((a >= 32 ** 2) & (a < 96 ** 2)).sum()), large=int((a >= 96 ** 2).sum()))


def color(label):
    h = zlib.crc32(label.encode())
    return (int(60 + h % 190), int(60 + (h >> 8) % 190), int(60 + (h >> 16) % 190))


def panel(img_bgr, title, d=None):
    h, w = img_bgr.shape[:2]; s = PANEL_H / h
    canvas = cv2.resize(img_bgr, (int(w * s), PANEL_H))
    if d is not None:
        if d["masks"] is not None:
            for k, m in enumerate(d["masks"]):
                m = cv2.resize(m.astype(np.uint8), (canvas.shape[1], PANEL_H), interpolation=cv2.INTER_NEAREST).astype(bool)
                canvas[m] = (0.45 * canvas[m] + 0.55 * np.array(color(d["labels"][k] + str(k)))).astype(np.uint8)
        for k in range(len(d["labels"])):
            x1, y1, x2, y2 = [int(v * s) for v in d["boxes"][k]]
            c = color(d["labels"][k])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
            cv2.putText(canvas, f"{d['labels'][k]} {d['scores'][k]:.2f}", (x1 + 2, max(y1 + 12, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(canvas, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def grid(panels):
    w = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, w - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    while len(panels) % COLS:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + COLS]) for i in range(0, len(panels), COLS)]
    return np.vstack(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    registry = list(torchvision_models()) + list(ultralytics_models())
    print(f"{len(registry)} models on {dev}")
    summary = {}
    for fname in sorted(os.listdir(IMAGES)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        img = Image.open(os.path.join(IMAGES, fname)).convert("RGB")
        bgr = np.array(img)[:, :, ::-1].copy()
        res, meta = {}, {}
        for name, params, mb, kind, run in registry:
            res[name] = run(img); meta[name] = (params, mb, kind)
        cons, only = consensus(res)
        found = {m: sum(1 for g in cons if any(x[0] == m for x in g)) for m in res}
        rows = []
        for name, r in res.items():
            params, mb, kind = meta[name]
            n = len(r["labels"])
            cover = None
            if r["masks"] is not None and n:
                union = np.zeros(r["masks"][0].shape, bool)
                for m in r["masks"]:
                    union |= m
                cover = round(float(union.mean()), 3)
            rows.append(dict(model=name, kind=kind, median_ms=round(r["ms"][0], 1), min_ms=round(r["ms"][1], 1), max_ms=round(r["ms"][2], 1),
                             params_M=round(params, 1), weights_MB=round(mb, 1), detections=n,
                             masks=None if r["masks"] is None else len(r["masks"]), mask_coverage=cover,
                             mean_conf=round(float(np.mean(r["scores"])), 3) if n else 0.0, sizes=size_buckets(r["boxes"]),
                             consensus_recall=round(found[name] / len(cons), 3) if cons else None, only_this_model=only[name],
                             classes=dict(sorted({l: r["labels"].count(l) for l in set(r["labels"])}.items()))))
        summary[stem] = dict(image=fname, size=img.size, consensus_objects=len(cons), models=rows)
        panels = [panel(bgr, f"INPUT {fname} {img.size[0]}x{img.size[1]}")]
        for row in rows:
            panels.append(panel(bgr, f"{row['model']} | {row['median_ms']} ms | {row['detections']} found", res[row["model"]]))
        cv2.imwrite(os.path.join(OUT, f"sidebyside_{stem}.jpg"), grid(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\n=== {fname}: {len(cons)} consensus objects (found by >= half of the {len(res)} models)")
        print(f"{'model':<26}{'ms':>7}{'MB':>7}{'found':>6}{'masks':>6}{'consRecall':>11}{'onlyMe':>7}{'conf':>6}  small/med/large")
        for r in rows:
            s = r["sizes"]
            print(f"{r['model']:<26}{r['median_ms']:>7}{r['weights_MB']:>7}{r['detections']:>6}{str(r['masks']):>6}{str(r['consensus_recall']):>11}{r['only_this_model']:>7}{r['mean_conf']:>6}  {s['small']}/{s['medium']}/{s['large']}")
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
