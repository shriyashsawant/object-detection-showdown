"""Video -> frames -> arrays -> CNN (YOLO) -> GRU that keeps state -> "what is happening" timeline.

    python video_state.py path/to/video.mp4 [--stride 2] [--weights yolo11n-seg.pt] [--max_frames N]

Pipeline (each step is a function below):
  1. extract()   read frames with OpenCV (every `stride`-th), run YOLO segmentation + ByteTrack tracking, take YOLO's CNN
                 feature vector for the frame, and measure motion (optical flow) -> one feature row per frame: array (T, D)
  2. train_gru() a GRU reads the rows in order and learns to predict the NEXT row (self-supervised: no labels needed).
                 Its hidden state is the "memory" of what has happened so far.
  3. label_states() cluster the hidden states into a few states; describe each state from measured numbers
  4. report()    timeline, annotated video, and a plain report with the honest checks (does the GRU beat 'repeat last frame'?
                 are its states steadier than clustering frames one at a time?)

It does NOT caption video in free text; descriptions come from detections and measurements, not from a language model.
"""
import argparse, json, os

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FEATURES = ["cx", "cy", "w", "h", "mask_area", "mask_cx", "mask_cy", "conf", "n_objects", "flow", "frame_diff"]


# ---- 1. frames -> arrays -> CNN features ---------------------------------------------------------------------------
def extract(video, weights, stride, max_frames=None, conf=0.4):
    from ultralytics import YOLO
    model = YOLO(weights)
    cap = cv2.VideoCapture(video)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    emb, hand, meta, prev = [], [], [], None
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and len(meta) >= max_frames):
            break
        i += 1
        if i % stride:
            continue
        r = model.track(frame, persist=True, conf=conf, verbose=False, tracker="bytetrack.yaml")[0]
        emb.append(model.embed(frame, verbose=False)[0].flatten().cpu().numpy())           # CNN feature vector for this frame
        small = cv2.cvtColor(cv2.resize(frame, (W // 4, H // 4)), cv2.COLOR_BGR2GRAY)
        flow = diff = 0.0
        if prev is not None:
            f = cv2.calcOpticalFlowFarneback(prev, small, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            flow, diff = float(np.linalg.norm(f, axis=2).mean()), float(np.abs(small.astype(float) - prev).mean())
        prev = small
        objs, row = [], [0] * 8
        for k in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[k].tolist()
            poly = r.masks.xy[k] if r.masks is not None and len(r.masks.xy[k]) > 2 else None
            area = float(cv2.contourArea(poly.astype(np.float32))) / (W * H) if poly is not None else 0.0
            objs.append(dict(label=r.names[int(r.boxes.cls[k])], conf=float(r.boxes.conf[k]), id=None if r.boxes.id is None else int(r.boxes.id[k]),
                             box=[x1, y1, x2, y2], poly=None if poly is None else poly.astype(np.int32), area=area, size=(x2 - x1) * (y2 - y1)))
        if objs:
            m = max(objs, key=lambda o: o["conf"] * o["size"])                              # the main object this frame
            x1, y1, x2, y2 = m["box"]
            mc = m["poly"].mean(0) if m["poly"] is not None else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            row = [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H, m["area"], mc[0] / W, mc[1] / H, m["conf"]]
        hand.append(row + [len(objs), flow, diff])
        meta.append(dict(frame=i, t=i / fps, objects=objs))
    cap.release()
    return np.array(emb), np.array(hand, dtype=np.float32), meta, dict(fps=fps, W=W, H=H, stride=stride)


def build_inputs(emb, hand, dims=16):
    """PCA the CNN vectors down to `dims`, add the hand-made measurements, standardise -> X (T, dims + 11)."""
    e = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(e, full_matrices=False)
    z = e @ vt[:dims].T
    X = np.hstack([z, hand])
    return ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)


# ---- 2. the RNN that keeps state -----------------------------------------------------------------------------------
class StateGRU(nn.Module):
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.gru, self.head = nn.GRU(d_in, hidden, batch_first=True), nn.Linear(hidden, d_in)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.head(out), out


def train_gru(X, epochs=60, win=24, seed=0, residual=True):
    """Self-supervised: from frames 0..t predict frame t+1. First 80% of time trains, last 20% is held out.
    residual=True: the GRU predicts the CHANGE from the current frame (so it starts as good as 'repeat the last frame'
    and only has to learn what moves). residual=False: it must predict the whole next frame from scratch."""
    torch.manual_seed(seed)
    T = len(X); cut = int(T * 0.8)
    model = StateGRU(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt = torch.tensor(X)
    starts = list(range(0, max(cut - win - 1, 1), 3))
    for ep in range(epochs):
        for s in starts:
            seq = Xt[None, s:s + win + 1]
            out, _ = model(seq[:, :-1])
            pred = seq[:, :-1] + out if residual else out
            loss = ((pred - seq[:, 1:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out, hidden = model(Xt[None, :-1])                                                   # whole video in one pass, state carried along
    pred, hidden = (X[:-1] + out[0].numpy() if residual else out[0].numpy()), hidden[0].numpy()
    target = X[1:]
    val = slice(cut, T - 1)
    gru_mse = float(((pred[val] - target[val]) ** 2).mean())
    persist_mse = float(((X[:-1][val] - target[val]) ** 2).mean())                           # baseline: 'next frame = this frame'
    mean_mse = float(((X[:cut].mean(0) - target[val]) ** 2).mean())                          # baseline: 'always the training average'
    hidden = np.vstack([hidden, hidden[-1:]])                                                # align length with X
    return hidden, dict(val_frames=int(T - 1 - cut), gru_mse=gru_mse, repeat_last_mse=persist_mse, train_mean_mse=mean_mse)


# ---- 3. states -----------------------------------------------------------------------------------------------------
def kmeans(Z, k, seed=0, iters=50):
    rng = np.random.RandomState(seed)
    C = [Z[rng.randint(len(Z))]]
    for _ in range(k - 1):
        d = np.min([((Z - c) ** 2).sum(1) for c in C], axis=0)
        C.append(Z[rng.choice(len(Z), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = ((Z[:, None] - C[None]) ** 2).sum(2).argmin(1)
        C = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def silhouette(Z, lab):
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(2))
    s = []
    for i in range(len(Z)):
        same = (lab == lab[i]); same[i] = False
        if not same.any():
            s.append(0); continue
        a = D[i, same].mean()
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def pca(A, dims=5):
    a = A - A.mean(0)
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    return a @ vt[:dims].T


def cluster(Z, ks=(3, 4, 5, 6)):
    best = max(((silhouette(Z, kmeans(Z, k)), k) for k in ks))
    return kmeans(Z, best[1]), best[1], best[0]


def runs(lab):
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append((s, i - 1, int(lab[s]))); s = i
    return out


def describe(hand, meta, lab, step_s):
    """Plain-language description of each state from MEASURED numbers (terciles of this clip)."""
    col = {n: hand[:, i] for i, n in enumerate(FEATURES)}
    lo = {n: np.percentile(col[n], 33) for n in FEATURES}; hi = {n: np.percentile(col[n], 66) for n in FEATURES}
    names = {}
    for s in sorted(set(lab)):
        m = lab == s
        mean = {n: float(col[n][m].mean()) for n in FEATURES}
        words = []
        words.append("fast movement" if mean["flow"] > hi["flow"] else "still" if mean["flow"] < lo["flow"] else "moderate movement")
        words.append("close/large in frame" if mean["mask_area"] > hi["mask_area"] else "small in frame" if mean["mask_area"] < lo["mask_area"] else "medium size")
        words.append("left" if mean["cx"] < lo["cx"] else "right" if mean["cx"] > hi["cx"] else "centre")
        words.append("high" if mean["cy"] < lo["cy"] else "low" if mean["cy"] > hi["cy"] else "mid-height")
        labels = [o["label"] for k, mt in enumerate(meta) if m[k] for o in mt["objects"]]
        main = max(set(labels), key=labels.count) if labels else "nothing detected"
        names[s] = dict(text=f"{main}, " + ", ".join(words), share=round(float(m.mean()), 3), mean_flow=round(mean["flow"], 3),
                        mean_mask_area=round(mean["mask_area"], 3), seconds=round(float(m.sum()) * step_s, 1))
    return names


# ---- 4. outputs ----------------------------------------------------------------------------------------------------
PALETTE = [(80, 200, 120), (60, 140, 255), (255, 120, 80), (220, 90, 220), (80, 220, 230), (200, 200, 80)]


def timeline_strip(lab, width, height=26):
    """One coloured column per pixel: the state at that moment of the video."""
    strip = np.zeros((height, width, 3), np.uint8)
    for x in range(width):
        k = min(int(x / width * len(lab)), len(lab) - 1)
        strip[:, x] = PALETTE[lab[k] % len(PALETTE)]
    return strip


def write_video(video, meta, lab, names, info, path):
    cap = cv2.VideoCapture(video)
    W, H, stride = info["W"], info["H"], info["stride"]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"] / stride, (W // 2, H // 2 + 74))
    wanted = {m["frame"]: k for k, m in enumerate(meta)}
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i not in wanted:
            continue
        k = wanted[i]
        overlay = frame.copy()
        for o in meta[k]["objects"]:
            if o["poly"] is not None:
                cv2.fillPoly(overlay, [o["poly"]], PALETTE[lab[k] % len(PALETTE)])
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        for o in meta[k]["objects"]:
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.putText(frame, f"{o['label']} {o['conf']:.2f} id{o['id']}", (x1 + 6, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        frame = cv2.resize(frame, (W // 2, H // 2))
        bar = np.zeros((74, W // 2, 3), np.uint8)
        bar[:] = (25, 25, 25)
        cv2.putText(bar, f"state {chr(65 + lab[k])}: {names[lab[k]]['text']}"[:60], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"t = {meta[k]['t']:.1f}s", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        strip = timeline_strip(lab, W // 2, 22)
        x = int(k / len(lab) * (W // 2))
        cv2.line(strip, (x, 0), (x, 22), (255, 255, 255), 2)
        bar[50:72] = strip
        out.write(np.vstack([frame, bar]))
    out.release(); cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--weights", default=os.path.join(HERE, "..", "detector_test", "weights", "yolo11n-seg.pt"))
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    emb, hand, meta, info = extract(a.video, a.weights, a.stride, a.max_frames)
    X = build_inputs(emb, hand)
    print(f"frames used: {len(X)} | CNN vector: {emb.shape[1]}-d -> input row: {X.shape[1]}-d | step {a.stride / info['fps']:.3f}s")
    _, direct = train_gru(X, epochs=a.epochs, residual=False)                                 # first design, kept for the honest comparison
    hidden, check = train_gru(X, epochs=a.epochs, residual=True)
    check["direct_gru_mse"] = direct["gru_mse"]
    step_s = a.stride / info["fps"]
    lab, k, sil = cluster(pca(hidden))
    base_lab = kmeans(pca(X), k)                                                             # memoryless: each frame on its own
    names = describe(hand, meta, lab, step_s)
    seg = runs(lab); base_seg = runs(base_lab)
    result = dict(video=os.path.basename(a.video), frames=len(X), step_s=round(step_s, 3), k=k, silhouette=round(sil, 3), check=check,
                  gru_state_changes=len(seg) - 1, memoryless_state_changes=len(base_seg) - 1,
                  states={chr(65 + s): v for s, v in names.items()},
                  segments=[dict(start_s=round(meta[s]["t"], 2), end_s=round(meta[e]["t"], 2), state=chr(65 + st)) for s, e, st in seg])
    json.dump(result, open(os.path.join(OUT, "analysis.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "arrays.npz"), X=X, hidden=hidden, labels=lab, features=hand)
    write_video(a.video, meta, lab, names, info, os.path.join(OUT, "annotated.mp4"))
    print(json.dumps({k_: result[k_] for k_ in ("k", "silhouette", "check", "gru_state_changes", "memoryless_state_changes")}, indent=1))
    for s, v in result["states"].items():
        print(f"  state {s}: {v['seconds']}s ({v['share']:.0%})  {v['text']}")


if __name__ == "__main__":
    main()
