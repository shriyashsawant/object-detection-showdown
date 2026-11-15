"""Supervised version: does a GRU that keeps state recognise what the cat is doing better than models without memory?

    python video_state_supervised.py          # needs results/arrays.npz (from video_state.py) and labels.csv

Labels (labels.csv) were written by looking at the clip every 0.5 s: facing_camera / head_angled_down / head_turned_away.
They are one person's judgement on ONE 20-second clip, so treat every number as a demonstration, not a benchmark.

Evaluation: 5 contiguous time blocks. Each block is held out once; the other blocks train; 0.5 s of training data next to the held-out
block is dropped ('purge') so neighbouring near-identical frames do not leak. Predictions from all 5 blocks are pooled.
Compared:  majority class | per-frame linear classifier (no memory) | per-frame linear + causal smoothing of its scores over the last 8 steps
           | GRU (32 units) that carries state through the held-out block.
"""
import csv, json, os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS, PURGE, WIN, EPOCHS, SEEDS = 5, 8, 24, 80, (0, 1, 2)


def load():
    z = np.load(os.path.join(HERE, "results", "arrays.npz"))
    X = z["X"]
    step = json.load(open(os.path.join(HERE, "results", "analysis.json")))["step_s"]
    rows = list(csv.DictReader(open(os.path.join(HERE, "labels.csv"))))
    names = sorted({r["label"] for r in rows})
    y = np.zeros(len(X), int)
    for k in range(len(X)):
        t = k * step
        for r in rows:
            if float(r["start_s"]) <= t <= float(r["end_s"]) + 1e-6:
                y[k] = names.index(r["label"])
    return X, y, names, step


class GRUClassifier(nn.Module):
    def __init__(self, d, c, hidden=32):
        super().__init__()
        self.gru, self.out = nn.GRU(d, hidden, batch_first=True), nn.Linear(hidden, c)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h)


def fit_gru(X, y, train_idx_blocks, seed, c):
    torch.manual_seed(seed)
    m = GRUClassifier(X.shape[1], c)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    wins = [(a, min(a + WIN, b)) for a, b in train_idx_blocks for a in range(a, b - 8, 4)]
    for _ in range(EPOCHS):
        for a, b in wins:
            loss = nn.functional.cross_entropy(m(Xt[None, a:b])[0], yt[a:b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def fit_linear(X, y, idx, seed, c):
    torch.manual_seed(seed)
    m = nn.Linear(X.shape[1], c)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
    Xt, yt = torch.tensor(X[idx]), torch.tensor(y[idx])
    for _ in range(300):
        loss = nn.functional.cross_entropy(m(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def macro_f1(y, p, c):
    f = []
    for k in range(c):
        tp = ((p == k) & (y == k)).sum(); fp = ((p == k) & (y != k)).sum(); fn = ((p != k) & (y == k)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    X, y, names, step = load()
    T, c = len(X), len(names)
    edges = np.linspace(0, T, FOLDS + 1).astype(int)
    print(f"{T} steps, classes {names}, share {[round(float((y == k).mean()), 2) for k in range(c)]}")
    results = {}
    for seed in SEEDS:
        pred = {k: np.zeros(T, int) for k in ("majority", "linear", "linear_smoothed", "gru")}
        for f in range(FOLDS):
            a, b = edges[f], edges[f + 1]
            train = np.array([i for i in range(T) if (i < a - PURGE or i >= b + PURGE)])
            blocks, s = [], None
            for i in train:                                                  # contiguous training blocks for GRU windows
                if s is None:
                    s = prev = i
                elif i != prev + 1:
                    blocks.append((s, prev + 1)); s = i
                prev = i
            blocks.append((s, prev + 1))
            pred["majority"][a:b] = np.bincount(y[train], minlength=c).argmax()
            lin = fit_linear(X, y, train, seed, c)
            with torch.no_grad():
                logits = lin(torch.tensor(X[a:b])).numpy()
                pred["linear"][a:b] = logits.argmax(1)
                sm = np.array([logits[max(0, i - 7):i + 1].mean(0) for i in range(len(logits))])      # causal: only past and present
                pred["linear_smoothed"][a:b] = sm.argmax(1)
            gru = fit_gru(X, y, blocks, seed, c)
            with torch.no_grad():
                pred["gru"][a:b] = gru(torch.tensor(X[None, a:b]))[0].argmax(1).numpy()                # state starts at zero in the block
        for k, p in pred.items():
            changes = sum(int((p[edges[f]:edges[f + 1]][1:] != p[edges[f]:edges[f + 1]][:-1]).sum()) for f in range(FOLDS))
            results.setdefault(k, []).append(dict(acc=float((p == y).mean()), f1=macro_f1(y, p, c), changes=changes,
                                                 conf=[[int(((y == i) & (p == j)).sum()) for j in range(c)] for i in range(c)] if seed == SEEDS[0] else None))
    summary = {}
    print(f"\n{'model':<18}{'accuracy':>10}{'macro-F1':>10}{'label flips':>13}   (mean over {len(SEEDS)} seeds; range in brackets)")
    true_changes = int((y[1:] != y[:-1]).sum())
    for k, v in results.items():
        acc, f1, ch = [r["acc"] for r in v], [r["f1"] for r in v], [r["changes"] for r in v]
        summary[k] = dict(accuracy=round(float(np.mean(acc)), 3), accuracy_range=[round(min(acc), 3), round(max(acc), 3)], macro_f1=round(float(np.mean(f1)), 3),
                          flips=round(float(np.mean(ch)), 1), confusion_seed0=v[0]["conf"])
        print(f"{k:<18}{np.mean(acc):>10.3f}{np.mean(f1):>10.3f}{np.mean(ch):>13.1f}   [{min(acc):.3f}-{max(acc):.3f}]")
    print(f"true label changes in the clip: {true_changes}; classes (rows=true, cols=predicted) {names}")
    for k in ("linear", "gru"):
        print(k, summary[k]["confusion_seed0"])
    json.dump(dict(classes=names, steps=T, step_s=step, true_label_changes=true_changes, folds=FOLDS, purge_steps=PURGE, seeds=list(SEEDS), results=summary),
              open(os.path.join(HERE, "results", "supervised.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
