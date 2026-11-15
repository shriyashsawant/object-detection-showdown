"""Clean speed numbers: load ONE model at a time (so memory pressure from the other 19 does not slow it), time it on 5 images
(2 warm-ups + median of 5 runs each), report the median over the 5 images. Output: results/speed.json"""
import gc, json, os, sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_models as cm  # noqa: E402

IMAGES = ["bus.jpg", "zidane.jpg", "000000000196.jpg", "000000000164.jpg", "000000000315.jpg"]
imgs = [Image.open(os.path.join(cm.IMAGES, f)).convert("RGB") for f in IMAGES]
out = {}
for gen in (cm.torchvision_models(), cm.ultralytics_models()):
    for name, params, mb, kind, run in gen:
        ms = [run(im)["ms"][0] for im in imgs]
        out[name] = dict(median_ms=round(float(np.median(ms)), 1), per_image_ms=[round(m, 1) for m in ms])
        print(f"{name:<26}{out[name]['median_ms']:>8} ms   {out[name]['per_image_ms']}", flush=True)
        del run
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
json.dump(out, open(os.path.join(cm.OUT, "speed.json"), "w"), indent=1)
