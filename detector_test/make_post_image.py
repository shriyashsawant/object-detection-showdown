"""Cut a 4-panel strip for the post from the saved crowded-scene grid: input | Mask R-CNN | YOLO11m-seg | YOLO11n-seg.

    python make_post_image.py   ->  results/post_crowded_strip.jpg
Panel positions follow the model order in compare_models.py (4 columns, panels of equal size):
0 input, 3 Mask R-CNN R50-FPN v2, 16 yolo11m-seg, 15 yolo11n-seg.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
grid = cv2.imread(os.path.join(HERE, "results", "sidebyside_000000000196.jpg"))
cols = 4
pw, ph = grid.shape[1] // cols, grid.shape[0] // 6
panels = [grid[(i // cols) * ph:(i // cols + 1) * ph, (i % cols) * pw:(i % cols + 1) * pw] for i in (0, 3, 16, 15)]
cv2.imwrite(os.path.join(HERE, "results", "post_crowded_strip.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved", np.hstack(panels).shape)
