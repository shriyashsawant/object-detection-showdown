### Accuracy vs ground truth (coco128, 128 images, 929 objects; contaminated, see caveat) + speed (median ms over 5 images, M5 / MPS, one model loaded at a time)

| Model | Task | Params (M) | Weights (MB) | Median ms | mAP50 | mAP50-95 | Precision@0.5 | Recall@0.5 | Recall small objects | Recall on crowded images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rtdetr-l | box | 33.0 | 66.5 | 86 | 0.849 | 0.653 | 0.884 | 0.671 | 0.40 | 0.59 |
| yolo11x | box | 57.0 | 114.6 | 97 | 0.786 | 0.643 | 0.932 | 0.561 | 0.24 | 0.45 |
| yolo12m | box | 20.2 | 40.9 | 46 | 0.775 | 0.616 | 0.934 | 0.521 | 0.20 | 0.41 |
| yolo11l | box | 25.4 | 51.4 | 45 | 0.752 | 0.599 | 0.923 | 0.503 | 0.17 | 0.39 |
| yolo11l-seg | mask | 27.7 | 56.1 | 72 | 0.752 | 0.599 | 0.923 | 0.503 | 0.17 | 0.39 |
| yolo26l-seg | mask | 31.5 | 63.7 | 82 | 0.751 | 0.599 | 0.914 | 0.516 | 0.23 | 0.41 |
| yolo26m | box | 21.9 | 44.3 | 38 | 0.754 | 0.594 | 0.913 | 0.534 | 0.25 | 0.43 |
| Mask R-CNN R50-FPN v2 | mask | 46.4 | 185.8 | 696 | 0.798 | 0.593 | 0.642 | 0.783 | 0.58 | 0.73 |
| yolo26l | box | 26.3 | 53.2 | 47 | 0.759 | 0.592 | 0.918 | 0.539 | 0.27 | 0.44 |
| yolo11m | box | 20.1 | 40.7 | 31 | 0.761 | 0.589 | 0.931 | 0.495 | 0.17 | 0.39 |
| yolo11m-seg | mask | 22.4 | 45.4 | 71 | 0.761 | 0.589 | 0.931 | 0.495 | 0.17 | 0.39 |
| yolo26m-seg | mask | 27.1 | 54.8 | 70 | 0.741 | 0.582 | 0.920 | 0.518 | 0.21 | 0.41 |
| Faster R-CNN R50-FPN v2 | box | 43.7 | 175.2 | 378 | 0.782 | 0.574 | 0.629 | 0.775 | 0.58 | 0.71 |
| yolo11s | box | 9.5 | 19.3 | 13 | 0.717 | 0.549 | 0.921 | 0.464 | 0.16 | 0.34 |
| yolo12n | box | 2.6 | 5.6 | 8 | 0.631 | 0.493 | 0.939 | 0.366 | 0.09 | 0.25 |
| yolo11n | box | 2.6 | 5.6 | 7 | 0.624 | 0.459 | 0.942 | 0.385 | 0.12 | 0.27 |
| yolo26n | box | 2.6 | 5.5 | 6 | 0.606 | 0.453 | 0.919 | 0.352 | 0.08 | 0.24 |
| yolo26n-seg | mask | 3.1 | 6.7 | 9 | 0.611 | 0.453 | 0.942 | 0.351 | 0.10 | 0.25 |
| yolo11n-seg | mask | 2.9 | 6.2 | 9 | 0.607 | 0.444 | 0.920 | 0.344 | 0.09 | 0.23 |
| Faster R-CNN MobileNetV3 | box | 19.4 | 77.8 | 118 | 0.625 | 0.429 | 0.658 | 0.546 | 0.22 | 0.46 |

### The three most crowded images (ground truth: 42, 40, 38 objects): detections / masks at confidence 0.5

| Model | 0164 (GT 40) | 0196 (GT 42) | 0315 (GT 38) | Masks produced (same images) |
| --- | ---: | ---: | ---: | --- |
| rtdetr-l | 23 | 33 | 22 | no masks |
| yolo11x | 6 | 20 | 17 | no masks |
| yolo12m | 4 | 18 | 15 | no masks |
| yolo11l | 4 | 21 | 13 | no masks |
| yolo11l-seg | 4 | 21 | 13 | 4/21/13 |
| yolo26l-seg | 10 | 15 | 15 | 10/15/15 |
| yolo26m | 5 | 20 | 14 | no masks |
| Mask R-CNN R50-FPN v2 | 47 | 40 | 32 | 47/40/32 |
| yolo26l | 10 | 17 | 15 | no masks |
| yolo11m | 3 | 18 | 16 | no masks |
| yolo11m-seg | 3 | 18 | 16 | 3/18/16 |
| yolo26m-seg | 5 | 17 | 16 | 5/17/16 |
| Faster R-CNN R50-FPN v2 | 46 | 40 | 32 | no masks |
| yolo11s | 2 | 14 | 11 | no masks |
| yolo12n | 1 | 10 | 2 | no masks |
| yolo11n | 2 | 10 | 6 | no masks |
| yolo26n | 1 | 10 | 5 | no masks |
| yolo26n-seg | 2 | 7 | 5 | 2/7/5 |
| yolo11n-seg | 1 | 8 | 7 | 1/8/7 |
| Faster R-CNN MobileNetV3 | 20 | 22 | 21 | no masks |
