# Which object detector / segmenter is best? R-CNN vs YOLO vs DETR, tested

Every number is labelled:

- **MEASURED** = I ran it on this Mac (Apple M5, 16 GB, PyTorch MPS), 20 models.
- **PUBLISHED** = read from an official page or paper; not run by me. Different sources use different GPUs and TensorRT builds, so published latencies are only comparable inside one source.
- **UNVERIFIED** = could not be confirmed. Do not cite.

## 1. Headline findings

1. **"R-CNN is more accurate but slower, YOLO is faster but less accurate" is only half true.** The slower part is true (Mask R-CNN ~700 ms vs YOLO11m ~31 ms here). The accuracy part depends on the metric:
   - Threshold-free accuracy (mAP, section 3): Mask R-CNN **0.593** mAP50-95 sits *level with* YOLO11m/l/26m (0.59-0.60), and *below* YOLO11x (0.643) and RT-DETR-L (0.653).
   - **Where R-CNN really wins is recall**: it finds the most objects, especially small ones and in crowded scenes (section 4). It pays for that with many extra low-confidence detections (precision 0.64 at confidence 0.5 vs 0.92-0.94 for YOLO).
2. **For masks in crowded scenes, Mask R-CNN is the only model here that produced a mask count close to the number of objects** (47 / 40 / 32 masks on images holding 40 / 42 / 38 labelled objects). The best YOLO seg models produced 3-21 per image. (Counts, not verified-correct masks; see section 4.)
3. **YOLO is the clear speed/size winner:** YOLO26n 6 ms and 5.5 MB, YOLO11n-seg 9 ms for boxes *and* masks.
4. **The ranking at the top is a transformer, not YOLO or R-CNN:** RT-DETR-L has the best mAP here (0.653) at 86 ms. Published tables put RF-DETR/D-FINE/LW-DETR above YOLO too (section 5), but I could not run those.

## 2. Method and caveats (read before quoting anything)
- **Test data:** `coco128` = 128 COCO *train2017* images with 929 labelled boxes (33 "crowded" images with at least 10 objects hold 585 of them). **All these models were trained on COCO train2017, so absolute scores are inflated** (YOLO11x scores mAP50-95 0.643 here vs 0.547 published on val2017). Use the *ordering* and the crowded-scene behaviour, not the absolute numbers.
- **mAP** = COCO-style 101-point AP averaged over classes, from all detections with confidence >= 0.05 (threshold-free). **Precision/recall @0.5** use only confidence >= 0.5 and are **not comparable across families**: R-CNN scores are higher on average than YOLO's, so a fixed 0.5 cut-off favours different things.
- **Speed:** median over 5 images (2 warm-ups + median of 5 runs each), one model loaded at a time, includes each library's own pre/post-processing. YOLO input is resized to 640 px, torchvision's to 800 px. Run-to-run variation between sessions was up to ~30% (e.g. Mask R-CNN on `bus`: 352 ms in one run, 450 ms in another); Mask R-CNN also gets slower with more objects (450 ms with 5, ~730 ms with 40+).
- **Masks:** there are no ground-truth masks in coco128 (boxes only), so mask *quality* was judged by eye on a few images, and mask *count* is compared with the number of labelled objects.
- **`yolo11m` = `yolo11m-seg` and `yolo11l` = `yolo11l-seg` on boxes.** In Ultralytics' official files their backbone and box head are identical (649/649 and 1015/1015 matching tensors, max box difference 0.0), so box metrics are identical by construction; the seg version only adds a mask head (~25-40 ms more here). YOLO26m-seg differs from YOLO26m (only 124 of 768 tensors match).
- Nothing here was tuned to improve a model's score.

## 3. MEASURED: accuracy vs ground truth + speed (20 models, sorted by mAP50-95)

| Model | Task | Params (M) | Weights (MB) | Median ms | mAP50 | mAP50-95 | Precision@0.5 | Recall@0.5 | Recall, small objects | Recall, crowded images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RT-DETR-L | box | 33.0 | 66.5 | 86 | 0.849 | 0.653 | 0.884 | 0.671 | 0.40 | 0.59 |
| YOLO11x | box | 57.0 | 114.6 | 97 | 0.786 | 0.643 | 0.932 | 0.561 | 0.24 | 0.45 |
| YOLO12m | box | 20.2 | 40.9 | 46 | 0.775 | 0.616 | 0.934 | 0.521 | 0.20 | 0.41 |
| YOLO11l | box | 25.4 | 51.4 | 45 | 0.752 | 0.599 | 0.923 | 0.503 | 0.17 | 0.39 |
| YOLO11l-seg | mask | 27.7 | 56.1 | 72 | 0.752 | 0.599 | 0.923 | 0.503 | 0.17 | 0.39 |
| YOLO26l-seg | mask | 31.5 | 63.7 | 82 | 0.751 | 0.599 | 0.914 | 0.516 | 0.23 | 0.41 |
| YOLO26m | box | 21.9 | 44.3 | 38 | 0.754 | 0.594 | 0.913 | 0.534 | 0.25 | 0.43 |
| **Mask R-CNN R50-FPN v2** | mask | 46.4 | 185.8 | 696 | **0.798** | 0.593 | 0.642 | **0.783** | **0.58** | **0.73** |
| YOLO26l | box | 26.3 | 53.2 | 47 | 0.759 | 0.592 | 0.918 | 0.539 | 0.27 | 0.44 |
| YOLO11m | box | 20.1 | 40.7 | 31 | 0.761 | 0.589 | 0.931 | 0.495 | 0.17 | 0.39 |
| YOLO11m-seg | mask | 22.4 | 45.4 | 71 | 0.761 | 0.589 | 0.931 | 0.495 | 0.17 | 0.39 |
| YOLO26m-seg | mask | 27.1 | 54.8 | 70 | 0.741 | 0.582 | 0.920 | 0.518 | 0.21 | 0.41 |
| Faster R-CNN R50-FPN v2 | box | 43.7 | 175.2 | 378 | 0.782 | 0.574 | 0.629 | 0.775 | 0.58 | 0.71 |
| YOLO11s | box | 9.5 | 19.3 | 13 | 0.717 | 0.549 | 0.921 | 0.464 | 0.16 | 0.34 |
| YOLO12n | box | 2.6 | 5.6 | 8 | 0.631 | 0.493 | 0.939 | 0.366 | 0.09 | 0.25 |
| YOLO11n | box | 2.6 | 5.6 | 7 | 0.624 | 0.459 | 0.942 | 0.385 | 0.12 | 0.27 |
| YOLO26n | box | 2.6 | 5.5 | **6** | 0.606 | 0.453 | 0.919 | 0.352 | 0.08 | 0.24 |
| YOLO26n-seg | mask | 3.1 | 6.7 | 9 | 0.611 | 0.453 | 0.942 | 0.351 | 0.10 | 0.25 |
| YOLO11n-seg | mask | 2.9 | 6.2 | 9 | 0.607 | 0.444 | 0.920 | 0.344 | 0.09 | 0.23 |
| Faster R-CNN MobileNetV3 | box | 19.4 | 77.8 | 118 | 0.625 | 0.429 | 0.658 | 0.546 | 0.22 | 0.46 |

Raw data: `results/coco128_eval.json`, `results/speed.json`, `results/summary.json`; tables regenerate with `python make_tables.py`.

## 4. MEASURED: crowded scenes, boxes and masks produced

The three most crowded images in coco128 hold 40, 42 and 38 labelled objects (`results/sidebyside_000000000164.jpg`, `..196.jpg`, `..315.jpg`: the input first, then every model's output on the same image).

| Model | image 0164 (40 labelled) | image 0196 (42) | image 0315 (38) | Masks produced |
| --- | ---: | ---: | ---: | --- |
| **Mask R-CNN R50-FPN v2** | 47 | 40 | 32 | **47 / 40 / 32** |
| Faster R-CNN R50-FPN v2 | 46 | 40 | 32 | no masks |
| RT-DETR-L | 23 | 33 | 22 | no masks |
| Faster R-CNN MobileNetV3 | 20 | 22 | 21 | no masks |
| YOLO11x | 6 | 20 | 17 | no masks |
| YOLO11l-seg | 4 | 21 | 13 | 4 / 21 / 13 |
| YOLO26l-seg | 10 | 15 | 15 | 10 / 15 / 15 |
| YOLO11m-seg | 3 | 18 | 16 | 3 / 18 / 16 |
| YOLO26m-seg | 5 | 17 | 16 | 5 / 17 / 16 |
| YOLO26n-seg | 2 | 7 | 5 | 2 / 7 / 5 |
| YOLO11n-seg | 1 | 8 | 7 | 1 / 8 / 7 |

- **Reading it:** image 0164 is mostly small objects, and every YOLO model finds fewer than 11 of its 40. The R-CNNs find 46-47, but 47 on an image with 40 labelled objects also means some extras (false positives or unlabelled objects).
- **A count is not mask accuracy.** I did not measure whether each mask is correct (no ground-truth masks). On the simple images I looked at, Mask R-CNN's masks were the cleanest, YOLO11m-seg's close, and YOLO11n-seg's had visible holes.
- On a simple scene (`bus`, `zidane`) the models agree much more: all 20 find the 5 bus objects within one object of each other. Crowding is what separates them.

## 5. PUBLISHED: COCO val2017 box AP (from official pages)

| Family | Size | AP | Params | Latency | Source / note |
| --- | --- | ---: | ---: | ---: | --- |
| YOLO11 | n / s / m / l / x | 39.5 / 47.0 / 51.5 / 53.4 / 54.7 | 2.6 / 9.4 / 20.1 / 25.3 / 56.9 M | 1.5 / 2.5 / 4.7 / 6.2 / 11.3 ms (T4, TensorRT10) | Ultralytics docs |
| YOLO12 | n / s / m / l / x | 40.6 / 48.0 / 52.5 / 53.7 / 55.2 | 2.6 / 9.3 / 20.2 / 26.4 / 59.1 M | 1.64 / 2.61 / 4.86 / 6.77 / 11.79 ms (T4) | Ultralytics docs |
| YOLO26 | n / s / m / l / x | 40.9 / 48.6 / 53.1 / 55.0 / 57.5 | 2.4 / 9.5 / 20.4 / 24.8 / 55.7 M | 1.7 / 2.5 / 4.7 / 6.2 / 11.8 ms (T4) | Ultralytics docs (NMS-free head) |
| RT-DETR | L / X | 53.0 / 54.8 | not stated | page's FPS and ms figures disagree | Ultralytics docs |
| D-FINE | N / S / M / L / X | 42.8 / 48.5 / 52.3 / 54.0 / 55.8 | 4 / 10 / 19 / 31 / 62 M | 2.12 / 3.49 / 5.62 / 8.07 / 12.89 ms (T4) | official repo; Roboflow's page lists slightly different values |
| RF-DETR | N / S / M / L / XL / 2XL | 48.4 / 53.0 / 54.7 / 56.5 / 58.6 / 60.1 | 30.5 / 32.1 / 33.7 / 33.9 / 126.4 / 126.9 M | 2.3 / 3.5 / 4.4 / 6.8 / 11.5 / 17.2 ms (T4, TensorRT FP16) | Roboflow docs / PyPI; Apache-2.0 for N-L, Platform Model License for XL, 2XL |
| LW-DETR | tiny / small / medium / large / xlarge | 42.6 / 48.0 / 52.5 / 56.1 / 58.3 | not read | not read | official repo README; a Hugging Face note says Objects365 pretraining adds ~5 mAP, so check pretraining before comparing |
| Faster R-CNN | MobileNetV3 / R50-FPN / R50-FPN v2 | 32.8 / 37.0 / 46.7 | 19.4 / 41.8 / 43.7 M | 4.5 / 134 / 280 GFLOPS | torchvision docs |
| Mask R-CNN | R50-FPN / R50-FPN v2 | box 37.9 / 47.4; **mask 34.6 / 41.8** | 44.4 / 46.4 M | 134 / 334 GFLOPS | torchvision docs |
| RetinaNet / FCOS / SSD | v2 / R50 / VGG16 | 41.5 / 39.2 / 25.1 | 38.2 / 32.3 / 35.6 M | 152 / 128 / 35 GFLOPS | torchvision docs |
| YOLO-ULM (CVPR 2026) | N | 41.6 | not read | 1.52 ms (T4) | authors' abstract only; code release not confirmed |

- **Highest published AP:** RF-DETR-2XL 60.1, RF-DETR-XL 58.6, LW-DETR-xlarge 58.3, YOLO26x 57.5, RF-DETR-L 56.5.
- **Matched size:** YOLO11m (51.5 AP, 20M params) already beats Faster R-CNN R50-FPN v2 (46.7 AP, 44M).
- **Instance masks:** the only published mask AP I read is Mask R-CNN's (41.8 for the v2). I found no mask-AP table for YOLO11-seg or YOLO26-seg on the pages I opened.
- **Not run by me:** RF-DETR, D-FINE, LW-DETR, YOLO-ULM (no weights/code downloaded). Their rows are from the sources above only.

### UNVERIFIED (do not cite)
YOLOv8/v9/v10 numbers, RT-DETRv2/v3, YOLO-ULM T/S/M/L/X and Turbo, any RF-DETR size called Atto/Femto/Pico, and Mask R-CNN figures from third-party tables. Video-RNN claims: Recurrent Mf-SSD +2.7 mAP on KITTI (MERL TR2018-137) and Tripathi et al. +7.1 mAP on YouTube-Objects were checked against their papers' abstracts.

## 6. Answers (strictly from the evidence)
| Question | Answer | Evidence |
| --- | --- | --- |
| Fastest | YOLO26n 6 ms, YOLO11n 7, YOLO12n 8; with masks YOLO11n-seg / YOLO26n-seg 9 ms | measured, 5 images |
| Most accurate (threshold-free) | RT-DETR-L 0.653 mAP50-95, then YOLO11x 0.643; Mask R-CNN has the best mAP50 among mask models (0.798) | measured, contaminated data |
| Most accurate, published val2017 | RF-DETR-2XL 60.1 AP (not run by me) | published |
| Most precise | YOLO family: 0.91-0.94 precision at 0.5, vs 0.63-0.66 for the R-CNNs (not comparable across families; mAP is the fair view) | measured |
| Captures the most objects | Mask R-CNN / Faster R-CNN R50 (recall 0.78 / 0.78; small objects 0.58); next RT-DETR-L (0.67); best YOLO is YOLO11x (0.56) | measured |
| How many objects can it mask in a crowd | Mask R-CNN: 47 / 40 / 32 masks on 40 / 42 / 38-object images; best YOLO seg models: 3-21 | measured counts, not mask correctness |
| Best masks per millisecond | YOLO11m-seg or YOLO26m-seg (~70 ms, 0.58-0.59 mAP50-95, ~10x faster than Mask R-CNN); nano seg models (9 ms) if the scene is simple | measured |
| Best overall | none: RT-DETR-L for boxes, Mask R-CNN for crowded instance masks, YOLO11n/26n for edge speed | |

## 7. Suggested honest post claims
- "I tested 20 detectors/segmenters on my laptop: R-CNN's edge is **recall**, not average accuracy."
- "At matched size YOLO11m scores like Mask R-CNN on mAP and runs ~20x faster; the R-CNN family still finds far more objects in crowded scenes, and was the only one whose mask count approached the real object count."
- Always add: coco128 images were in the models' training data (so ordering, not absolute scores), one Mac, boxes labelled by COCO, mask quality judged by eye.
- Don't claim: any ranking of RF-DETR / D-FINE / LW-DETR / YOLO-ULM from my runs (they weren't run), or mask accuracy rankings.

## 8. Files and reproduction
`compare_models.py` (side-by-side grids + proxy consensus), `eval_coco128.py` (mAP against ground truth), `speed_bench.py` (clean timings), `make_tables.py`, `make_charts.py` (the two charts), `make_post_image.py` (crowded-scene strip), `results/` (charts, grids `sidebyside_*.jpg`, JSON), and after `python ../get_assets.py`: `weights/`, `images/`, `data/coco128/` (not stored in git).
New to this? Read `../docs/LEARNING_GUIDE.md` first; the post draft is `../docs/LINKEDIN_POST.md`.
Environment: Python 3.12 venv with torch 2.15.0, torchvision 0.29.2, ultralytics 8.4.212, `lap`; run with `python -I script.py`.
Downloads used: YOLO11/12/26 and RT-DETR-L weights from github.com/ultralytics/assets (675 MB total), torchvision Faster/Mask R-CNN weights (about 440 MB), `coco128.zip` (7.0 MB).
