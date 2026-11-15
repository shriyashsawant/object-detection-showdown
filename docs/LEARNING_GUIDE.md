# Learning guide: how object detectors work, how to judge them, and how video memory (RNN) fits in

Written for someone who has seen "YOLO" and "R-CNN" mentioned but never had them explained. Every idea comes with
a plain-English picture first, then **what we actually measured** in this project.

Contents
1. [Three jobs: classify, detect, segment](#1-three-jobs-classify-detect-segment)
2. [How a detector is graded](#2-how-a-detector-is-graded) (confidence, IoU, precision, recall, AP, mAP)
3. [The three families: R-CNN, YOLO, transformers](#3-the-three-families)
4. [Speed, size and why numbers from different places do not match](#4-speed-size-and-why-numbers-do-not-match)
5. [What we found (the lessons)](#5-what-we-found-the-lessons)
6. [Video: frames, CNN, and an RNN that keeps state](#6-video-frames-cnn-and-an-rnn-that-keeps-state)
7. [Mistakes we made and fixed](#7-mistakes-we-made-and-fixed)
8. [Glossary](#8-glossary)
9. [Try it yourself](#9-try-it-yourself)

---

## 1. Three jobs: classify, detect, segment

Take a photo of a street with a bus and three people.

| Job | Question it answers | Output for the photo |
| --- | --- | --- |
| **Classification** | "What is in this picture?" | one label: `street` |
| **Object detection** | "What is here, and where?" | a **box** + label + confidence for each thing: `bus 0.99`, `person 0.97` (three times) |
| **Instance segmentation** | "Which exact pixels belong to each thing?" | a **mask** (an outline filled in) for each bus and each person, separately |

- A box is four numbers (left, top, right, bottom). A mask is a pixel-by-pixel yes/no picture. Masks are more precise but cost more computing.
- "Instance" means each person gets their own mask. (Semantic segmentation would paint all "person" pixels one colour.)
- This project compares **detectors** (boxes) and **instance segmenters** (boxes + masks).

**Confidence** is the model's own score (0 to 1) for "I think this box is a real object of this class". It is a *score*, not a guaranteed probability, and different model families score on different scales (this matters in section 2).

---

## 2. How a detector is graded

To say "model A is better than model B" you need a fair way to score them. Here is the whole toolkit.

### IoU: did the box land on the object?
**Intersection over Union** = (area where the predicted box and the true box overlap) / (area covered by either of them).
- IoU 1.0 = identical boxes, 0 = no overlap. By convention a prediction **counts as correct at IoU 0.5** (about "half-ish overlapping") and a very exact one at 0.9.

### The four outcomes
For each real object (**ground truth**, labelled by humans) and each predicted box:
- **True positive (TP):** the model drew a box on a real object, right class, IoU high enough.
- **False positive (FP):** the model drew a box where there is no such object (a "ghost"). Example from our test: one model drew a **"tennis racket"** on a pointing arm.
- **False negative (FN):** a real object the model missed.

### Precision and recall (the two questions)
- **Precision** = TP / (TP + FP) -> "Of the boxes the model drew, how many were right?" (Is it trustworthy?)
- **Recall** = TP / (TP + FN) -> "Of the real objects, how many did the model find?" (Does it miss things?)

They pull against each other. A model that only draws boxes when it is *very* sure will be precise but miss things; a model that draws boxes on anything vaguely object-like finds a lot but also draws many ghosts.

### The confidence threshold trap
Every detector outputs many boxes with scores; you choose a **threshold** (say 0.5) and keep only boxes above it. Precision and recall depend heavily on that choice, and **different families use different score scales.** In our first evaluation at a fixed 0.5:
- the R-CNN models had **recall 0.78, precision 0.64** (lots of boxes, many ghosts);
- the YOLO models had **recall about 0.5, precision about 0.93** (fewer boxes, mostly right).

That looked like "R-CNN is sloppy, YOLO is precise", but it was mostly the two families scoring on different scales, not a real quality gap. Comparing at one fixed threshold is **not a fair test across families**. We fixed it with AP (next).

### AP and mAP: one fair number
Instead of one threshold, lower the threshold step by step from strict to lenient. Plot precision against recall as you go (the **precision-recall curve**). The area under that curve is the **Average Precision (AP)** for one class (e.g. "person"). Average across classes and you have **mAP** (mean AP).
- **mAP50** = a box counts as correct at IoU >= 0.5 (forgiving about exact placement).
- **mAP50-95** = average over IoU thresholds 0.5, 0.55, ..., 0.95. Stricter: it rewards boxes that fit tightly. This is the standard COCO score ("AP").
- Higher is better. Because it sweeps every threshold, **mAP is the fair way to compare different families**, which is why our main accuracy numbers are mAP.

### Object size
COCO groups objects by pixel area: **small** (under 32x32 pixels), **medium**, **large** (over 96x96). Small objects are the hardest, so recall on small objects is a good test of how much detail a model can see.

---

## 3. The three families

### 3a. R-CNN family: "propose, then look closely" (two stages)
- **Idea:** first find *places that might contain something* (region proposals), then examine each place carefully to decide what it is and refine its box.
- **History in one breath:** R-CNN (2014) ran a CNN on every proposed region (very slow) -> Fast R-CNN shared the heavy computation across regions -> **Faster R-CNN** let the network learn its own proposals (the *Region Proposal Network*) -> **Mask R-CNN** added a third branch that draws a mask inside each box.
- **Strength:** careful, tends to find many objects, including small ones. **Weakness:** two stages = slow; heavy files.
- Tested here: Faster R-CNN (MobileNetV3 and ResNet50-FPN v2) and Mask R-CNN (ResNet50-FPN v2).
- *(Note: "RNN" is something else entirely. It is a recurrent network for sequences, see section 6. "Faster RNN" in some posts is a typo for Faster **R-CNN**.)*

### 3b. YOLO family: "look once" (one stage)
- **Idea:** *You Only Look Once*. One network looks at the whole image one time and directly outputs all boxes (and, for `-seg` versions, masks). No separate proposal step.
- **NMS:** a one-pass detector often draws several overlapping boxes on the same object, so a clean-up step called **non-maximum suppression** keeps the best and deletes near-duplicates. (YOLO26 has a design that avoids this step.)
- **Strength:** very fast, tiny files, easy to run on phones and small devices. **Weakness:** the small models miss more small and crowded objects.
- Tested here: YOLO11 (n, s, m, l, x), YOLO12 (n, m), YOLO26 (n, m, l) and `-seg` versions. The letters are sizes: **n**ano, **s**mall, **m**edium, **l**arge, e**x**tra-large. Bigger = more accurate, slower, bigger file.

### 3c. Transformer detectors (DETR family): "treat detection as set prediction"
- **Idea:** use **attention** (the mechanism behind modern language models) so each part of the image can look at every other part, and have the network output a fixed-size *set* of objects directly, matching predictions to true objects during training. This removes hand-made steps like anchors and NMS.
- **RT-DETR** is the real-time version. Newer ones (RF-DETR, D-FINE, LW-DETR) sit at the top of published accuracy tables.
- Tested here: **RT-DETR-L only**. The others are in the README as *published numbers, not run by us*.

### Pieces you will see in names
- **Backbone** (ResNet50, MobileNetV3, ...): the CNN that turns pixels into features. **FPN** (feature pyramid): combines features at several scales so both small and large objects are handled. `v2` = an improved training recipe.
- **Params** = number of learned weights (millions, M); **FLOPs/GFLOPS** = how much arithmetic one image costs. Both are rough guides to size and speed.

---

## 4. Speed, size and why numbers do not match

- **Latency** = time for one image. It depends on the **hardware** (a datacentre GPU with TensorRT is not a MacBook), the **software build**, image size, and what is included (resizing, post-processing like NMS and mask drawing).
- That is why a model "5 ms" on one website and "30 ms" in our table are not in conflict. **Only compare latencies from the same source and setup.**
- Our numbers: Apple M5 laptop (GPU via PyTorch "MPS"), median of several runs after warm-ups, **one model loaded at a time**.
- Real timing pitfalls we hit (section 7): the first call on a new image size is slower (the GPU is "warming up"), and having 20 models loaded at once made everything about 1.3 to 1.8x slower.
- Mask R-CNN gets slower when there are many objects (about 450 ms with 5 objects, 730 ms with 40+), because every object needs a mask pasted back onto the image.

---

## 5. What we found (the lessons)

**Setup:** 20 models, one MacBook, **128 labelled COCO images** (929 objects; 33 "crowded" images with 10+ objects). Full tables are in [`detector_test/README.md`](../detector_test/README.md).

| # | Lesson | Evidence |
| --- | --- | --- |
| 1 | **"R-CNN is more accurate, YOLO is faster" is only half true.** On average accuracy (mAP50-95), Mask R-CNN is level with mid-size YOLO, not ahead. | Mask R-CNN 0.593 vs YOLO11m 0.589; YOLO11x 0.643 |
| 2 | **Speed is the real gap.** | Mask R-CNN about 696 ms vs YOLO11m about 31 ms (about 22x); YOLO11n-seg (boxes **and** masks) about 9 ms |
| 3 | **R-CNN's real strength is finding things** (recall), especially small and crowded. | recall 78% vs YOLO11m 50%; small objects 58% vs about 18%; crowded scenes 73% vs 39% |
| 4 | **In crowds, only Mask R-CNN produced a mask count near the real number of objects.** | 47 / 40 / 32 masks on images with 40 / 42 / 38 labelled objects; best YOLO-seg: 3 to 21 |
| 5 | **Neither family won on accuracy.** A transformer did. | RT-DETR-L 0.653 mAP50-95 |
| 6 | **Compare at matched size, not by family name.** | published: YOLO11m 51.5 AP (20M params) beats Faster R-CNN R50-FPN v2 46.7 AP (44M) |
| 7 | **A fixed confidence threshold misleads across families** (see section 2). | at 0.5 the precision gap looked huge (0.64 vs 0.93); threshold-free mAP50 showed the two families close (Mask R-CNN 0.798 vs YOLO 0.76 to 0.79) |
| 8 | **Test data that models trained on inflates scores.** Our 128 images are a slice of COCO *training* data, so YOLO11x scores 0.643 here but 0.547 on its official validation set. | use the **order**, not the absolute numbers |
| 9 | **Count of masks is not quality of masks.** 47 masks on a 40-object image means some are extras. We judged mask *quality* by eye on a few images only. | we have no ground-truth masks |
| 10 | **Different file names can hold the same model.** In the official files, `yolo11m` and `yolo11m-seg` have identical box weights, so their box scores are identical by construction. | 649 of 649 tensors equal |

**Choose by the job, not the acronym**
- Need to **find everything**, even small or crowded things, and speed is not critical -> R-CNN family.
- Need the **best average accuracy** -> a strong transformer (RT-DETR-L here; RF-DETR and others in published tables).
- Need **speed or a tiny file** (phone, camera, drone) -> a nano/small YOLO.

---

## 6. Video: frames, CNN, and an RNN that keeps state

**Question:** can we turn a video into frames -> arrays -> CNN output -> an RNN that remembers what happened? **Yes**, and `video_state/` does exactly that. Here it is step by step, with the array shapes.

1. **Video to frames.** A video is just many pictures. OpenCV reads it frame by frame. We keep every 2nd frame (30 fps becomes 15 steps per second). Each frame is an array of shape `(height, width, 3)` (three colour channels).
2. **Frames to features with a CNN.** A CNN looks at one frame and compresses it into a short list of numbers (a **feature vector**, here 256 numbers) that captures what is in it. Stack all frames: shape `(T, 256)`, where T is the number of frames. We also add easy measurements: where the main object is, how big, how fast things move (optical flow).
3. **A CNN has no memory.** It treats every frame separately, so it cannot know "the cat was turning its head a second ago".
4. **An RNN adds memory.** A recurrent network reads the rows **in order**, and after each step it updates a **hidden state** (a small list of numbers, 32 in our GRU) summarising everything seen so far. At step t it sees: the new frame's features + the hidden state from step t-1. That hidden state **is** the "state about the video".
   - **GRU** (Gated Recurrent Unit) is a popular RNN. Its "gates" are small learned switches that decide how much old memory to keep and how much new information to let in.
5. **What do we train it to do?**
   - *Unsupervised* (no labels): predict the **next** frame's features. If it does that well, the hidden state must have captured the motion. Then we cluster the hidden states into "states" and describe them.
   - *Supervised* (labels): we labelled time ranges of our clip (`facing_camera`, `head_angled_down`, `head_turned_away`) and trained the GRU to output the right label at every step.
6. **Tracking** (ByteTrack) gives every object an ID that stays the same across frames, so "the same cat" is recognised through the clip.

### What happened (honest results on one 20-second clip)
- **Unsupervised:** the GRU did **worse than the trivial rule "next frame = this frame"** (error 0.38 vs 0.17), and its states flickered more than clustering each frame alone.
- **Supervised:** accuracy GRU 0.569, memoryless linear model 0.558, always-guess-the-majority 0.521. **No clear win for the RNN**, and plain smoothing cut label flicker more than the memory did.
- **Why:** one short clip is tiny training data; poses last seconds, so at 15 steps per second the "state" barely helps; and the features say little about head direction.
- **Lesson:** **always test against a simple baseline** (repeat last frame; smooth the per-frame answers). An RNN pays off with lots of labelled sequences or long-range questions ("did X happen before Y?"), not a 20-second demo.
- **Also:** YOLO sometimes switched the cat's label to "dog" for a few frames even though the *tracker* kept the same ID, so labels can flicker while identity stays stable.

**Other ways to use video:** 3D CNNs and video transformers for actions; adding temporal context on top of a detector (published video-detection gains: Tripathi et al. +7.1 mAP on YouTube-Objects; Recurrent Mf-SSD +2.7 mAP on KITTI).

---

## 7. Mistakes we made and fixed

Real mistakes, kept here because they teach more than a clean story.

1. **Unfair first evaluation.** Precision/recall at a fixed 0.5 favoured one family. Fix: threshold-free mAP.
2. **Distorted timing (a).** Timing each image once made YOLO11n look like 49 ms; it is about 6 ms. Cause: new image sizes trigger GPU warm-up. Fix: warm-ups + repeated runs on fixed images.
3. **Distorted timing (b).** Keeping 20 models in memory made all of them slower. Fix: load one model at a time.
4. **Suspicious identical scores** (`yolo11m` = `yolo11m-seg`). Checked the files: identical box weights (not a bug).
5. **First video GRU lost badly** (0.63 vs 0.17). We changed it to predict the *change* from the current frame (0.38) and stopped there rather than keep tuning until it looked good. Both numbers are reported.
6. **A coding bug:** the supervised GRU was trained inside a "no gradients" block, so it crashed. Fixed by moving the training out.
7. **Overstated claims in early drafts** (an unverified "+2.7 mAP", a mixed-up size ratio, "transformers lead" when only one was run) were corrected against sources before publishing.

---

## 8. Glossary

| Term | Plain meaning |
| --- | --- |
| Ground truth | the human-made correct labels |
| Bounding box | rectangle around an object (4 numbers) |
| Mask | pixel-by-pixel outline of an object |
| Confidence | the model's score for one prediction |
| Threshold | the cut-off score for keeping a prediction |
| IoU | overlap between predicted and true box (0 to 1) |
| TP / FP / FN | right box / ghost box / missed object |
| Precision | of the boxes drawn, share that were right |
| Recall | of the real objects, share that were found |
| AP, mAP | area under the precision-recall curve, averaged over classes |
| mAP50-95 | mAP averaged over strict IoU thresholds (the COCO score) |
| NMS | clean-up that removes duplicate boxes |
| Backbone | the CNN that extracts features |
| FPN | multi-scale feature combiner for small and large objects |
| Parameters | learned weights, in millions (M) |
| FLOPs | arithmetic needed per image |
| Latency | time per image |
| Two-stage / one-stage | propose-then-classify (R-CNN) / predict everything at once (YOLO) |
| Transformer / attention | model design where every part looks at every other part |
| CNN | convolutional neural network, good at images |
| RNN / GRU | network with memory for sequences / a popular RNN |
| Hidden state | the RNN's running summary of what it has seen |
| Self-supervised | learn from the data itself (e.g. predict the next frame), no labels |
| Overfitting / contamination | scoring well because the test data was seen in training |

---

## 9. Try it yourself

1. Run `python get_assets.py` (downloads weights and the test images), then `python detector_test/compare_models.py` and open `detector_test/results/sidebyside_*.jpg`. Look at *what each model misses*, not just the numbers.
2. Open `detector_test/results/chart_speed_vs_accuracy.png`. Find the point that gives the most accuracy for under 50 ms.
3. Change the confidence threshold in `compare_models.py` (`CONF`) to 0.25 and 0.75. Watch which family's box count changes the most (section 2's trap).
4. Run the video demo on your own short clip and compare the GRU to "repeat last frame" yourself.
5. Stretch: test a model that is only in the published tables (RF-DETR, D-FINE) and see whether the published order holds on your images.
