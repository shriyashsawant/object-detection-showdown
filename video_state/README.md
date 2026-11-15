# Video -> frames -> arrays -> CNN -> RNN that keeps state

**Can this be done? Yes.** `video_state.py` does exactly this, end to end, on a real clip. What it does *not* do is prove that an RNN is
better than simpler methods: on this clip it was not (section 3).

## 1. The pipeline (all in `video_state.py`)
| Step | What happens | Shape |
| --- | --- | --- |
| Frames | OpenCV reads the video; every 2nd frame is kept (30 fps -> 15 steps/s) | one `(H, W, 3)` array per frame |
| Detector + tracker | YOLO11n-seg finds and masks objects; ByteTrack gives each object an ID that persists across frames | boxes, masks, ids |
| CNN features | YOLO's own feature vector for the frame (the pooled layer before its head) | `(T, 256)` |
| Measurements | main-object box and mask position/size, confidence, object count, optical-flow motion, frame difference | `(T, 11)` |
| Input rows | CNN vector reduced to 16-d by PCA + the 11 measurements, standardised | `X = (T, 27)` |
| RNN | a GRU (32 hidden units) reads the rows in time order and learns to predict the next row (self-supervised, no labels). Its hidden state is the running memory | `hidden = (T, 32)` |
| States | the hidden states are clustered (k = 3-6, chosen by silhouette); each state is described from measured numbers | one label per step |
| Outputs | `results/annotated.mp4` (masks, track id, state name, timeline bar), `results/analysis.json`, `results/arrays.npz` | |

Run: `python video_state.py your_clip.mp4` (needs `ultralytics`, `torch`, `lap`, `opencv-python`, `numpy`). Weights: `../detector_test/weights/yolo11n-seg.pt`.

## 2. The clip and what was detected
a 20-second phone clip (not included in the repo): 20 s, 608x1080, an orange cat on a wall (looks at the camera, lowers its head, turns away, returns). 303 steps analysed.
The cat was detected in every step and tracked as one ID throughout. YOLO sometimes switched the label to "dog" (e.g. `dog 0.62` while the head is lowered), so **class names flicker even when the track does not**.

## 3. Honest results (these are the ones from the final run; nothing was tuned afterwards to look better)
| Check (held-out last 20% of the clip, 60 steps) | Result |
| --- | --- |
| Next-step error, GRU predicting the change from the current frame | 0.38 |
| Next-step error, GRU predicting the whole frame (first design) | 0.68 |
| Baseline: "next frame = this frame" | **0.17** (best) |
| Baseline: "always the training average" | 0.79 |
| State changes, GRU hidden states clustered | 24 |
| State changes, clustering each frame on its own (no memory) | 11 |

- **The GRU lost to repeating the last frame**, and its states flickered *more*, not less. State A absorbed 68% of the time and merged three visibly different poses (eyes closed, head tilted, head down).
- Why this is expected: one 20-second clip gives about 240 training steps. An RNN needs far more data, or labels, before its memory beats a strong trivial baseline at 15 steps per second.
- An earlier version (full-frame prediction, 200 epochs) did worse (0.63 vs 0.17). I changed the design to predict the *change* and cut epochs to 60 for a fair re-run, then stopped; both numbers are kept above.
- The descriptions ("cat, fast movement, medium size, left, high") come from tercile cut-offs of this clip's own measurements. They are not a caption model, and "left/high/fast" are relative to this clip.

## 4. Supervised version (built): labels + GRU classifier (`video_state_supervised.py`)
Labels (`labels.csv`) were written by me from looking at the clip every 0.5 s: `facing_camera` (52% of steps), `head_angled_down` (26%), `head_turned_away` (22%); 6 label changes in 20 s. They are one person's judgement on one clip.
Test: 5 contiguous time blocks, each held out once (0.5 s purge next to the block), 3 random seeds, predictions pooled.

| Model | Accuracy | Macro-F1 | Label flips (true: 6) |
| --- | ---: | ---: | ---: |
| Always the most common class | 0.521 | 0.228 | 0 |
| Per-frame linear classifier (no memory) | 0.558 [0.548-0.571] | 0.554 | 28.3 |
| Same + causal smoothing of scores over the last 8 steps | 0.543 [0.525-0.568] | 0.549 | 10.7 |
| **GRU (32 units) carrying state** | 0.569 [0.554-0.578] | 0.560 | 24.3 |

- **The GRU is not clearly better than a memoryless classifier**: 0.569 vs 0.558, and the seed ranges overlap. All three trained models are only modestly above always guessing the majority class (0.52), though macro-F1 (0.55-0.56 vs 0.23) shows they do recognise the poses somewhat.
- **Cheap smoothing cut label flips more (10.7) than the GRU's memory did (24.3)**, at about the same accuracy. For this clip, memory is not the bottleneck.
- Why: the poses are long (each lasts 2-5 s), the held-out blocks remove most of a pose from training (the `head_turned_away` pose exists only in 7.0-11.4 s), and the features (YOLO's pooled vector reduced to 16 numbers + box/mask measurements) say little about head direction.
- Run: `python video_state_supervised.py`. Output: `results/supervised.json` (includes confusion matrices).

## 5. What would make the RNN worth it
1. **More labelled clips** (dozens, not one) with several examples of every state, so a held-out clip still contains each pose in training.
2. **Better per-frame features** for the thing you care about (e.g. head-pose or keypoint models) instead of a pooled detector vector.
3. **More clips** (dozens, not one) so the GRU has enough sequences to learn from.
4. **Longer-range tasks** (events minutes long, counting entries/exits, "did X happen before Y"), where a memory helps more than at 15 steps per second.
5. Always compare against simple smoothing of per-frame results before claiming an RNN is needed (section 4 did).
6. Alternatives worth a look: 3D CNNs and video transformers for action recognition; temporal fusion on top of a detector for video detection (Tripathi +7.1 mAP on YouTube-Objects; Recurrent Mf-SSD +2.7 mAP on KITTI).

## 5. Privacy
The clip is your own footage and was processed only on this machine; `results/annotated.mp4` contains frames from it, so do not publish it unless you intend to.
