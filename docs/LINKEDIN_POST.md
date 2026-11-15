# LinkedIn post: what to post and how to post it

Every number below comes from `detector_test/results/*.json` (the README tables use the same files). If someone challenges a number, you can re-run `python eval_coco128.py` and `python speed_bench.py` and show the output.

## Before you post (checklist)
- [ ] **Re-run it yourself** at least once (`get_assets.py`, then `eval_coco128.py` and `speed_bench.py`). Then "I tested" is literally true, and you will know where each number comes from.
- [ ] **Make the repo public** if you want to link it (it starts private). On GitHub: Settings, then "Danger Zone", then "Change visibility". Look through the repo first: no personal video is in it (`*.mp4` is git-ignored), but check you are happy with everything.
- [ ] Read `detector_test/README.md` section 2 (caveats). Anything in the post must stay inside what is measured.
- [ ] Open each image you will attach at full size and check labels are readable on a phone screen.

## The post (main version, about 230 words)

> "R-CNN is more accurate, YOLO is just faster." I kept hearing this, so I tested it.
>
> 20 object detectors. One MacBook. 128 labelled images. Here's what I found 👇
>
> 1️⃣ On average accuracy, Mask R-CNN ≈ YOLO11m (0.593 vs 0.589 mAP). But Mask R-CNN took ~700 ms per image vs ~31 ms. About 22x slower for the same score.
>
> 2️⃣ Where R-CNN really wins: finding things. It found 78% of all objects (YOLO11m: 50%) and 58% of the small ones (YOLO11m: under 20%).
>
> 3️⃣ Crowded scenes: on 3 images with ~40 objects each, Mask R-CNN produced 47 / 40 / 32 masks. The best YOLO segmentation models produced 3 to 21. (Count ≠ quality, I judged that by eye.)
>
> 4️⃣ The most accurate model wasn't either family: RT-DETR-L, a transformer (0.653).
>
> 5️⃣ YOLO11n-seg draws boxes AND masks in ~9 ms from a 6 MB file.
>
> So the question isn't "R-CNN or YOLO?" It's: do you need to find everything, the best average accuracy, or speed on small hardware?
>
> Honest limits: the test images came from COCO's training set (models had seen them), so compare the order, not exact scores. One laptop. RF-DETR and D-FINE: published numbers only, not run.
>
> I also built a video pipeline (frames → CNN → GRU with memory). On my 20-second clip it did NOT beat simple smoothing. A negative result, documented anyway.
>
> Code + a beginner-friendly learning guide in the first comment 👇
>
> What should I test next?
>
> #ComputerVision #ObjectDetection #YOLO #MachineLearning #DeepLearning

### Short version (about 90 words, if you want less text)
> Is R-CNN really "more accurate" than YOLO? I tested 20 detectors on a MacBook.
>
> • Average accuracy: Mask R-CNN ≈ YOLO11m (0.593 vs 0.589), but ~22x slower.
> • R-CNN's real edge: it finds more objects (78% vs 50%), especially small ones.
> • The most accurate model was a transformer (RT-DETR-L).
> • YOLO11n-seg: boxes + masks in ~9 ms, 6 MB.
>
> Caveat: images came from COCO's training set, so trust the order, not the exact numbers. Code and a learning guide in the comments.

## First comment (post this right after publishing)
> Code, charts, and a plain-English learning guide: <your repo link>
> How I tested it: 128 labelled COCO images on an Apple M5 MacBook, accuracy as mAP (a threshold-free score), speed as the median of repeated runs with one model loaded at a time. Full caveats are in the README (the test images were in the models' training data; mask quality judged by eye).

Why put the link in a comment? It is common advice that LinkedIn shows posts with outside links to fewer people. That is folklore I could not verify, so it is optional. Putting it in the comment is harmless either way.

## Which images to attach (in this order)
LinkedIn allows several images in one post (and "document" carousels, below). Files are in `detector_test/results/`.

| Order | File | Why |
| --- | --- | --- |
| 1 (lead image) | `chart_speed_vs_accuracy.png` | shows the whole story at a glance: R-CNN far right, YOLO far left, a transformer on top |
| 2 | `chart_recall.png` | backs point 2 (who finds the objects) |
| 3 | `post_crowded_strip.jpg` | proof people can see: input | Mask R-CNN | YOLO11m-seg | YOLO11n-seg on a crowded scene |

- **Caption for image 3 (use this wording):** "Crowded scene, 42 labelled objects. Mask R-CNN draws 40 masks (some overlapping duplicates), YOLO11m-seg 18, YOLO11n-seg 8." Do **not** write "R-CNN found 40 of 42", because "found" here includes overlapping boxes that are not all correct.
- **Alt text** (LinkedIn lets you add it, helps screen-reader users): "Scatter chart of 20 object detectors: accuracy (vertical) against time per image (horizontal, log scale). YOLO models cluster at the fast end, Mask R-CNN is far slower at similar accuracy, and RT-DETR-L has the highest accuracy."
- **Optional carousel:** export slides to a PDF (title; chart 1; chart 2; crowded strip; "what I'd choose for each job"; "limits and how to reproduce"; link). Carousels tend to be read slide by slide, which suits a result with several parts.

## How to post, step by step
1. On LinkedIn press **Start a post**, paste the main version, and check the emoji/numbered lines look right.
2. Press the **image icon**, add the three images in the order above, and add alt text to each (the pencil/Alt option).
3. Keep **three to five hashtags** at the end (already included).
4. **Preview, then Post.** Immediately add the **first comment** with the repo link.
5. **Reply to every comment in the first hours.** Questions like "did you try RF-DETR?" are good: answer honestly ("not run, only published numbers"). That honesty is what makes the post credible.
6. Timing: there is no reliable rule. Post when the people you want to reach are usually online (weekday working hours in their time zone is a safe default).
7. A **follow-up post** idea: "I tried to fix the video model and here is what happened" or "I ran RF-DETR and D-FINE too". Both are honest next steps listed in the README.

## What NOT to say (it would not survive a challenge)
- Do not say "R-CNN is more accurate" or "YOLO is less accurate" in general. Our result is: similar average accuracy at mid sizes, R-CNN better recall.
- Do not say any ranking of **RF-DETR, D-FINE, LW-DETR, YOLO-ULM**; they were not run.
- Do not quote **absolute** accuracy scores as if they were official benchmark scores (the test images were training data).
- Do not say Mask R-CNN's masks are "more accurate"; only mask counts were compared, plus a by-eye look at a few images.
- Do not say "an RNN is better/worse than YOLO". They do different jobs; an RNN is a sequence model, and "Faster RNN" is not a detector (it is Faster R-CNN).
- Do not post the cat-clip video or frames unless you intend to (it is your personal footage and is not in the repo).

## Questions you will probably get (and honest answers)
| Question | Answer |
| --- | --- |
| "Isn't testing on COCO training images cheating?" | Yes, it inflates all scores. That is why I only trust the order and why the README says so. The published val2017 numbers (in the README) are the official accuracy figures. |
| "One laptop is not a benchmark." | Correct. Speeds will differ on a GPU server or phone; the *relative* gaps (about 20x) are the useful part. |
| "Why not RF-DETR / D-FINE?" | Not run. Their published numbers are in the README table, clearly marked as published. A follow-up post would run them. |
| "R-CNN found more, but were they right?" | Partly: precision at 0.5 was lower (0.64 vs about 0.93) but that is not comparable across families; mAP is the fair measure, where they are close. Some of Mask R-CNN's crowded-scene masks are duplicates. |
| "Why does your RNN not help?" | One 20-second clip is tiny; simple smoothing already removes most flicker. The README lists what would be needed (more labelled clips). |
| "Which one should I use?" | By job: find everything -> R-CNN family; best average accuracy -> a strong transformer; speed or edge devices -> nano/small YOLO. |
