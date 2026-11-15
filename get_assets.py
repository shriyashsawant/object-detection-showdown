"""Download everything the experiments need (about 1.1 GB). Run once from the repo root:   python get_assets.py

 - Ultralytics weights (YOLO11 / YOLO12 / YOLO26 / RT-DETR, detection and segmentation)  -> detector_test/weights/
 - torchvision Faster R-CNN and Mask R-CNN weights                                         -> torch's own cache (~/.cache/torch)
 - the coco128 dataset (128 labelled COCO images, 7 MB)                                    -> detector_test/data/coco128/
 - the 3 most crowded coco128 images and two sample images                                 -> detector_test/images/
Existing files are skipped. Set ASSETS_DIR to use a different detector_test folder.
Licences: Ultralytics weights are AGPL-3.0; COCO images keep their own licences (see README).
"""
import os, shutil, urllib.request, zipfile

BASE = os.environ.get("ASSETS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_test"))
WEIGHTS, DATA, IMAGES = BASE + "/weights", BASE + "/data", BASE + "/images"
NAMES = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo12n", "yolo12m", "yolo26n", "yolo26m", "yolo26l", "rtdetr-l",
         "yolo11n-seg", "yolo11m-seg", "yolo11l-seg", "yolo26n-seg", "yolo26m-seg", "yolo26l-seg"]
CROWDED = ["000000000196.jpg", "000000000164.jpg", "000000000315.jpg"]          # the 3 most crowded images (40, 42 and 38 labelled objects)
SAMPLES = {"bus.jpg": "https://ultralytics.com/images/bus.jpg", "zidane.jpg": "https://ultralytics.com/images/zidane.jpg"}


def main():
    for d in (WEIGHTS, DATA, IMAGES):
        os.makedirs(d, exist_ok=True)
    from ultralytics import RTDETR, YOLO
    here = os.getcwd()
    os.chdir(WEIGHTS)                                        # ultralytics saves a downloaded weight into the current folder
    for n in NAMES:
        if not os.path.exists(n + ".pt"):
            print("downloading", n)
            (RTDETR if n.startswith("rtdetr") else YOLO)(n + ".pt")
    os.chdir(here)

    from torchvision.models import detection as D
    for builder, weights in [(D.fasterrcnn_mobilenet_v3_large_fpn, D.FasterRCNN_MobileNet_V3_Large_FPN_Weights),
                             (D.fasterrcnn_resnet50_fpn_v2, D.FasterRCNN_ResNet50_FPN_V2_Weights),
                             (D.maskrcnn_resnet50_fpn_v2, D.MaskRCNN_ResNet50_FPN_V2_Weights)]:
        builder(weights=weights.DEFAULT)                     # downloads into torch's cache if missing

    if not os.path.isdir(DATA + "/coco128"):
        print("downloading coco128")
        urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", DATA + "/coco128.zip")
        with zipfile.ZipFile(DATA + "/coco128.zip") as z:
            z.extractall(DATA)
    for f in CROWDED:
        if not os.path.exists(IMAGES + "/" + f):
            shutil.copy(DATA + "/coco128/images/train2017/" + f, IMAGES + "/" + f)
    for f, url in SAMPLES.items():
        if not os.path.exists(IMAGES + "/" + f):
            urllib.request.urlretrieve(url, IMAGES + "/" + f)
    total = sum(os.path.getsize(os.path.join(WEIGHTS, f)) for f in os.listdir(WEIGHTS)) / 1e6
    print(f"ready: {len(os.listdir(WEIGHTS))} weight files ({total:.0f} MB), images: {sorted(os.listdir(IMAGES))}")


if __name__ == "__main__":
    main()
