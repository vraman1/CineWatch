from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "videos"
FRAMES = ROOT / "frames"  # created by phase 3 only; the web app does not need it
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
# YOLOv8s (small) is more accurate than v8n (nano); fall back to nano if the small weights are missing.
YOLO_WEIGHTS = next((str(MODELS/w) for w in ("yolov8s.pt", "yolov8n.pt") if (MODELS/w).exists()), "yolov8s.pt")
# Annotated clips show only people and cars. COCO calls vans and pickups "truck", so those are labelled car too.
TRACK_CLASSES = {0: "person", 2: "car", 7: "car"}
TRACK_CONF = .30
# Run YOLO on every Nth frame of an annotated clip and keep its boxes on the frames between (~3x faster on CPU).
DETECT_EVERY = 3
for p in [MODELS, OUT/"metadata", OUT/"anomaly", OUT/"detections", OUT/"tracking",
          OUT/"events", OUT/"summaries", OUT/"visual_verification", OUT/"evaluation"]:
    p.mkdir(parents=True, exist_ok=True)
EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

def inventory():
    rows = []
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            cat = p.parent.name
            rows.append({"video": p.name, "video_path": str(p.resolve()),
                         "category": cat,
                         "label": "normal" if "normal" in cat.lower() else "abnormal"})
    return pd.DataFrame(rows, columns=["video", "video_path", "category", "label"])

def is_blank(frame_path):
    import cv2
    return is_blank_img(cv2.imread(str(frame_path)))

def is_blank_img(im):
    # True for black, white or title-card frames (logos, fades) that UCF-Crime's edited videos contain.
    import cv2, numpy as np
    if im is None: return True
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); h, w = g.shape
    c = g[int(h*.15):int(h*.85), int(w*.1):int(w*.9)]  # ignore letterbox bars and corner watermarks
    mode = np.bincount((c//16).ravel(), minlength=16).max()/c.size
    return bool(c.std() < 18 or mode > .6 or (c.mean() < 40 and c.std() < 35))

def need(p):
    if not Path(p).exists():
        raise FileNotFoundError(f"{p} missing. Run the previous phase first.")
    return Path(p)
