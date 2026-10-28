from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "videos"
FRAMES = ROOT / "frames"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
YOLO_WEIGHTS = str(MODELS/"yolov8n.pt") if (MODELS/"yolov8n.pt").exists() else "yolov8n.pt"
for p in [FRAMES, MODELS, OUT/"metadata", OUT/"anomaly", OUT/"detections", OUT/"tracking",
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
    # True for black, white or title-card frames (logos, fades) that UCF-Crime's edited videos contain.
    import cv2, numpy as np
    im = cv2.imread(str(frame_path))
    if im is None: return True
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); h, w = g.shape
    c = g[int(h*.15):int(h*.85), int(w*.1):int(w*.9)]  # ignore letterbox bars and corner watermarks
    mode = np.bincount((c//16).ravel(), minlength=16).max()/c.size
    return bool(c.std() < 18 or mode > .6 or (c.mean() < 40 and c.std() < 35))

def need(p):
    if not Path(p).exists():
        raise FileNotFoundError(f"{p} missing. Run the previous phase first.")
    return Path(p)
