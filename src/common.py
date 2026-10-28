from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "videos"
FRAMES = ROOT / "frames"
OUT = ROOT / "outputs"
for p in [FRAMES, OUT/"metadata", OUT/"anomaly", OUT/"detections", OUT/"tracking",
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

def need(p):
    if not Path(p).exists():
        raise FileNotFoundError(f"{p} missing. Run the previous phase first.")
    return Path(p)
