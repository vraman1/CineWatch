# CineWatch — VS Code starter

This package recreates a modular local workflow for the CineWatch prototype. It does not include UCF-Crime videos.

## Dataset size from prior work
- Earlier Colab selection: 100 videos, approximately 4.39 GB (about 4,390 MB).
- Latest development selection: 20 videos; total size was not recorded.
- These are Colab download figures, not confirmation of files currently stored on Windows.

## Setup (Windows PowerShell)
Install Python 3.11, then:
```powershell
cd path\to\CineWatch_VSCode
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy your videos into `dataset/videos/<Category>/`, e.g. `dataset/videos/Abuse/clip.mp4` and `dataset/videos/Training_Normal/clip.mp4`.

Run phases in order:
```powershell
python src/phase02_download.py   # optional: fetches 100 short original UCF-Crime videos (~420 MB)
python src/phase01_inventory.py
python src/phase03_sample_frames.py
python src/phase04_anomaly_detection.py
python src/phase05_yolo_detection.py
python src/phase06_tracking.py
python src/phase07_08_09_events.py
python src/phase10_summaries.py
python src/phase11_visual_verification.py
python src/phase12_evaluation.py
python src/phase13_clip_classifier.py   # learned CLIP classifier, 5-fold cross-validated
python dashboard/app.py
```

Phase 11 can be very slow on CPU; do not rerun unnecessarily. Reuse the CSVs and clips saved from Colab where possible. `/content/...` paths are temporary Colab paths, not Windows paths.

## Important limitations
This is a prototype. Anomaly detection uses motion/scene-change heuristics, YOLO detects objects rather than crimes, BLIP captions do not prove a crime occurred, and severity is a heuristic rather than a human-validated assessment. Results apply only to the selected videos and should not be described as general UCF-Crime accuracy.
