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
python src/phase11_visual_verification.py   # key-moment clips (run before phase 10)
python src/phase10_summaries.py             # captions the key-moment frame
python src/phase12_evaluation.py
python src/phase13_clip_classifier.py   # learned CLIP classifier, 5-fold cross-validated
python src/export_models.py             # collects YOLO, CLIP, BLIP, classifier + calibration into models/
```

## Web app (React + Flask)
Requires Node.js 18+. Build the frontend once, then start the backend:
```powershell
cd frontend
npm install
npm run build
cd ..
python backend/app.py
```
Open http://127.0.0.1:5000 and upload a video. `src/analyze.py` detects incidents automatically with the saved models (motion scoring, CLIP classifier, YOLOv8s people and cars, BLIP caption) and shows the full video beside an annotated clip of each incident. It needs the CLIP and BLIP models in `models/` (`python src/export_models.py`).

`dataset/s_videos/` is for the 10 selected UCF-Crime videos listed in its README (videos are not in this repository; the full 100-video set for `dataset/videos/` can be downloaded with `python src/phase02_download.py`). The app recognises the 10 selected videos by content hash, so their selected incident also shows the reviewed description from `curation.json`; any other video gets the AI caption only. UCF-Crime videos are for research and academic use.

For frontend development with hot reload, keep `python backend/app.py` running and run `npm run dev` in `frontend/` (opens on port 5173 and proxies `/api` and `/media` to Flask).

Phase 11 can be very slow on CPU; do not rerun unnecessarily. Reuse the CSVs and clips saved from Colab where possible. `/content/...` paths are temporary Colab paths, not Windows paths.

## Curated incidents
`curation.json` lists the incidents shown first in the app (`featured`), a description of each written by a person after watching the clip (`reviewed`, shown above the BLIP caption), and manual clip windows (`clip_override`, used by phase 11) where the motion window missed part of the event. Unverified motion flags are listed last.

## Important limitations
This is a prototype. Anomaly detection uses motion/scene-change heuristics, YOLO detects objects rather than crimes, BLIP captions do not prove a crime occurred, and severity is a heuristic rather than a human-validated assessment. Results apply only to the selected videos and should not be described as general UCF-Crime accuracy.
