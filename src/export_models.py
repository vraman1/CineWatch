import json, shutil, joblib, pandas as pd
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor
from common import ROOT, OUT, MODELS, need
# Collects every model the project uses into models/ so the app runs offline from one folder.
yolo=ROOT/"yolov8n.pt"
if yolo.exists() and not (MODELS/"yolov8n.pt").exists(): shutil.move(str(yolo),MODELS/"yolov8n.pt")
if not (MODELS/"yolov8n.pt").exists():
    from ultralytics import YOLO; YOLO("yolov8n.pt"); shutil.move("yolov8n.pt",MODELS/"yolov8n.pt")
print("YOLO:",MODELS/"yolov8n.pt")
for name,cls,proc,mid in [("clip",CLIPModel,CLIPProcessor,"openai/clip-vit-base-patch32"),
                          ("blip",BlipForConditionalGeneration,BlipProcessor,"Salesforce/blip-image-captioning-base")]:
    d=MODELS/name
    if not (d/"config.json").exists():
        cls.from_pretrained(mid).save_pretrained(d); proc.from_pretrained(mid).save_pretrained(d)
    print(name.upper()+":",d)
shutil.copy(need(OUT/"evaluation"/"clip_classifier.joblib"),MODELS/"clip_classifier.joblib")
print("Classifier:",MODELS/"clip_classifier.joblib")
# Phase 4 motion calibration (median + robust sigma of normal-video frame changes), needed to score new videos.
d=pd.read_csv(need(OUT/"anomaly"/"anomaly_scores.csv")); ref=d[d.label=="normal"].change
med=float(ref.median()); sigma=max(float((ref-med).abs().median())*1.4826,1e-6)
(MODELS/"calibration.json").write_text(json.dumps({"median":med,"sigma":sigma,"z_threshold":3.5,
    "sample_every_seconds":2},indent=2))
print("Calibration:",MODELS/"calibration.json")
