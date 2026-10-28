import pandas as pd
from PIL import Image
from transformers import pipeline
from common import OUT, MODELS, need
events=pd.read_csv(need(OUT/"events"/"structured_events.csv"))
inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv")).set_index("incident_id")
sc=pd.read_csv(need(OUT/"anomaly"/"anomaly_scores.csv"))
det=pd.read_csv(need(OUT/"detections"/"yolo_detections.csv"))
# Key moments (and edit-only flags) come from phase 11, so run phase 11 before this phase.
vv=pd.read_csv(need(OUT/"visual_verification"/"visual_verification_index.csv")).set_index("incident_id")
captioner=pipeline("image-to-text",model=str(MODELS/"blip") if (MODELS/"blip"/"config.json").exists() else "Salesforce/blip-image-captioning-base",device=-1)
rows=[]
for r in events.itertuples(index=False):
    # Caption the key-moment frame shown in the annotated clip; fall back to the incident's middle frame.
    frame=None
    if r.incident_id in vv.index:
        k=sc[(sc.video==r.video)&(sc.timestamp_seconds==vv.loc[r.incident_id,"peak_seconds"])]
        if len(k): frame=k.frame_path.iloc[0]
    if frame is None:
        frames=[f for f in str(inc.loc[r.incident_id,"frame_paths"]).split("|") if f]
        frame=frames[len(frames)//2] if frames else None
    caption="No caption available."
    if frame:
        try: caption=captioner(Image.open(frame).convert("RGB"),max_new_tokens=40)[0].get("generated_text",caption)
        except Exception as e: caption=f"Captioning failed: {e}"
    # Factual context: YOLO objects detected in that same frame (phase 5).
    d=det[det.frame_path==frame] if frame else det.iloc[0:0]
    objs=", ".join(f"{n} {c}" for c,n in d.class_name.value_counts().items()) or "none"
    edit=bool(vv.loc[r.incident_id,"edit_only"]) if r.incident_id in vv.index and "edit_only" in vv.columns else False
    rows.append({"incident_id":r.incident_id,"video":r.video,"category":r.category,
                 "severity":r.severity,"visual_caption":caption,"key_frame":frame,"key_objects":objs,"edit_only":edit,
                 "caution":"Caption describes visible content; it does not establish that a crime occurred."})
    print("Captioned",r.incident_id)
out=OUT/"summaries"/"incident_summaries.csv";pd.DataFrame(rows).to_csv(out,index=False)
print("Saved:",out)
