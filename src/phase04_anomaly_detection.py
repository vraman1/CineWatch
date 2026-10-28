import cv2, numpy as np, pandas as pd
from common import OUT, need

m = pd.read_csv(need(OUT/"metadata"/"frame_metadata.csv"))
rows=[]
for video,g in m.groupby("video",sort=False):
    prev=None
    for r in g.sort_values("timestamp_seconds").itertuples(index=False):
        im=cv2.imread(r.frame_path)
        if im is None: continue
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,(32,24)); gray=cv2.GaussianBlur(gray,(5,5),0)
        change=0.0 if prev is None else float(np.mean(cv2.absdiff(gray,prev)))
        rows.append({"video":r.video,"video_path":r.video_path,"category":r.category,
                     "label":r.label,"timestamp_seconds":r.timestamp_seconds,
                     "sample_number":r.sample_number,"frame_path":r.frame_path,"change":change})
        prev=gray
d=pd.DataFrame(rows)
ref=d[d.label=="normal"].change
if ref.empty: ref=d.change; print("WARNING: no normal videos; calibration uses all frames.")
med=float(ref.median()); sigma=max(float((ref-med).abs().median())*1.4826,1e-6)
d["z_score"]=(d.change-med)/sigma
d["anomaly"]=d.z_score>=3.5
d["anomaly_confidence"]=np.clip((d.z_score-2)/6,0,1)
d.to_csv(OUT/"anomaly"/"anomaly_scores.csv",index=False)
inc=[]
for video,g in d[d.anomaly].groupby("video",sort=False):
    g=g.sort_values("timestamp_seconds")
    chunks=[]; cur=[]; last=None
    for r in g.itertuples(index=False):
        if last is not None and r.timestamp_seconds-last>4:
            chunks.append(cur);cur=[]
        cur.append(r);last=r.timestamp_seconds
    if cur:chunks.append(cur)
    for c in chunks:
        a,b=c[0],c[-1]
        inc.append({"incident_id":f"INC_{len(inc)+1:04d}","video":video,
                    "video_path":a.video_path,"category":a.category,"label":a.label,
                    "start_seconds":float(a.timestamp_seconds),"end_seconds":float(b.timestamp_seconds),
                    "duration_seconds":float(b.timestamp_seconds-a.timestamp_seconds),
                    "sample_count":len(c),"mean_anomaly_confidence":float(np.mean([x.anomaly_confidence for x in c])),
                    "frame_paths":"|".join(x.frame_path for x in c)})
out=OUT/"anomaly"/"incidents.csv";pd.DataFrame(inc).to_csv(out,index=False)
print("Scored frames:",len(d),"incidents:",len(inc),"saved:",out)
print("Motion-change heuristic only; not a trained crime classifier.")
