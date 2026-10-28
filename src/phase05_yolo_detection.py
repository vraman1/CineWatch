import pandas as pd
from ultralytics import YOLO
from common import OUT, need
inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv")); model=YOLO("yolov8n.pt"); rows=[]
sc=pd.read_csv(need(OUT/"anomaly"/"anomaly_scores.csv")); ts=dict(zip(sc.frame_path,sc.timestamp_seconds))
for r in inc.itertuples(index=False):
    for fp in str(r.frame_paths).split("|"):
        if not fp: continue
        res=model.predict(fp,conf=.35,verbose=False)[0]
        for b in res.boxes:
            cid=int(b.cls.item()); xy=b.xyxy[0].tolist()
            rows.append({"incident_id":r.incident_id,"video":r.video,"category":r.category,
                         "timestamp_seconds":ts.get(fp,r.start_seconds),"frame_path":fp,"class_id":cid,
                         "class_name":model.names.get(cid,str(cid)),"confidence":float(b.conf.item()),
                         "x1":xy[0],"y1":xy[1],"x2":xy[2],"y2":xy[3]})
    print("Detected",r.incident_id)
out=OUT/"detections"/"yolo_detections.csv";pd.DataFrame(rows).to_csv(out,index=False)
print("Detections:",len(rows),"saved:",out)
