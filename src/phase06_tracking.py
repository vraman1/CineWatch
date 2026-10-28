import pandas as pd
from ultralytics import YOLO
from common import OUT, need, YOLO_WEIGHTS
inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv"));rows=[]
sc=pd.read_csv(need(OUT/"anomaly"/"anomaly_scores.csv"));ts=dict(zip(sc.frame_path,sc.timestamp_seconds))
for r in inc.itertuples(index=False):
    # Reinitialize tracker for each incident to avoid carrying track IDs between incidents.
    model=YOLO(YOLO_WEIGHTS)
    for fp in str(r.frame_paths).split("|"):
        if not fp:continue
        res=model.track(fp,persist=True,tracker="bytetrack.yaml",conf=.35,verbose=False)[0]
        if res.boxes is None:continue
        ids=res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None]*len(res.boxes)
        for i,b in enumerate(res.boxes):
            cid=int(b.cls.item());xy=b.xyxy[0].tolist()
            rows.append({"incident_id":r.incident_id,"video":r.video,"category":r.category,
                         "timestamp_seconds":ts.get(fp,r.start_seconds),"frame_path":fp,"track_id":ids[i],
                         "class_id":cid,"class_name":model.names.get(cid,str(cid)),
                         "confidence":float(b.conf.item()),"x1":xy[0],"y1":xy[1],"x2":xy[2],"y2":xy[3]})
    print("Tracked",r.incident_id)
out=OUT/"tracking"/"tracking_records.csv";pd.DataFrame(rows).to_csv(out,index=False)
print("Tracking records:",len(rows),"saved:",out)
print("Track IDs are local to an incident, not real-world identities.")
