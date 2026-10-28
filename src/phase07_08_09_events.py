import pandas as pd, json
from common import OUT, need
inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv"))
dp=OUT/"detections"/"yolo_detections.csv";tp=OUT/"tracking"/"tracking_records.csv"
det=pd.read_csv(dp) if dp.exists() else pd.DataFrame()
trk=pd.read_csv(tp) if tp.exists() else pd.DataFrame()
events=[]
for r in inc.itertuples(index=False):
    d=det[det.incident_id==r.incident_id] if not det.empty else pd.DataFrame()
    t=trk[trk.incident_id==r.incident_id] if not trk.empty else pd.DataFrame()
    classes=sorted(set(d.class_name.astype(str))) if not d.empty else []
    tracked=sorted(set(t.class_name.astype(str))) if not t.empty else []
    conf=float(r.mean_anomaly_confidence); duration=float(r.duration_seconds)
    score=min(1,.60*conf+.15*min(duration/30,1)+.10*min(len(d)/5,1)+
              .05*min(r.sample_count/10,1)+.05*min(len(classes)/3,1)+.05*(1 if len(t) else 0))
    severity="High" if score>=.65 else ("Medium" if score>=.45 else "Low")
    events.append({"incident_id":r.incident_id,"video":r.video,"category":r.category,
                   "reference_label":r.label,"start_seconds":r.start_seconds,"end_seconds":r.end_seconds,
                   "duration_seconds":duration,"sample_count":int(r.sample_count),
                   "anomaly_confidence":conf,"object_count":len(d),"object_classes":", ".join(classes),
                   "tracking_records":len(t),"tracked_classes":", ".join(tracked),
                   "severity_score":score,"severity":severity,
                   "event_context":"Candidate motion/scene-change interval; human review required."})
df=pd.DataFrame(events)
df.to_csv(OUT/"events"/"structured_events.csv",index=False)
(OUT/"events"/"structured_events.json").write_text(json.dumps(events,indent=2),encoding="utf-8")
print("Structured events and heuristic severity saved.")
