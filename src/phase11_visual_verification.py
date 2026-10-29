import cv2, json, subprocess, pandas as pd
from multiprocessing import Pool
from common import ROOT, OUT, need, YOLO_WEIGHTS, is_blank
# Optional quick test: set MAX_INCIDENTS=2. None processes all incidents.
MAX_INCIDENTS=None
WORKERS=4          # incidents processed in parallel
THREADS=3          # torch threads per worker
STEP=2             # frame sampling interval of phase 3, in seconds
SCALE=2            # annotated clips are drawn at 2x so labels stay small and don't cover faces
AD=OUT/"visual_verification"/"annotated"

def to_h264(p):
    # OpenCV's mp4v is not browser-playable; re-encode to H.264 for the web app.
    import imageio_ffmpeg
    tmp=p.with_suffix(".tmp.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-i",str(p),"-c:v","libx264",
                    "-preset","veryfast","-pix_fmt","yuv420p","-movflags","+faststart",str(tmp)],check=True)
    tmp.replace(p)

def incident_window(sc,r):
    # The clip is the incident alone. A flagged sample's z measures change since the previous sample (STEP s earlier),
    # so the incident spans from STEP s before its first flagged sample to its last one.
    # Samples that are, or follow, a blank/title frame are video edits (cuts to logos, fades) and are trimmed off.
    # Returns (clip start, clip end, peak sample time, edit_only).
    s=sc[(sc.video==r.video)&sc.anomaly&sc.timestamp_seconds.between(r.start_seconds,r.end_seconds)]
    real=s[~s.edit]; edit_only=real.empty; s=s if edit_only else real
    if s.empty: return max(0,float(r.start_seconds)-STEP),float(r.end_seconds),float(r.start_seconds),True
    peak=float(s.loc[s.z_score.idxmax(),"timestamp_seconds"])
    start,end=max(0,float(s.timestamp_seconds.min())-STEP),float(s.timestamp_seconds.max())
    if r.incident_id in OVERRIDE: start,end=map(float,OVERRIDE[r.incident_id])  # manual fix from curation.json
    return start,end,peak,edit_only

# Clip windows corrected by hand in curation.json, where the motion window missed part of the event.
OVERRIDE=json.loads((ROOT/"curation.json").read_text(encoding="utf-8")).get("clip_override",{}) if (ROOT/"curation.json").exists() else {}

def draw(frame,res,names,title):
    # YOLO runs on the native frame (fast); boxes are drawn on the upscaled frame with thin lines and small labels.
    from ultralytics.utils.plotting import colors
    big=cv2.resize(frame,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_CUBIC)
    b=res.boxes
    if b is not None and len(b):
        ids=b.id.int().tolist() if b.id is not None else [None]*len(b)
        for (x1,y1,x2,y2),c,i in zip((b.xyxy*SCALE).int().tolist(),b.cls.int().tolist(),ids):
            col=colors(c,True); label=f"{'' if i is None else f'id:{i} '}{names[c]}"
            cv2.rectangle(big,(x1,y1),(x2,y2),col,1)
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.4,1)
            cv2.rectangle(big,(x1,max(0,y1-th-4)),(x1+tw+4,y1),col,-1)
            ink=(0,0,0) if .114*col[0]+.587*col[1]+.299*col[2]>150 else (255,255,255)  # readable on light colours
            cv2.putText(big,label,(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,.4,ink,1,cv2.LINE_AA)
    cv2.putText(big,title,(8,big.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return big

def work(job):
    import torch; torch.set_num_threads(THREADS)
    from ultralytics import YOLO
    r,start,end,peak,edit_only=job
    cap=cv2.VideoCapture(r["video_path"])
    if not cap.isOpened(): return None
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC,start*1000)
    ap=AD/f"{r['incident_id']}.mp4"
    aw=cv2.VideoWriter(str(ap),cv2.VideoWriter_fourcc(*"mp4v"),fps,(SCALE*w,SCALE*h))
    model=YOLO(YOLO_WEIGHTS);n=0  # fresh tracker per incident
    title=f"CineWatch | {r['incident_id']} | {r['category']}"
    while cap.get(cv2.CAP_PROP_POS_MSEC)/1000<=end:
        ok,frame=cap.read()
        if not ok:break
        res=model.track(frame,persist=True,tracker="bytetrack.yaml",conf=.35,imgsz=320,verbose=False)[0]
        aw.write(draw(frame,res,model.names,title));n+=1
    cap.release();aw.release()
    if n: to_h264(ap)
    print("Verified",r["incident_id"],"frames:",n,flush=True)
    return {"incident_id":r["incident_id"],"video":r["video"],"category":r["category"],
            "peak_seconds":peak,"edit_only":edit_only,"start_seconds":start,"end_seconds":end,
            "frames_written":n,"annotated_clip":str(ap.resolve())}

if __name__=="__main__":
    inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv"))
    if MAX_INCIDENTS is not None: inc=inc.head(MAX_INCIDENTS)
    sc=pd.read_csv(need(OUT/"anomaly"/"anomaly_scores.csv")).sort_values(["video","timestamp_seconds"])
    sc["blank"]=[is_blank(p) for p in sc.frame_path]
    sc["edit"]=sc.blank|sc.groupby("video").blank.shift(1,fill_value=False)
    AD.mkdir(parents=True,exist_ok=True)
    jobs=[(r,*incident_window(sc,pd.Series(r))) for r in inc.to_dict("records")]
    jobs.sort(key=lambda j:-(j[2]-j[1]))  # longest clips first keeps the workers evenly loaded
    with Pool(WORKERS) as pool: rows=[x for x in pool.imap_unordered(work,jobs) if x]
    out=OUT/"visual_verification"/"visual_verification_index.csv"
    pd.DataFrame(rows).sort_values("incident_id").to_csv(out,index=False)
    print("Saved index:",out)
