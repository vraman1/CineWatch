import cv2, subprocess, pandas as pd
import imageio_ffmpeg
from ultralytics import YOLO
from common import OUT, need
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
def to_h264(p):
    # OpenCV's mp4v is not browser-playable; re-encode to H.264 for the dashboard.
    tmp=p.with_suffix(".tmp.mp4")
    subprocess.run([FFMPEG,"-y","-loglevel","error","-i",str(p),"-c:v","libx264","-pix_fmt","yuv420p",
                    "-movflags","+faststart",str(tmp)],check=True)
    tmp.replace(p)
inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv"))
# Optional quick test: set MAX_INCIDENTS=2 before running. None processes all incidents.
MAX_INCIDENTS=None
if MAX_INCIDENTS is not None: inc=inc.head(MAX_INCIDENTS)
od=OUT/"visual_verification"/"original";ad=OUT/"visual_verification"/"annotated"
od.mkdir(parents=True,exist_ok=True);ad.mkdir(parents=True,exist_ok=True)
rows=[]
for r in inc.itertuples(index=False):
    cap=cv2.VideoCapture(r.video_path)
    if not cap.isOpened():print("Cannot open",r.video);continue
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start=max(0,float(r.start_seconds)-1);end=max(start+2,float(r.end_seconds)+1)
    cap.set(cv2.CAP_PROP_POS_MSEC,start*1000)
    op=od/f"{r.incident_id}.mp4";ap=ad/f"{r.incident_id}.mp4"
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    ow=cv2.VideoWriter(str(op),fourcc,fps,(w,h));aw=cv2.VideoWriter(str(ap),fourcc,fps,(w,h))
    model=YOLO("yolov8n.pt");n=0
    while cap.get(cv2.CAP_PROP_POS_MSEC)/1000<=end:
        ok,frame=cap.read()
        if not ok:break
        ow.write(frame)
        result=model.track(frame,persist=True,tracker="bytetrack.yaml",conf=.35,verbose=False)[0]
        annotated=result.plot()
        cv2.putText(annotated,f"CineWatch | {r.incident_id} | {r.category}",
                    (12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        aw.write(annotated);n+=1
    cap.release();ow.release();aw.release()
    if n: to_h264(op);to_h264(ap)
    rows.append({"incident_id":r.incident_id,"video":r.video,"category":r.category,
                 "start_seconds":start,"end_seconds":end,"frames_written":n,
                 "original_clip":str(op.resolve()),"annotated_clip":str(ap.resolve())})
    print("Verified",r.incident_id,"frames:",n)
out=OUT/"visual_verification"/"visual_verification_index.csv";pd.DataFrame(rows).to_csv(out,index=False)
print("Saved index:",out)
print("Dense per-frame processing is slow on CPU; run only when needed.")
