import cv2, pandas as pd
from common import inventory, FRAMES, OUT

videos = inventory()
if videos.empty: raise SystemExit("No videos found. Add videos to dataset/videos/<Category>/")
rows = []
for v in videos.itertuples(index=False):
    cap = cv2.VideoCapture(v.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release(); print("Skipping unreadable:", v.video); continue
    step = max(1, round(2*fps))
    folder = FRAMES / (v.video.rsplit(".",1)[0].replace(" ","_"))
    folder.mkdir(parents=True, exist_ok=True)
    idx = n = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: break
        ts = idx/fps
        fp = folder/f"frame_{n:05d}.jpg"
        cv2.imwrite(str(fp), frame)
        rows.append({"video":v.video,"video_path":v.video_path,"category":v.category,
                     "label":v.label,"sample_number":n,"original_frame":idx,
                     "timestamp_seconds":ts,"fps":fps,"frame_path":str(fp.resolve())})
        idx += step; n += 1
    cap.release()
    print(v.video, "sampled:", n)
out = OUT/"metadata"/"frame_metadata.csv"
pd.DataFrame(rows).to_csv(out,index=False)
print("Total frames:",len(rows),"saved:",out)
