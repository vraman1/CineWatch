import pandas as pd
from PIL import Image
from transformers import pipeline
from common import OUT, need
events=pd.read_csv(need(OUT/"events"/"structured_events.csv"))
inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv")).set_index("incident_id")
captioner=pipeline("image-to-text",model="Salesforce/blip-image-captioning-base",device=-1)
rows=[]
for r in events.itertuples(index=False):
    # Caption the middle frame of this incident (not the first anomalous frame of the video).
    frames=[f for f in str(inc.loc[r.incident_id,"frame_paths"]).split("|") if f]
    caption="No caption available."
    if frames:
        try:
            image=Image.open(frames[len(frames)//2]).convert("RGB")
            caption=captioner(image,max_new_tokens=40)[0].get("generated_text",caption)
        except Exception as e: caption=f"Captioning failed: {e}"
    rows.append({"incident_id":r.incident_id,"video":r.video,"category":r.category,
                 "severity":r.severity,"visual_caption":caption,
                 "caution":"Caption describes visible content; it does not establish that a crime occurred."})
    print("Captioned",r.incident_id)
out=OUT/"summaries"/"incident_summaries.csv";pd.DataFrame(rows).to_csv(out,index=False)
print("Saved:",out)
