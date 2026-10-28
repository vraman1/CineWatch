import json, urllib.request, pandas as pd
from common import DATA, OUT
# Original UCF-Crime MP4s mirrored on Hugging Face. AllenXeon has all 13 crime classes;
# mteb has the same files plus Testing_Normal videos. Smallest files = shortest clips (all 320x240).
ANOMALY_REPO="AllenXeon/ucf_crime"; NORMAL_REPO="mteb/ucf-crime"
PER_CLASS=4; NORMALS=48   # 13 classes x 4 + 48 normal = 100 videos (~balanced)

def tree(repo):
    url=f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url) as r: items=json.load(r)
    return [{"repo":repo,"path":x["path"],"size":x["size"],"category":x["path"].split("/")[-2],
             "video":x["path"].split("/")[-1]} for x in items
            if x["type"]=="file" and x["path"].endswith(".mp4")]

a=pd.DataFrame(tree(ANOMALY_REPO)); a=a[~a.category.str.contains("Normal")]
n=pd.DataFrame(tree(NORMAL_REPO)); n=n[n.category=="Testing_Normal"]
pick=pd.concat([a.sort_values("size").groupby("category").head(PER_CLASS),
                n.sort_values("size").head(NORMALS)]).sort_values(["category","video"])
print("Selected:",len(pick),"videos,",round(pick["size"].sum()/1e6,1),"MB")
for r in pick.itertuples(index=False):
    dest=DATA/r.category/r.video; dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==r.size: continue
    url=f"https://huggingface.co/datasets/{r.repo}/resolve/main/{r.path}"
    urllib.request.urlretrieve(url,dest); print("Downloaded",r.category,r.video,round(r.size/1e6,1),"MB")
out=OUT/"metadata"/"download_manifest.csv"; pick.to_csv(out,index=False)
print("Saved:",out)
