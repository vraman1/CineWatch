import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from common import OUT, need
# Learned video-level classifier: frozen CLIP image features + logistic regression.
# Evaluated with stratified 5-fold cross-validation, so every video is scored by a model that never saw it.
m=pd.read_csv(need(OUT/"metadata"/"frame_metadata.csv"))
sc=pd.read_csv(need(OUT/"anomaly"/"anomaly_scores.csv"))[["frame_path","z_score"]]
m=m.merge(sc,on="frame_path",how="left")
feat_path=OUT/"evaluation"/"clip_frame_features.npy"
model_id="openai/clip-vit-base-patch32"
if feat_path.exists():
    F=np.load(feat_path)
else:
    model=CLIPModel.from_pretrained(model_id).eval(); proc=CLIPProcessor.from_pretrained(model_id); F=[]
    with torch.no_grad():
        for i in range(0,len(m),32):
            ims=[Image.open(p).convert("RGB") for p in m.frame_path[i:i+32]]
            e=model.get_image_features(**proc(images=ims,return_tensors="pt"))
            F.append(torch.nn.functional.normalize(e,dim=-1).numpy()); print("Embedded",min(i+32,len(m)),"/",len(m))
    F=np.concatenate(F); np.save(feat_path,F)
X=[];y=[];vids=[]
for v,g in m.groupby("video",sort=False):
    e=F[g.index]; z=g.z_score.fillna(0).to_numpy()
    # Per-video descriptor: mean + max CLIP embedding, plus motion statistics from phase 4.
    X.append(np.concatenate([e.mean(0),e.max(0),[z.mean(),z.max(),(z>=3.5).mean()]]))
    y.append(int(g.label.iloc[0]!="normal")); vids.append(v)
X=np.array(X);y=np.array(y)
clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=5000))
accs=[];last=None
for seed in range(5):
    p=cross_val_predict(clf,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=seed))
    accs.append(accuracy_score(y,p)); last=p if seed==0 else last
tn,fp,fn,tp=confusion_matrix(y,last,labels=[0,1]).ravel()
res={"method":"CLIP ViT-B/32 + logistic regression (5-fold CV)","videos":len(y),
     "TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn),"accuracy":accuracy_score(y,last),
     "precision":precision_score(y,last,zero_division=0),"recall":recall_score(y,last,zero_division=0),
     "f1":f1_score(y,last,zero_division=0),"specificity":tn/(tn+fp) if tn+fp else 0,
     "accuracy_mean_5_seeds":float(np.mean(accs)),"accuracy_std_5_seeds":float(np.std(accs))}
pd.DataFrame([res]).to_csv(OUT/"evaluation"/"clip_classifier_metrics.csv",index=False)
pd.DataFrame({"video":vids,"y_true":y,"y_pred":last}).to_csv(OUT/"evaluation"/"clip_classifier_predictions.csv",index=False)
print(pd.DataFrame([res]).T.to_string(header=False))
# Final model: fit on all videos for later use. The metrics above come from cross-validation, not this fit.
import joblib
clf.fit(X,y)
joblib.dump({"model":clf,"clip_model":model_id,"features":"mean+max CLIP embedding + [z mean, z max, flagged fraction]"},
            OUT/"evaluation"/"clip_classifier.joblib")
print("Saved model:",OUT/"evaluation"/"clip_classifier.joblib")
print("Cross-validated on this 100-video subset only; not full-dataset performance.")
