import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from common import OUT, need
inv=pd.read_csv(need(OUT/"metadata"/"video_inventory.csv"))
inc=pd.read_csv(need(OUT/"anomaly"/"incidents.csv"))
pred=set(inc.video.astype(str))
inv["y_true"]=(inv.label.astype(str)!="normal").astype(int)
inv["y_pred"]=inv.video.astype(str).isin(pred).astype(int)
tn,fp,fn,tp=confusion_matrix(inv.y_true,inv.y_pred,labels=[0,1]).ravel()
m={"videos":len(inv),"incidents":len(inc),"TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn),
   "accuracy":accuracy_score(inv.y_true,inv.y_pred),
   "precision":precision_score(inv.y_true,inv.y_pred,zero_division=0),
   "recall":recall_score(inv.y_true,inv.y_pred,zero_division=0),
   "f1":f1_score(inv.y_true,inv.y_pred,zero_division=0),
   "specificity":tn/(tn+fp) if tn+fp else 0}
pd.DataFrame([m]).to_csv(OUT/"evaluation"/"video_level_metrics.csv",index=False)
inv.to_csv(OUT/"evaluation"/"video_level_predictions.csv",index=False)
print(pd.DataFrame([m]).to_string(index=False))
print("Subset-specific evaluation; not full-dataset performance.")
