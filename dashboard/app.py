from pathlib import Path
import pandas as pd
import gradio as gr
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs"
def load(p): return pd.read_csv(p) if p.exists() else pd.DataFrame()
events=load(OUT/"events"/"structured_events.csv")
summaries=load(OUT/"summaries"/"incident_summaries.csv")
clips=load(OUT/"visual_verification"/"visual_verification_index.csv")
ids=events.incident_id.astype(str).tolist() if not events.empty else []
def show(i):
    if events.empty or i is None:return "Run the pipeline first.",None,None,"No event data.",""
    x=events[events.incident_id.astype(str)==str(i)]
    if x.empty:return "Incident not found.",None,None,"No event data.",""
    e=x.iloc[0]; orig=ann=None
    if not clips.empty:
        c=clips[clips.incident_id.astype(str)==str(i)]
        if not c.empty:
            orig=c.iloc[0].get("original_clip");ann=c.iloc[0].get("annotated_clip")
            orig=orig if isinstance(orig,str) and Path(orig).exists() else None
            ann=ann if isinstance(ann,str) and Path(ann).exists() else None
    context="No visual caption yet."
    if not summaries.empty:
        s=summaries[summaries.incident_id.astype(str)==str(i)]
        if not s.empty:context=str(s.iloc[0].get("visual_caption",""))+"\n\n"+str(s.iloc[0].get("caution",""))
    details="\n".join(f"{k}: {e.get(k,'')}" for k in
      ["incident_id","video","category","severity","severity_score","duration_seconds","object_classes","tracked_classes"])
    temporal=f"Start: {e.get('start_seconds')} s\nEnd: {e.get('end_seconds')} s\nSamples: {e.get('sample_count')}\nAnomaly confidence: {e.get('anomaly_confidence')}"
    return details,orig,ann,context,temporal
with gr.Blocks(title="CineWatch") as demo:
    gr.Markdown("# CineWatch — Surveillance Video Analytics")
    gr.Markdown("Candidate incidents require human verification. Category is dataset metadata; severity is heuristic.")
    sel=gr.Dropdown(ids,value=ids[0] if ids else None,label="Select incident")
    with gr.Row():
        orig=gr.Video(label="Original incident clip",height=300)
        ann=gr.Video(label="Annotated incident clip",height=300)
    ctx=gr.Textbox(label="Event Context",lines=4)
    with gr.Row():
        details=gr.Textbox(label="Incident Details",lines=8)
        temporal=gr.Textbox(label="Temporal Progression",lines=8)
    sel.change(show,inputs=sel,outputs=[details,orig,ann,ctx,temporal])
    if ids: demo.load(show,inputs=sel,outputs=[details,orig,ann,ctx,temporal])
    with gr.Accordion("All detected incidents",open=False):
        if not events.empty:gr.Dataframe(events,interactive=False)
if __name__=="__main__":demo.launch(allowed_paths=[str(OUT)])
