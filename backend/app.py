import json, math, threading
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory, abort

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"; DIST = ROOT/"frontend"/"dist"
app = Flask(__name__, static_folder=None)

def load(p):
    p = OUT/p
    try: return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError: return pd.DataFrame()

def clean(v):
    # JSON-safe scalars (pandas NaN -> None, numpy -> python).
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return None if math.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.bool_,)): return bool(v)
    return v

def records(df, cols=None):
    df = df if cols is None else df[[c for c in cols if c in df.columns]]
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]

# ---------- pipeline results ----------
@app.get("/api/summary")
def summary():
    inv, inc, ev = load("metadata/video_inventory.csv"), load("anomaly/incidents.csv"), load("events/structured_events.csv")
    base, clip = load("evaluation/video_level_metrics.csv"), load("evaluation/clip_classifier_metrics.csv")
    det = load("detections/yolo_detections.csv")
    return jsonify({
        "videos": len(inv), "crime_videos": int((inv.label != "normal").sum()) if len(inv) else 0,
        "normal_videos": int((inv.label == "normal").sum()) if len(inv) else 0,
        "frames": len(load("metadata/frame_metadata.csv")), "incidents": len(inc), "detections": len(det),
        "categories": records(inv.groupby("category").size().rename("videos").reset_index()) if len(inv) else [],
        "severity": records(pd.crosstab(ev.reference_label, ev.severity).reset_index()) if len(ev) else [],
        "top_objects": records(det.class_name.value_counts().head(8).rename_axis("name").reset_index(name="count")) if len(det) else [],
        "metrics": {"baseline": records(base)[0] if len(base) else None, "clip": records(clip)[0] if len(clip) else None},
    })

HIDDEN = OUT/"hidden_incidents.json"  # incidents the user chose to hide in the app
_hidden_lock = threading.Lock()
def hidden_ids():
    try: return set(json.loads(HIDDEN.read_text()))
    except (FileNotFoundError, ValueError): return set()

@app.post("/api/incidents/<iid>/hidden")
def set_hidden(iid):
    ev = load("events/structured_events.csv")
    if ev.empty or not (ev.incident_id == iid).any(): abort(404)
    hide = bool((request.get_json(silent=True) or {}).get("hidden", True))
    with _hidden_lock:
        ids = hidden_ids(); ids.add(iid) if hide else ids.discard(iid)
        HIDDEN.write_text(json.dumps(sorted(ids)))
    return jsonify({"incident_id": iid, "hidden": hide})

def verified_videos():
    # Videos the CLIP classifier labels Crime in its cross-validated predictions (phase 13).
    p = load("evaluation/clip_classifier_predictions.csv")
    return set(p[p.y_pred == 1].video) if len(p) else None

@app.get("/api/incidents")
def incidents():
    ev, sm = load("events/structured_events.csv"), load("summaries/incident_summaries.csv")
    if ev.empty: return jsonify([])
    if len(sm): ev = ev.merge(sm[[c for c in ("incident_id", "visual_caption", "edit_only") if c in sm.columns]], on="incident_id", how="left")
    edit = ev.edit_only.fillna(False).astype(bool) if "edit_only" in ev.columns else False
    # Verified = CLIP calls the video Crime and the key moment is real footage, not a cut to a logo/black frame.
    vv = verified_videos(); ev["verified"] = (ev.video.isin(vv) if vv is not None else True) & ~edit
    ev["hidden"] = ev.incident_id.isin(hidden_ids())
    return jsonify(records(ev, ["incident_id", "video", "category", "reference_label", "severity", "severity_score",
                                "start_seconds", "end_seconds", "duration_seconds", "anomaly_confidence", "visual_caption",
                                "edit_only", "verified", "hidden"]))

@app.get("/api/incidents/<iid>")
def incident(iid):
    ev = load("events/structured_events.csv"); e = ev[ev.incident_id == iid]
    if e.empty: abort(404)
    e = {k: clean(v) for k, v in e.iloc[0].to_dict().items()}
    sm = load("summaries/incident_summaries.csv"); s = sm[sm.incident_id == iid] if len(sm) else sm
    get = lambda k: clean(s.iloc[0][k]) if len(s) and k in s.columns else None
    e["visual_caption"], e["caution"], e["key_objects"] = get("visual_caption"), get("caution"), get("key_objects")
    e["edit_only"] = bool(get("edit_only") or False)
    e["key_frame"] = f"/media/frame/{iid}.jpg" if get("key_frame") else None
    vv = verified_videos(); e["verified"] = ((e["video"] in vv) if vv is not None else True) and not e["edit_only"]
    e["hidden"] = iid in hidden_ids()
    det = load("detections/yolo_detections.csv"); d = det[det.incident_id == iid] if len(det) else det
    e["objects"] = records(d.class_name.value_counts().rename_axis("name").reset_index(name="count")) if len(d) else []
    sc = load("anomaly/anomaly_scores.csv")
    e["timeline"] = records(sc[sc.video == e["video"]].sort_values("timestamp_seconds"), ["timestamp_seconds", "z_score", "anomaly"])
    # Left: the full source video. Right: the annotated key-moment clip from phase 11.
    vv = load("visual_verification/visual_verification_index.csv"); v = vv[vv.incident_id == iid] if len(vv) else vv
    e["key_moment"] = {k: clean(v.iloc[0][k]) for k in ("peak_seconds", "start_seconds", "end_seconds") if k in v.columns} if len(v) else None
    ann = OUT/"visual_verification"/"annotated"/f"{iid}.mp4"
    # ?v=<mtime> makes the browser fetch a regenerated clip instead of a cached older one.
    e["clips"] = {"full": f"/media/source/{iid}.mp4",
                  "annotated": f"/media/annotated/{iid}.mp4?v={int(ann.stat().st_mtime)}" if len(v) and ann.exists() else None}
    return jsonify(e)

def source_video(iid):
    inc = load("anomaly/incidents.csv"); r = inc[inc.incident_id == iid]
    if r.empty: return None
    p = Path(r.iloc[0].video_path)
    return p if p.exists() and p.resolve().is_relative_to((ROOT/"dataset").resolve()) else None

@app.get("/media/<kind>/<iid>.mp4")
def media(kind, iid):
    if kind == "source": p = source_video(iid)
    elif kind in ("original", "annotated"): p = OUT/"visual_verification"/kind/f"{iid}.mp4"
    else: abort(404)
    if p is None or not p.exists(): abort(404)
    r = send_file(p, mimetype="video/mp4", conditional=True)  # conditional=True enables seeking (range requests)
    r.headers["Cache-Control"] = "no-cache"  # always revalidate: clips are regenerated in place
    return r

@app.get("/media/frame/<iid>.jpg")
def key_frame(iid):
    # The exact frame the caption describes (phase 10).
    sm = load("summaries/incident_summaries.csv"); s = sm[sm.incident_id == iid] if len(sm) else sm
    if s.empty or "key_frame" not in s.columns or pd.isna(s.iloc[0].key_frame): abort(404)
    p = Path(s.iloc[0].key_frame)
    if not (p.exists() and p.resolve().is_relative_to((ROOT/"frames").resolve())): abort(404)
    return send_file(p, mimetype="image/jpeg")

# ---------- React build ----------
@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def spa(path):
    if not DIST.exists(): return "Frontend not built. Run: cd frontend && npm install && npm run build", 503
    return send_from_directory(DIST, path) if path and (DIST/path).is_file() else send_from_directory(DIST, "index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
