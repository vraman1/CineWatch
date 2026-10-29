import json, math, sys, threading, uuid
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory, abort

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"; DIST = ROOT/"frontend"/"dist"
UPLOADS = OUT/"uploads"
sys.path.insert(0, str(ROOT/"src"))  # the Analyze tab reuses the pipeline code in src/analyze.py
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 500*1024*1024  # 500 MB upload limit

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

# Only people and cars are shown (same rule as the annotated clips); vans/pickups ("truck") count as cars.
SHOWN_OBJECTS = {"person": "person", "car": "car", "truck": "car"}

def people_and_cars(text):
    # "6 car, 1 bus, 1 person, 2 truck" -> "8 car, 1 person"
    if not isinstance(text, str): return text
    counts = {}
    for part in text.split(","):
        n, _, name = part.strip().partition(" ")
        if name in SHOWN_OBJECTS and n.isdigit(): counts[SHOWN_OBJECTS[name]] = counts.get(SHOWN_OBJECTS[name], 0) + int(n)
    return ", ".join(f"{c} {k}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1])) or "none"

def curation():
    # Hand-picked featured incidents and human-reviewed descriptions (curation.json in the project root).
    try: return json.loads((ROOT/"curation.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError): return {}

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
    # Featured incidents come first, in the curated order; the rest keep their id order.
    cur = curation(); rank = {iid: k+1 for k, iid in enumerate(cur.get("featured", []))}
    ev["featured"] = ev.incident_id.map(rank)
    ev["reviewed_context"] = ev.incident_id.map(cur.get("reviewed", {}))
    ev = ev.sort_values(["featured", "incident_id"], na_position="last")
    return jsonify(records(ev, ["incident_id", "video", "category", "reference_label", "severity", "severity_score",
                                "start_seconds", "end_seconds", "duration_seconds", "anomaly_confidence", "visual_caption",
                                "edit_only", "verified", "hidden", "featured", "reviewed_context"]))

@app.get("/api/incidents/<iid>")
def incident(iid):
    ev = load("events/structured_events.csv"); e = ev[ev.incident_id == iid]
    if e.empty: abort(404)
    e = {k: clean(v) for k, v in e.iloc[0].to_dict().items()}
    sm = load("summaries/incident_summaries.csv"); s = sm[sm.incident_id == iid] if len(sm) else sm
    get = lambda k: clean(s.iloc[0][k]) if len(s) and k in s.columns else None
    e["visual_caption"], e["caution"] = get("visual_caption"), get("caution")
    e["key_objects"] = people_and_cars(get("key_objects"))
    e["edit_only"] = bool(get("edit_only") or False)
    vv = verified_videos(); e["verified"] = ((e["video"] in vv) if vv is not None else True) and not e["edit_only"]
    e["hidden"] = iid in hidden_ids()
    cur = curation(); feat = cur.get("featured", [])
    e["featured"] = feat.index(iid)+1 if iid in feat else None
    e["reviewed_context"] = cur.get("reviewed", {}).get(iid)
    det = load("detections/yolo_detections.csv"); d = det[det.incident_id == iid] if len(det) else det
    d = d[d.class_name.isin(SHOWN_OBJECTS)].class_name.map(SHOWN_OBJECTS) if len(d) else d
    e["objects"] = records(d.value_counts().rename_axis("name").reset_index(name="count")) if len(d) else []
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

def local_path(stored, base, depth):
    # CSVs hold absolute paths from the machine that ran the pipeline. If that path doesn't exist here
    # (project copied or unzipped elsewhere), rebuild it from its last `depth` parts under `base`.
    base = base.resolve()
    for p in (Path(str(stored).replace("\\", "/")), base.joinpath(*Path(str(stored).replace("\\", "/")).parts[-depth:])):
        if p.exists() and p.resolve().is_relative_to(base): return p
    return None

def source_video(iid):
    inc = load("anomaly/incidents.csv"); r = inc[inc.incident_id == iid]
    if r.empty: return None
    return local_path(r.iloc[0].video_path, ROOT/"dataset"/"videos", 2)  # <Category>/<video>.mp4

@app.get("/media/<kind>/<iid>.mp4")
def media(kind, iid):
    if kind == "source": p = source_video(iid)
    elif kind in ("original", "annotated"): p = OUT/"visual_verification"/kind/f"{iid}.mp4"
    else: abort(404)
    if p is None or not p.exists(): abort(404)
    r = send_file(p, mimetype="video/mp4", conditional=True)  # conditional=True enables seeking (range requests)
    r.headers["Cache-Control"] = "no-cache"  # always revalidate: clips are regenerated in place
    return r

# ---------- analyse a video (Analyze tab) ----------
_analyze_lock = threading.Lock()  # one analysis at a time: it is CPU-heavy

# Analysis runs as a background job so the page can show which step is running.
JOBS = {}  # job id -> {"status": "queued|running|done|error", "step": str, "result": dict, "error": str}

def run_job(job, src, folder, label):
    from analyze import analyze
    def progress(msg): JOBS[job]["step"] = msg
    try:
        with _analyze_lock:  # queued jobs wait here
            JOBS[job]["status"] = "running"
            r = analyze(src, folder, progress=progress)
        base = f"/media/uploads/{job}/"
        for i in r["incidents"]:
            i["clip"] = base+i["clip"] if i["clip"] else None; i["key_frame"] = base+i["key_frame"]
        r.update(file=label, full_video=base+"full.mp4")
        JOBS[job].update(status="done", result=r)
    except ValueError as e:
        JOBS[job].update(status="error", error=str(e))
    except Exception as e:
        JOBS[job].update(status="error", error=f"Analysis failed: {e}")
    finally:
        src.unlink(missing_ok=True)  # keep only the generated clips

@app.post("/api/analyze")
def analyze_video():
    f = request.files.get("video")
    if not (f and f.filename): return jsonify({"error": "Choose a video first."}), 400
    job = uuid.uuid4().hex[:12]; folder = UPLOADS/job; folder.mkdir(parents=True, exist_ok=True)
    src = folder/("input"+(Path(f.filename).suffix.lower() or ".mp4")); f.save(src)
    JOBS[job] = {"status": "queued", "step": "Waiting for the previous analysis to finish"}
    threading.Thread(target=run_job, args=(job, src, folder, f.filename), daemon=True).start()
    return jsonify({"job": job})

@app.get("/api/analyze/<job>")
def analyze_status(job):
    if job not in JOBS: abort(404)
    return jsonify(JOBS[job])

@app.get("/media/uploads/<job>/<name>")
def upload_media(job, name):
    p = UPLOADS/job/name
    if not (job.isalnum() and p.suffix in (".mp4", ".jpg") and p.resolve().parent == (UPLOADS/job).resolve() and p.exists()): abort(404)
    return send_file(p, mimetype="video/mp4" if p.suffix == ".mp4" else "image/jpeg", conditional=True)

# ---------- React build ----------
@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def spa(path):
    if not DIST.exists(): return "Frontend not built. Run: cd frontend && npm install && npm run build", 503
    return send_from_directory(DIST, path) if path and (DIST/path).is_file() else send_from_directory(DIST, "index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
