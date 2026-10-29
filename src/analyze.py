"""Analyse one video end to end with the saved models, using the same rules as the pipeline:
motion incidents (phases 3-4), video-edit trimming and annotated incident clips (phase 11),
YOLO objects (phase 5), BLIP caption of the strongest frame (phase 10) and the CLIP classifier (phase 13).

Videos from the curated set (curation.json) are recognised by their content hash. Their matching incident
also gets the reviewed description, which a person wrote after watching the clip; it is returned
separately from the AI caption so the two are never confused.

Command line:  python src/analyze.py path/to/video.mp4
"""
import hashlib, json, subprocess, sys, threading
from pathlib import Path
import cv2, numpy as np, pandas as pd
from common import ROOT, OUT, MODELS, YOLO_WEIGHTS, TRACK_CLASSES, TRACK_CONF, DETECT_EVERY, is_blank_img
from phase11_visual_verification import draw, SCALE

GAP = 4  # flagged samples closer than this (seconds) form one incident, as in phase 4
_models, _lock = {}, threading.Lock()

def models():
    with _lock:
        if not _models:
            import joblib, torch
            from transformers import CLIPModel, CLIPProcessor, pipeline
            _models["torch"] = torch
            # models/clip and models/blip are not in git (too large); download them once if they're missing.
            local = lambda name, hub: str(MODELS/name) if (MODELS/name/"config.json").exists() else hub
            clip_id = local("clip", "openai/clip-vit-base-patch32")
            _models["clip"] = CLIPModel.from_pretrained(clip_id).eval()
            _models["proc"] = CLIPProcessor.from_pretrained(clip_id)
            _models["clf"] = joblib.load(MODELS/"clip_classifier.joblib")["model"]
            _models["blip"] = pipeline("image-to-text", model=local("blip", "Salesforce/blip-image-captioning-base"), device=-1)
            _models["cal"] = json.loads((MODELS/"calibration.json").read_text())
    return _models

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

_known = None
def known_videos():
    # content hash -> curated incidents of that video: [(incident_id, clip_start, clip_end, reviewed text)]
    global _known
    if _known is None:
        _known = {}
        cur = json.loads((ROOT/"curation.json").read_text(encoding="utf-8")) if (ROOT/"curation.json").exists() else {}
        inc = pd.read_csv(OUT/"anomaly"/"incidents.csv").set_index("incident_id")
        vv = pd.read_csv(OUT/"visual_verification"/"visual_verification_index.csv").set_index("incident_id")
        for iid, text in cur.get("reviewed", {}).items():
            r = inc.loc[iid]
            src = next((p for p in (ROOT/"dataset"/"s_videos"/r.video, ROOT/"dataset"/"videos"/r.category/r.video) if p.exists()), None)
            if src is None: continue
            _known.setdefault(sha256(src), {"video": r.video, "category": r.category, "incidents": []})["incidents"].append(
                (iid, float(vv.loc[iid, "start_seconds"]), float(vv.loc[iid, "end_seconds"]), text))
    return _known

def sample(path, every):
    # Phase 3 sampling: one frame every `every` seconds, round-tripped through JPEG like the saved frames.
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    if fps <= 0: cap.release(); return [], 0
    step = max(1, round(every*fps)); idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = cap.read()
        if not ok: break
        out.append((idx/fps, cv2.imdecode(cv2.imencode(".jpg", f)[1], cv2.IMREAD_COLOR))); idx += step
    cap.release(); return out, total/fps

def transcode(src, dst):
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)

def incident_clip(video, start, end, title, dst):
    # Phase 11 style: YOLO tracking on native frames, small labels drawn at SCALE x.
    from ultralytics import YOLO
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dst.with_suffix(".raw.mp4")
    aw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (SCALE*w, SCALE*h))
    yolo = YOLO(YOLO_WEIGHTS); n = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, start*1000)
    while cap.get(cv2.CAP_PROP_POS_MSEC)/1000 <= end:
        ok, frame = cap.read()
        if not ok: break
        if n % DETECT_EVERY == 0:  # boxes from the last detection are reused on the frames in between
            res = yolo.track(frame, persist=True, tracker="bytetrack.yaml", conf=TRACK_CONF, imgsz=320,
                             classes=list(TRACK_CLASSES), verbose=False)[0]
        aw.write(draw(frame, res, title)); n += 1
    cap.release(); aw.release()
    if n: transcode(tmp, dst)
    tmp.unlink(missing_ok=True)
    return n

def analyze(video, job_dir, max_incidents=5, progress=print):
    video, job_dir = Path(video), Path(job_dir); job_dir.mkdir(parents=True, exist_ok=True)
    progress("Loading the models")
    m = models(); cal = m["cal"]; torch = m["torch"]; step = cal["sample_every_seconds"]
    progress("Sampling frames")
    frames, duration = sample(video, step)
    if len(frames) < 2: raise ValueError("Could not read enough frames from this video.")
    times = [t for t, _ in frames]

    # Motion z-scores (phase 4) and video-edit flags (phase 11).
    prev, z = None, []
    for _, im in frames:
        g = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (32, 24)), (5, 5), 0)
        z.append(((0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev))))-cal["median"])/cal["sigma"]); prev = g
    z = np.array(z); flagged = z >= cal["z_threshold"]
    blank = [is_blank_img(im) for _, im in frames]
    edit = [b or (k > 0 and blank[k-1]) for k, b in enumerate(blank)]

    # CLIP classifier (phase 13): crime vs normal for the whole video.
    progress("Running the CLIP classifier")
    from PIL import Image
    rgb = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for _, im in frames]; E = []
    with torch.no_grad():
        for i in range(0, len(rgb), 32):
            e = m["clip"].get_image_features(**m["proc"](images=rgb[i:i+32], return_tensors="pt"))
            E.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    E = np.concatenate(E)
    prob = float(m["clf"].predict_proba(np.concatenate([E.mean(0), E.max(0), [z.mean(), z.max(), flagged.mean()]])[None])[0, 1])

    # Incidents: group flagged samples, trim video edits, window = one step before first real sample to the last.
    groups, cur = [], []
    for k in np.flatnonzero(flagged):
        if cur and times[k]-times[cur[-1]] > GAP: groups.append(cur); cur = []
        cur.append(k)
    if cur: groups.append(cur)
    found, edits = [], 0
    for g in groups:
        real = [k for k in g if not edit[k]]
        if not real: edits += 1; continue
        pk = max(real, key=lambda k: z[k])
        found.append({"start": max(0, times[real[0]]-step), "end": times[real[-1]], "peak_k": pk, "peak_z": float(z[pk])})

    # Curated videos: recognised by content; their reviewed incident fixes the window and adds the description.
    progress("Checking the curated set")
    known = known_videos().get(sha256(video))
    for iid, s, e, text in (known["incidents"] if known else []):
        hit = next((f for f in found if f["start"] <= e and s <= f["end"]), None)
        if hit is None:
            k = int(np.argmin([abs(t-e) for t in times])); hit = {"peak_k": k, "peak_z": float(z[k])}; found.append(hit)
        hit.update(start=s, end=e, reviewed=text, curated_id=iid)
    keep = [f for f in found if "reviewed" in f]
    # Curated videos show only their reviewed incident; other motion spikes in them are not the event.
    if not keep: keep = sorted(found, key=lambda f: -f["peak_z"])[:max_incidents]
    keep.sort(key=lambda f: f["start"])

    # Per incident: annotated clip, key frame, YOLO objects and BLIP caption.
    from ultralytics import YOLO
    det = YOLO(YOLO_WEIGHTS); out = []
    for n, f in enumerate(keep, 1):
        name = f"UPL_{n:02d}"; progress(f"Incident {n} of {len(keep)}: clip, objects and caption")
        t, im = frames[f["peak_k"]]
        cv2.imwrite(str(job_dir/f"{name}_frame.jpg"), im)
        boxes = det.predict(im, conf=TRACK_CONF, imgsz=320, classes=list(TRACK_CLASSES), verbose=False)[0].boxes
        objs = pd.Series([TRACK_CLASSES[int(c)] for c in boxes.cls.tolist()]).value_counts() if len(boxes) else pd.Series(dtype=int)
        caption = m["blip"](Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), max_new_tokens=40)[0]["generated_text"]
        frames_written = incident_clip(video, f["start"], f["end"], f"CineWatch | {name}", job_dir/f"{name}.mp4")
        out.append({"incident_id": name, "start_seconds": round(f["start"], 1), "end_seconds": round(f["end"], 1),
                    "peak_seconds": round(t, 1), "peak_z": round(f["peak_z"], 2), "frames_written": frames_written,
                    "clip": f"{name}.mp4" if frames_written else None, "key_frame": f"{name}_frame.jpg",
                    "ai_caption": caption, "key_objects": ", ".join(f"{c} {k}" for k, c in objs.items()) or "none",
                    "reviewed_context": f.get("reviewed"), "curated_id": f.get("curated_id")})

    progress("Preparing the full video")
    transcode(video, job_dir/"full.mp4")
    return {"prediction": "Crime" if prob >= .5 else "Normal", "crime_probability": round(prob, 4),
            "duration_seconds": round(duration, 1), "frames_sampled": len(frames),
            "incidents_found": len(found), "video_edits_skipped": edits, "incidents": out,
            "known_video": {"video": known["video"], "category": known["category"]} if known else None,
            "timeline": [{"timestamp_seconds": round(t, 1), "z_score": round(float(s), 3), "anomaly": bool(a)}
                         for t, s, a in zip(times, z, flagged)]}

if __name__ == "__main__":
    import uuid
    r = analyze(sys.argv[1], OUT/"uploads"/uuid.uuid4().hex[:12])
    print(json.dumps({k: v for k, v in r.items() if k != "timeline"}, indent=2))
