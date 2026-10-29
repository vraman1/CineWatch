import React, { useEffect, useMemo, useRef, useState } from "react";

const Z_THRESHOLD = 3.5;
const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
const clock = (s) => { s = Math.max(0, Math.round(s ?? 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const mb = (bytes) => `${(bytes / 2 ** 20).toFixed(1)} MB`;

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `Request failed (${r.status})`);
  return body;
}

// Steps reported by src/analyze.py, in order. "Incident n of m" messages belong to the "Incident" step.
const STEPS = [
  ["Loading the models", "Loading the models"],
  ["Sampling frames", "Sampling frames and scoring motion"],
  ["Running the CLIP classifier", "Running the CLIP crime classifier"],
  ["Checking the curated set", "Checking for a known video"],
  ["Incident", "Detecting people and cars, captioning incidents"],
  ["Preparing the full video", "Preparing the full video"],
];
const stepIndex = (msg) => (msg ? STEPS.findIndex(([key]) => msg.startsWith(key)) : -1);

export default function App() {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);       // {id, status, step}
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [started, setStarted] = useState(null);
  const [now, setNow] = useState(Date.now());

  // Poll the background job once a second while it runs.
  useEffect(() => {
    if (!job) return;
    const t = setInterval(async () => {
      setNow(Date.now());
      try {
        const s = await getJSON(`/api/analyze/${job.id}`);
        if (s.status === "done") { setRes(s.result); setJob(null); }
        else if (s.status === "error") { setErr(s.error); setJob(null); }
        else setJob({ id: job.id, ...s });
      } catch (e) { setErr(e.message); setJob(null); }
    }, 1000);
    return () => clearInterval(t);
  }, [job]);

  async function run() {
    setErr(null); setRes(null); setStarted(Date.now()); setNow(Date.now());
    const fd = new FormData(); fd.append("video", file);
    try { const { job: id } = await getJSON("/api/analyze", { method: "POST", body: fd }); setJob({ id, status: "queued" }); }
    catch (e) { setErr(e.message); }
  }
  function reset() { setFile(null); setRes(null); setErr(null); setJob(null); }

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-inner">
          <div className="brand">
            <span className="rec" aria-hidden="true" />
            <div>
              <h1>CineWatch</h1>
              <p>AI incident analysis for surveillance video</p>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        {!job && !res && <Upload file={file} setFile={setFile} onRun={run} err={err} />}
        {job && <Progress job={job} file={file} elapsed={(now - started) / 1000} />}
        {res && <Results res={res} onReset={reset} />}
      </main>

      <footer className="foot">
        Detected incidents are candidates for human review. The AI caption describes visible content and does not establish that a crime occurred.
      </footer>
    </div>
  );
}

/* ---------------- Upload ---------------- */
function Upload({ file, setFile, onRun, err }) {
  const [drag, setDrag] = useState(false);
  const input = useRef(null);
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);
  const pick = (f) => { if (f && f.type.startsWith("video/")) setFile(f); };

  return (
    <section className="panel upload-panel">
      {!file ? (
        <div className={`drop ${drag ? "over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
          onClick={() => input.current?.click()} role="button" tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}>
          <svg className="drop-icon" viewBox="0 0 48 48" aria-hidden="true">
            <rect x="6" y="12" width="28" height="24" rx="4" /><path d="M34 21l8-5v16l-8-5z" />
            <path className="arrow" d="M20 30V18m-5 5l5-5 5 5" />
          </svg>
          <h2>Drop a CCTV video here</h2>
          <p>or <span className="link-like">browse your files</span> · MP4, AVI, MOV, MKV</p>
          <input ref={input} type="file" accept="video/*" hidden onChange={(e) => pick(e.target.files[0])} />
        </div>
      ) : (
        <div className="chosen">
          <video src={preview} controls muted />
          <div className="chosen-info">
            <span className="eyebrow">Ready to analyze</span>
            <h2 title={file.name}>{file.name}</h2>
            <p className="muted">{mb(file.size)}</p>
            <div className="actions">
              <button className="btn primary" onClick={onRun}>Analyze video</button>
              <button className="btn ghost" onClick={() => setFile(null)}>Choose another</button>
            </div>
            <p className="muted small">Usually 5–20 seconds; the first video after starting the app also loads the models.</p>
          </div>
        </div>
      )}
      {err && <p className="error-text">{err}</p>}
    </section>
  );
}

/* ---------------- Progress ---------------- */
function Progress({ job, file, elapsed }) {
  const cur = stepIndex(job.step);
  return (
    <section className="panel progress-panel">
      <div className="progress-head">
        <div className="spinner" aria-hidden="true" />
        <div>
          <h2>Analyzing {file?.name}</h2>
          <p className="muted">{job.status === "queued" ? job.step ?? "Starting…" : `${Math.round(elapsed)} s elapsed`}</p>
        </div>
      </div>
      <ol className="steps">
        {STEPS.map(([key, label], k) => (
          <li key={key} className={k < cur ? "done" : k === cur ? "active" : ""}>
            <span className="dot" aria-hidden="true" />
            <span>{label}{k === cur && key === "Incident" && job.step ? ` · ${job.step.split(":")[0]}` : ""}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ---------------- Results ---------------- */
function Results({ res, onReset }) {
  const [hidden, setHidden] = useState([]);
  const shown = res.incidents.filter((i) => !hidden.includes(i.incident_id));
  const crime = res.prediction === "Crime";
  return (
    <>
      <section className="summary">
        <div className={`panel verdict ${crime ? "crime" : "normal"}`}>
          <span className="eyebrow">CLIP classifier verdict</span>
          <strong>{res.prediction}</strong>
          <div className="gauge" role="img" aria-label={`Crime probability ${pct(res.crime_probability)}`}>
            <i style={{ width: pct(res.crime_probability) }} /><b />
          </div>
          <small>Crime probability {pct(res.crime_probability)} · threshold 50%</small>
        </div>
        <div className="panel fact">
          <span className="eyebrow">Video</span>
          <strong>{clock(res.duration_seconds)}</strong>
          <small className="clip-name" title={res.file}>{res.file}</small>
        </div>
        <div className="panel fact">
          <span className="eyebrow">Incidents detected</span>
          <strong>{res.incidents_found}</strong>
          <small>{res.video_edits_skipped ? `${res.video_edits_skipped} video edit(s) ignored` : "from motion analysis"}</small>
        </div>
        <button className="btn ghost again" onClick={onReset}>Analyze another video</button>
      </section>

      {res.known_video && (
        <div className="panel known">
          <Check /> <span>Recognised as <b>{res.known_video.video}</b> ({res.known_video.category}) from the reviewed set, so its selected incident includes a reviewed description.</span>
        </div>
      )}

      {res.incidents.length === 0 && <div className="panel muted">No incidents above the motion threshold in this video.</div>}
      {hidden.length > 0 && (
        <div className="panel muted small">{hidden.length} incident(s) hidden. <button className="link" onClick={() => setHidden([])}>Show all</button></div>
      )}
      {shown.map((i) => (
        <IncidentCard key={i.incident_id} i={i} n={res.incidents.indexOf(i) + 1} full={res.full_video}
          onHide={() => setHidden((h) => [...h, i.incident_id])} />
      ))}

      <section className="panel">
        <div className="section-head"><h3>Motion over time</h3><span className="muted small">Shaded: detected incidents · dashed: alert threshold</span></div>
        <Timeline points={res.timeline} incidents={res.incidents} />
      </section>
    </>
  );
}

function IncidentCard({ i, n, full, onHide }) {
  return (
    <article className="panel incident">
      <header className="incident-head">
        <span className="num">{n}</span>
        <div>
          <h3>Incident {n}</h3>
          <p className="muted">{clock(i.start_seconds)} – {clock(i.end_seconds)} · strongest moment at {clock(i.peak_seconds)}</p>
        </div>
        <div className="head-right">
          {i.reviewed_context && <span className="badge ok"><Check /> Reviewed</span>}
          <button className="btn ghost small" onClick={onHide}>Hide</button>
        </div>
      </header>

      <div className="videos">
        <figure>
          <video src={`${full}#t=${i.start_seconds}`} controls muted preload="metadata" />
          <figcaption>Full video · opens at {clock(i.start_seconds)}</figcaption>
        </figure>
        <figure>
          {i.clip ? <video src={i.clip} controls muted preload="metadata" /> : <div className="novideo">No clip</div>}
          <figcaption>Incident only · people and cars detected</figcaption>
        </figure>
      </div>

      <div className="context">
        {i.reviewed_context && (
          <div className="reviewed">
            <span className="eyebrow">Reviewed description</span>
            <p>{i.reviewed_context}</p>
          </div>
        )}
        <div className="ai">
          <img src={i.key_frame} alt="Strongest frame of the incident" />
          <div>
            <span className="eyebrow">AI caption · strongest frame</span>
            <p className="caption">“{i.ai_caption}”</p>
            <Objects text={i.key_objects} />
          </div>
        </div>
      </div>
    </article>
  );
}

function Objects({ text }) {
  const items = (text && text !== "none") ? text.split(",").map((s) => s.trim().split(" ")) : [];
  if (!items.length) return <p className="muted small">No people or cars detected in this frame.</p>;
  return (
    <div className="chips">
      {items.map(([count, name]) => (
        <span key={name} className={`chip ${name}`}>
          {name === "person" ? <PersonIcon /> : <CarIcon />}
          {count} {name === "person" ? (count === "1" ? "person" : "people") : (count === "1" ? "car" : "cars")}
        </span>
      ))}
    </div>
  );
}

function Timeline({ points, incidents }) {
  if (!points?.length) return <p className="muted">No timeline data.</p>;
  const W = 900, H = 190, P = { l: 36, r: 12, t: 14, b: 28 };
  const tmax = Math.max(...points.map((p) => p.timestamp_seconds), 1);
  const zs = points.map((p) => p.z_score ?? 0);
  const zmax = Math.max(Z_THRESHOLD + 1, ...zs), zmin = Math.min(0, ...zs);
  const x = (t) => P.l + (t / tmax) * (W - P.l - P.r);
  const y = (z) => P.t + (1 - (z - zmin) / (zmax - zmin)) * (H - P.t - P.b);
  const line = points.map((p, k) => `${k ? "L" : "M"}${x(p.timestamp_seconds).toFixed(1)},${y(p.z_score ?? 0).toFixed(1)}`).join(" ");
  const last = points[points.length - 1];
  const area = `${line} L${x(last.timestamp_seconds).toFixed(1)},${y(zmin)} L${x(points[0].timestamp_seconds).toFixed(1)},${y(zmin)} Z`;
  return (
    <svg className="timeline" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Motion score over time with detected incidents">
      {incidents.map((w, k) => (
        <g key={w.incident_id}>
          <rect className="window" x={x(w.start_seconds)} y={P.t} width={Math.max(6, x(w.end_seconds) - x(w.start_seconds))} height={H - P.t - P.b} rx="4" />
          <text className="window-l" x={x(w.start_seconds) + 4} y={P.t + 13}>{k + 1}</text>
        </g>
      ))}
      <line className="thr" x1={P.l} x2={W - P.r} y1={y(Z_THRESHOLD)} y2={y(Z_THRESHOLD)} />
      <path className="zarea" d={area} />
      <path className="zline" d={line} />
      {points.filter((p) => p.anomaly).map((p) => <circle key={p.timestamp_seconds} className="dot" cx={x(p.timestamp_seconds)} cy={y(p.z_score)} r="3.5" />)}
      <line className="axis" x1={P.l} x2={W - P.r} y1={H - P.b} y2={H - P.b} />
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <text key={f} className="tick" x={x(f * tmax)} y={H - 8} textAnchor="middle">{clock(f * tmax)}</text>
      ))}
    </svg>
  );
}

const Check = () => (<svg className="ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3 3 7-7" /></svg>);
const PersonIcon = () => (<svg className="ico" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="4.5" r="2.5" /><path d="M3.5 14c0-2.8 2-4.5 4.5-4.5s4.5 1.7 4.5 4.5" /></svg>);
const CarIcon = () => (<svg className="ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 10.5V8l1.6-3.2c.2-.5.7-.8 1.2-.8h6.4c.5 0 1 .3 1.2.8L14 8v2.5" /><rect x="1.5" y="8" width="13" height="3.5" rx="1" /><circle cx="4.5" cy="12" r="1" /><circle cx="11.5" cy="12" r="1" /></svg>);
