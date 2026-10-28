import React, { useEffect, useMemo, useState } from "react";

const Z_THRESHOLD = 3.5;
const fmt = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `Request failed (${r.status})`);
  return body;
}

export default function App() {
  const [tab, setTab] = useState("overview");
  const [summary, setSummary] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getJSON("/api/summary"), getJSON("/api/incidents")])
      .then(([s, i]) => { setSummary(s); setIncidents(i); })
      .catch((e) => setError(e.message));
  }, []);

  // Hidden incidents are stored by the backend, so they stay hidden after a refresh or restart.
  async function setHidden(id, hidden) {
    await getJSON(`/api/incidents/${id}/hidden`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ hidden }),
    });
    setIncidents((list) => list.map((i) => (i.incident_id === id ? { ...i, hidden } : i)));
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">●</span>
          <div>
            <h1>CineWatch</h1>
            <p>Surveillance video analytics · UCF-Crime</p>
          </div>
        </div>
        <nav className="tabs">
          {[["overview", "Overview"], ["incidents", "Incidents"]].map(([k, label]) => (
            <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>
          ))}
        </nav>
      </header>
      <main>
        {error && <div className="card error">Could not reach the backend: {error}. Is <code>python backend/app.py</code> running?</div>}
        {tab === "overview" && <Overview s={summary} />}
        {tab === "incidents" && <Incidents list={incidents} onHide={setHidden} />}
      </main>
      <footer>Candidate incidents require human verification. Severity is heuristic; captions describe visible content and do not establish that a crime occurred.</footer>
    </div>
  );
}

/* ---------------- Overview ---------------- */
function Overview({ s }) {
  if (!s) return <div className="card muted">Loading results…</div>;
  const b = s.metrics.baseline, c = s.metrics.clip;
  const rows = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1 score", "f1"], ["Specificity", "specificity"]];
  const maxObj = Math.max(1, ...s.top_objects.map((o) => o.count));
  return (
    <>
      <section className="stats">
        <Stat label="Videos" value={s.videos} sub={`${s.crime_videos} crime · ${s.normal_videos} normal`} />
        <Stat label="Frames sampled" value={s.frames.toLocaleString()} sub="1 every 2 s" />
        <Stat label="Incidents" value={s.incidents} sub="motion-flagged" />
        <Stat label="Object detections" value={s.detections.toLocaleString()} sub="YOLOv8n" />
        <Stat label="CLIP accuracy" value={pct(c?.accuracy)} sub="5-fold cross-validated" accent />
      </section>

      <section className="grid2">
        <div className="card">
          <h2>Model comparison</h2>
          <p className="muted">Video-level crime vs normal, same 100 videos.</p>
          <div className="legend"><span className="sw base" />Motion baseline<span className="sw clip" />CLIP classifier</div>
          {rows.map(([label, k]) => (
            <div className="metric" key={k}>
              <span className="mlabel">{label}</span>
              <div className="bars">
                <Bar v={b?.[k]} cls="base" />
                <Bar v={c?.[k]} cls="clip" />
              </div>
            </div>
          ))}
        </div>
        <div className="card">
          <h2>Confusion matrices</h2>
          <div className="cms">
            <Confusion title="Motion baseline" m={b} />
            <Confusion title="CLIP classifier" m={c} />
          </div>
          {c?.accuracy_mean_5_seeds !== undefined && (
            <p className="muted small">CLIP accuracy over 5 random splits: {pct(c.accuracy_mean_5_seeds)} ± {fmt(c.accuracy_std_5_seeds * 100, 1)} pts.</p>
          )}
        </div>
      </section>

      <section className="grid2">
        <div className="card">
          <h2>Severity by reference label</h2>
          <table className="table">
            <thead><tr><th>Label</th><th>High</th><th>Medium</th><th>Low</th></tr></thead>
            <tbody>
              {s.severity.map((r) => (
                <tr key={r.reference_label}>
                  <td>{r.reference_label === "normal" ? "Normal" : "Crime"}</td>
                  <td>{r.High ?? 0}</td><td>{r.Medium ?? 0}</td><td>{r.Low ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h2>Most detected objects</h2>
          {s.top_objects.map((o) => (
            <div className="hbar" key={o.name}>
              <span>{o.name}</span>
              <div><i style={{ width: `${(o.count / maxObj) * 100}%` }} /></div>
              <b>{o.count}</b>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

const Stat = ({ label, value, sub, accent }) => (
  <div className={`card stat ${accent ? "accent" : ""}`}><span>{label}</span><strong>{value}</strong><small>{sub}</small></div>
);
const Bar = ({ v, cls }) => (
  <div className="bar"><i className={cls} style={{ width: `${(v ?? 0) * 100}%` }} /><em>{fmt(v)}</em></div>
);
function Confusion({ title, m }) {
  if (!m) return null;
  return (
    <div className="cm">
      <h3>{title}</h3>
      <div className="cmgrid">
        <span /><span className="h">Pred. normal</span><span className="h">Pred. crime</span>
        <span className="h">Normal</span><b className="ok">{m.TN}</b><b className="bad">{m.FP}</b>
        <span className="h">Crime</span><b className="bad">{m.FN}</b><b className="ok">{m.TP}</b>
      </div>
    </div>
  );
}

/* ---------------- Incidents ---------------- */
function Incidents({ list, onHide }) {
  const [cat, setCat] = useState("All");
  const [sev, setSev] = useState("All");
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [showHidden, setShowHidden] = useState(false);
  const cats = useMemo(() => ["All", ...new Set(list.map((i) => i.category))], [list]);
  const verifiedCount = list.filter((i) => i.verified).length;
  const hiddenCount = list.filter((i) => i.hidden).length;
  const shown = list.filter((i) => (showAll || i.verified) && (showHidden || !i.hidden) &&
    (cat === "All" || i.category === cat) && (sev === "All" || i.severity === sev) &&
    (!q || `${i.incident_id} ${i.video} ${i.visual_caption ?? ""}`.toLowerCase().includes(q.toLowerCase())));
  useEffect(() => { if (!shown.some((i) => i.incident_id === sel)) setSel(shown[0]?.incident_id ?? null); }, [shown, sel]);

  async function toggleHidden(id, hidden) {
    // When hiding the open incident, move on to its neighbour in the list.
    const k = shown.findIndex((i) => i.incident_id === id);
    const next = shown[k + 1] ?? shown[k - 1];
    await onHide(id, hidden);
    if (hidden && !showHidden) setSel(next?.incident_id ?? null);
  }

  return (
    <div className="incidents">
      <aside className="card list">
        <div className="filters">
          <input placeholder="Search id, video, caption…" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="row">
            <select value={cat} onChange={(e) => setCat(e.target.value)}>{cats.map((c) => <option key={c}>{c}</option>)}</select>
            <select value={sev} onChange={(e) => setSev(e.target.value)}>{["All", "High", "Medium", "Low"].map((c) => <option key={c}>{c}</option>)}</select>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
            Also show unverified motion flags ({list.length - verifiedCount})
          </label>
          {hiddenCount > 0 && (
            <label className="toggle">
              <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} />
              Show incidents I hid ({hiddenCount})
            </label>
          )}
          <small className="muted">{shown.length} shown · {verifiedCount} of {list.length} verified by the CLIP classifier</small>
        </div>
        <ul>
          {shown.map((i) => (
            <li key={i.incident_id} className={`${sel === i.incident_id ? "sel" : ""} ${i.hidden ? "is-hidden" : ""}`} onClick={() => setSel(i.incident_id)}>
              <div className="li-top"><b>{i.incident_id}</b><span>{i.hidden && <span className="flag">hidden</span>}{!i.verified && <span className="flag">{i.edit_only ? "video edit" : "motion only"}</span>} <Sev s={i.severity} /></span></div>
              <div className="li-sub">{i.category} · {fmt(i.start_seconds, 0)}–{fmt(i.end_seconds, 0)} s</div>
              {i.visual_caption && <div className="li-cap">“{i.visual_caption}”</div>}
            </li>
          ))}
        </ul>
      </aside>
      {sel ? <IncidentDetail id={sel} hidden={!!list.find((i) => i.incident_id === sel)?.hidden} onToggleHidden={toggleHidden} />
        : <div className="card muted">No incidents to show.{hiddenCount > 0 && " Tick “Show incidents I hid” to bring hidden ones back."}</div>}
    </div>
  );
}

const Sev = ({ s }) => <span className={`sev ${String(s).toLowerCase()}`}>{s}</span>;

function IncidentDetail({ id, hidden, onToggleHidden }) {
  const [d, setD] = useState(null);
  const [busy, setBusy] = useState(false);
  async function toggle() { setBusy(true); try { await onToggleHidden(id, !hidden); } finally { setBusy(false); } }
  useEffect(() => { setD(null); getJSON(`/api/incidents/${id}`).then(setD).catch(() => setD({ error: true })); }, [id]);
  if (!d) return <div className="card muted">Loading {id}…</div>;
  if (d.error) return <div className="card error">Could not load {id}.</div>;
  return (
    <div className="detail">
      <div className="card">
        <div className="dhead">
          <div><h2>{d.incident_id}</h2><p className="muted">{d.video} · {d.category} · reference label: {d.reference_label}</p></div>
          <span>{d.verified ? <span className="vbadge ok">Verified by CLIP</span>
            : <span className="vbadge">{d.edit_only ? "Video edit (cut / logo / black frame)" : "Motion only · CLIP says Normal"}</span>} <Sev s={d.severity} />
            <button className={`hide-btn ${hidden ? "on" : ""}`} disabled={busy} onClick={toggle}
              title={hidden ? "Show this incident in the list again" : "Hide this incident from the list"}>{hidden ? "Unhide" : "Hide"}</button></span>
        </div>
        <SideBySide full={d.clips.full} start={d.start_seconds} end={d.end_seconds}
          annotated={d.clips.annotated} km={d.key_moment && { start: d.key_moment.start_seconds, end: d.key_moment.end_seconds, peak: d.key_moment.peak_seconds }} />
      </div>
      <div className="grid2">
        <div className="card">
          <h3>Event context</h3>
          <div className="context">
            {d.key_frame && <img src={d.key_frame} alt="Key-moment frame the caption describes" />}
            <div>
              <p className="muted small">AI caption of the incident's strongest frame (BLIP):</p>
              <p className="caption">{d.visual_caption ? `“${d.visual_caption}”` : "No caption yet."}</p>
              <p className="small"><b>Objects in this frame (YOLO):</b> {d.key_objects ?? "—"}</p>
            </div>
          </div>
          {d.caution && <p className="muted small">{d.caution} Captions come from a general-purpose model and can be inaccurate.</p>}
          <h3>Objects detected (whole incident)</h3>
          <div className="chips">{d.objects.length ? d.objects.map((o) => <span key={o.name}>{o.name} × {o.count}</span>) : <span className="muted">none</span>}</div>
        </div>
        <div className="card">
          <h3>Details</h3>
          <dl className="kv">
            <dt>Severity score</dt><dd>{fmt(d.severity_score)}</dd>
            <dt>Anomaly confidence</dt><dd>{fmt(d.anomaly_confidence)}</dd>
            <dt>Start → end</dt><dd>{fmt(d.start_seconds, 1)} s → {fmt(d.end_seconds, 1)} s</dd>
            <dt>Duration</dt><dd>{fmt(d.duration_seconds, 1)} s</dd>
            <dt>Samples</dt><dd>{d.sample_count}</dd>
            <dt>Tracking records</dt><dd>{d.tracking_records}</dd>
          </dl>
        </div>
      </div>
      <div className="card">
        <h3>Motion timeline — {d.video}</h3>
        <Timeline points={d.timeline} start={d.start_seconds} end={d.end_seconds} />
      </div>
    </div>
  );
}

// Left: the full original video, opened at the incident start. Right: the annotated incident alone.
function SideBySide({ full, start, end, annotated, km }) {
  return (
    <div className="videos">
      <figure>
        {full ? <video key={full} src={`${full}#t=${start ?? 0}`} controls muted /> : <div className="novideo">Full video unavailable</div>}
        <figcaption>Full original video · incident {fmt(start, 0)}–{fmt(end, 0)} s (opens at {fmt(start, 0)} s)</figcaption>
      </figure>
      <figure>
        {annotated ? <video key={annotated} src={annotated} controls muted /> : <div className="novideo">No incident clip. Run phase 11.</div>}
        <figcaption>Incident only {km ? `${fmt(km.start, 1)}–${fmt(km.end, 1)} s (${fmt(km.end - km.start, 0)} s) · strongest at ${fmt(km.peak, 0)} s` : ""} · annotated: YOLO + tracking</figcaption>
      </figure>
    </div>
  );
}

function Timeline({ points, start, end, windows }) {
  const wins = windows ?? (start !== undefined && end !== undefined ? [{ start, end }] : []);
  if (!points?.length) return <p className="muted">No timeline data.</p>;
  const W = 800, H = 180, P = { l: 40, r: 10, t: 10, b: 28 };
  const tmax = Math.max(...points.map((p) => p.timestamp_seconds), 1);
  const zs = points.map((p) => p.z_score ?? 0);
  const zmax = Math.max(Z_THRESHOLD + 1, ...zs), zmin = Math.min(0, ...zs);
  const x = (t) => P.l + (t / tmax) * (W - P.l - P.r);
  const y = (z) => P.t + (1 - (z - zmin) / (zmax - zmin)) * (H - P.t - P.b);
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(p.timestamp_seconds).toFixed(1)},${y(p.z_score ?? 0).toFixed(1)}`).join(" ");
  return (
    <svg className="timeline" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Motion z-score over time">
      {wins.map((w) => (
        <rect key={w.start} className="window" x={x(w.start) - 4} y={P.t} width={Math.max(8, x(w.end) - x(w.start) + 8)} height={H - P.t - P.b} />
      ))}
      <line className="thr" x1={P.l} x2={W - P.r} y1={y(Z_THRESHOLD)} y2={y(Z_THRESHOLD)} />
      <text className="thr-l" x={W - P.r} y={y(Z_THRESHOLD) - 4} textAnchor="end">threshold z = {Z_THRESHOLD}</text>
      <path className="zline" d={line} />
      {points.filter((p) => p.anomaly).map((p) => <circle key={p.timestamp_seconds} className="dot" cx={x(p.timestamp_seconds)} cy={y(p.z_score)} r="3.5" />)}
      <line className="axis" x1={P.l} x2={W - P.r} y1={H - P.b} y2={H - P.b} />
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <text key={f} className="tick" x={x(f * tmax)} y={H - 8} textAnchor="middle">{Math.round(f * tmax)} s</text>
      ))}
      <text className="tick" x={4} y={y(zmax) + 10}>z</text>
    </svg>
  );
}
