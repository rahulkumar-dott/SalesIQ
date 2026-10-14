import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

const fmtTime = (t) => {
  const s = Math.floor(t || 0);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const money = (x) => (x == null ? "–" : `$${Number(x).toLocaleString()}`);
const arr = (x) => (Array.isArray(x) ? x : []);
const BASIS = {
  one_time: "one-time", monthly: "per month", yearly: "per year", total_contract: "total contract",
  premium_monthly: "premium / month", premium_yearly: "premium / year", principal: "loan principal",
  rent_monthly: "rent / month",
};
// Deal object with a fallback for results written before `deal` existed
const dealOf = (s) => s?.deal?.amount != null ? s.deal : s?.deal_value_usd != null ? { amount: s.deal_value_usd, currency: "USD" } : null;
const fmtMoney = (amount, currency) => {
  if (amount == null) return "–";
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency: currency || "USD", maximumFractionDigits: 0 }).format(amount);
  } catch {
    return `${currency || ""} ${Number(amount).toLocaleString()}`.trim();
  }
};
const fmtDeal = (d) => (d ? fmtMoney(d.amount, d.currency) : "–");
const dealSub = (d, product) => [BASIS[d?.basis], product].filter(Boolean).join(" · ") || null;
const humanize = (k) => k.replace(/_/g, " ");
const sentimentText = (s) => (s && typeof s === "object" ? s.sentiment : s) || "–";
const sentimentTrend = (s) => (s && typeof s === "object" ? s.trend : null);

function usePoll(fn, ms, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const run = useCallback(async () => {
    try {
      setData(await fn());
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    run();
    if (!ms) return;
    const t = setInterval(run, ms);
    return () => clearInterval(t);
  }, [run, ms]);
  return { data, error, reload: run };
}

function Badge({ kind, children }) {
  return <span className={`badge ${kind || ""}`}>{children}</span>;
}

const outcomeKind = { won: "good", lost: "bad", follow_up: "warn", undecided: "warn", unclear: "" };

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? "–"}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section className="card">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function List({ items, empty = "None" }) {
  const xs = arr(items);
  if (!xs.length) return <p className="muted">{empty}</p>;
  return (
    <ul>
      {xs.map((x, i) => (
        <li key={i}>{typeof x === "string" ? x : JSON.stringify(x)}</li>
      ))}
    </ul>
  );
}

/* ---------------------------------------------------------------- new analysis */

function NewAnalysis({ onStarted }) {
  const [tab, setTab] = useState("dataset");
  const [sample, setSample] = useState("");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const ds = usePoll(api.dataset, 0);
  const input = useRef(null);

  const start = async () => {
    setBusy(true);
    setErr(null);
    try {
      const job = tab === "dataset" ? await api.startSample(sample) : await api.upload(file);
      onStarted(job);
      setFile(null);
      if (input.current) input.current.value = "";
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };
  const ready = tab === "dataset" ? !!sample : !!file;

  return (
    <section className="card">
      <h3>Analyse a call</h3>
      <div className="tabs small">
        <button className={tab === "dataset" ? "on" : ""} onClick={() => setTab("dataset")}>
          Dataset sample
        </button>
        <button className={tab === "upload" ? "on" : ""} onClick={() => setTab("upload")}>
          Upload file
        </button>
      </div>

      {tab === "dataset" ? (
        <>
          <label className="field">
            <span>
              {ds.data?.dataset || "liva-ai/mock-sales"} <em>({ds.data?.source || "Hugging Face"})</em>
            </span>
            <select value={sample} onChange={(e) => setSample(e.target.value)}>
              <option value="">Select a sample…</option>
              {arr(ds.data?.samples).map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                  {s.analysed ? " ✓ analysed" : s.downloaded ? " · downloaded" : ""}
                  {s.accents?.length ? ` — ${s.accents.filter(Boolean).join(" / ")}` : ""}
                </option>
              ))}
            </select>
          </label>
          <p className="muted">5-minute mock sales clips. Not yet downloaded samples are fetched first.</p>
        </>
      ) : (
        <>
          <label className="field">
            <span>Audio or video file (mp3, wav, m4a, mp4, mov, mkv, webm… · up to 1 GB, 60 min)</span>
            <input
              ref={input}
              type="file"
              accept="audio/*,video/*,.mkv,.m4a,.opus,.wma,.amr,.ts"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>
          <p className="muted">Only the audio track is used: video is ignored and the original file is deleted after the audio is extracted. Calls contain voices of real people, so only upload recordings you have consent for.</p>
        </>
      )}

      <button className="primary" disabled={!ready || busy} onClick={start}>
        {busy ? "Starting…" : "Start analysis"}
      </button>
      {err ? <p className="error">{err}</p> : null}
    </section>
  );
}

function Jobs({ jobs, onOpen }) {
  const list = arr(jobs).slice(0, 6);
  if (!list.length) return null;
  const kind = { done: "good", failed: "bad", running: "warn", queued: "" };
  return (
    <section className="card">
      <h3>Jobs</h3>
      {list.map((j) => (
        <div className="job" key={j.id}>
          <div className="job-top">
            <strong>{j.name}</strong>
            <Badge kind={kind[j.status]}>{j.status === "running" ? j.step : j.status}</Badge>
          </div>
          {j.status === "running" || j.status === "queued" ? (
            <div className="bar">
              <div className="bar-fill pulse" style={{ width: "100%" }} />
            </div>
          ) : null}
          {j.status === "running" && j.log?.length ? <code className="log">{j.log[j.log.length - 1]}</code> : null}
          {j.status === "failed" ? <p className="error">{j.error}</p> : null}
          {j.status === "done" ? (
            <button className="link" onClick={() => onOpen(j.name)}>
              Open results →
            </button>
          ) : null}
        </div>
      ))}
    </section>
  );
}

function CallList({ calls, selected, onSelect }) {
  const xs = arr(calls);
  return (
    <section className="card">
      <h3>Analysed calls</h3>
      {!xs.length ? <p className="muted">Nothing yet. Start an analysis above.</p> : null}
      {xs.map((c) => (
        <button key={c.name} className={`call-row ${selected === c.name ? "on" : ""}`} onClick={() => onSelect(c.name)}>
          <span>{c.name}</span>
          {c.conversation?.verdict === "not_sales" ? (
            <Badge kind="bad">not sales</Badge>
          ) : (
            <Badge kind={outcomeKind[c.outcome]}>{c.outcome || "n/a"}</Badge>
          )}
        </button>
      ))}
    </section>
  );
}

/* ---------------------------------------------------------------- call detail */

function TalkBar({ ratio, roles }) {
  const entries = Object.entries(ratio || {});
  if (!entries.length) return null;
  return (
    <div>
      <div className="talk">
        {entries.map(([spk, r]) => (
          <div
            key={spk}
            className={`talk-seg ${roles?.[spk] === "salesperson" ? "seller" : "buyer"}`}
            style={{ width: `${r * 100}%` }}
            title={`${spk} (${roles?.[spk] || "?"}) ${pct(r)}`}
          >
            {pct(r)}
          </div>
        ))}
      </div>
      <div className="legend">
        <span className="dot seller" /> salesperson <span className="dot buyer" /> customer
      </div>
    </div>
  );
}

const SECTIONS = [
  ["overview", "Overview"],
  ["audio", "Audio metrics"],
  ["sales", "Sales metrics"],
  ["insights", "Insights"],
  ["deal", "Deal details"],
  ["compliance", "Compliance"],
  ["conversation", "Conversation"],
];

const CHECK_KIND = { yes: "good", no: "bad", unclear: "warn", not_applicable: "" };
const QUAL_KIND = { confirmed: "good", likely: "warn", unknown: "" };

function Table({ rows }) {
  return (
    <table className="kv">
      <tbody>
        {rows.map(([k, v], i) => (
          <tr key={i}>
            <td>{k}</td>
            <td>{v ?? "–"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CallDetail({ name, section: wanted, setSection }) {
  const [showEmpty, setShowEmpty] = useState(false);
  const { data: d, error } = usePoll(() => api.call(name), 0, [name]);
  if (error) return <p className="error">{error}</p>;
  if (!d) return <p className="muted">Loading…</p>;

  const ins = d.insights || {};
  const a = d.metrics?.audio || {};
  const s = d.metrics?.sales || {};
  const roles = d.roles || {};
  const speakers = Object.keys(a.talk_seconds || roles);
  const objByType = Object.entries(s.objections_by_type || {});
  const nextStep = s.next_step_set || {};
  const label = (spk) => `${roles[spk] || "speaker"} (${spk})`;
  const yn = (b) => (b ? "yes" : "no");
  const conv = d.conversation || {};
  const notSales = conv.verdict === "not_sales";
  const sections = notSales ? SECTIONS.filter(([id]) => ["overview", "audio", "conversation"].includes(id)) : SECTIONS;
  const section = sections.some(([id]) => id === wanted) ? wanted : "overview";

  return (
    <div className="stack">
      <div className="head">
        <h2>{d.name}</h2>
        <Badge kind={outcomeKind[ins.outcome]}>{ins.outcome || "n/a"}</Badge>
        {d.category_label ? <Badge>{d.category_label}</Badge> : null}
        {dealOf(s) ? <span className="muted">{fmtDeal(dealOf(s))} {dealSub(dealOf(s)) ? `(${dealSub(dealOf(s))})` : ""}</span> : null}
      </div>

      <nav className="tabs sections">
        {sections.map(([id, title]) => (
          <button key={id} className={section === id ? "on" : ""} onClick={() => setSection(id)}>
            {title}
          </button>
        ))}
      </nav>

      {conv.verdict && conv.verdict !== "sales" ? (
        <div className={`banner ${notSales ? "bad" : "warn"}`}>
          <b>{notSales ? "Not a sales conversation" : "Could not tell if this is a sales conversation"}</b>
          {conv.kind && conv.kind !== "unclear" ? ` · looks like: ${humanize(conv.kind)}` : ""}
          {conv.confidence != null ? ` (${pct(conv.confidence)} confidence)` : ""}
          {conv.reason ? <div className="muted">{conv.reason}</div> : null}
          {notSales ? <div className="muted">Sales metrics are hidden and this call is left out of the Overview totals.</div> : null}
        </div>
      ) : null}

      {section === "overview" && notSales && (
        <>
          <p className="summary">{d.summary || "No summary."}</p>
        </>
      )}

      {section === "overview" && !notSales && (
        <>
          <p className="summary">{d.summary || "No summary."}</p>
          <div className="stats">
            <Stat label="Outcome" value={ins.outcome || "unclear"} sub={s.stage_reached ? `stage: ${s.stage_reached}` : null} />
            <Stat label="Deal" value={fmtDeal(dealOf(s))} sub={dealSub(dealOf(s), s.product_sold)} />
            <Stat label="Next step" value={nextStep.set ? "set" : "not set"} sub={nextStep.set ? nextStep.detail : null} />
            <Stat label="Customer" value={sentimentText(ins.customer_sentiment)} />
          </div>
          <div className="grid2">
            <Section title="What went well">
              <List items={arr(ins.rep_performance?.strengths).slice(0, 3)} empty="Nothing flagged" />
            </Section>
            <Section title="Worth coaching">
              <List items={arr(ins.rep_performance?.improvements).slice(0, 3)} empty="Nothing flagged" />
            </Section>
          </div>
          <p className="muted">
            {s.pitch_attempts ?? 0} pitch and {s.close_attempts ?? 0} close attempt(s). Talk time, pauses and other audio
            measures are on the Audio metrics tab.
          </p>
        </>
      )}

      {section === "audio" && (
        <>
          <p className="muted">Computed exactly from timestamps and speaker diarization: no LLM involved.</p>
          <div className="stats">
            <Stat label="Duration" value={a.duration_s != null ? `${a.duration_s} s` : "–"} />
            <Stat label="Speakers" value={a.speakers} />
            <Stat label="Dead air" value={`${a.dead_air_count ?? "–"}×`} sub={`${a.dead_air_s ?? "–"} s total (gaps ≥ 2 s)`} />
            <Stat label="Overlap" value={`${a.overlap_s ?? 0} s`} sub="diarizer rarely detects overlap" />
          </div>
          <Section title="Talk ratio">
            <TalkBar ratio={a.talk_ratio} roles={roles} />
            <p className="muted">Context only. There is no ideal ratio: it depends on the stage of the call.</p>
          </Section>
          <Section title="Per speaker">
            <table className="kv">
              <thead>
                <tr>
                  <th>Speaker</th>
                  <th>Talk time</th>
                  <th>Share</th>
                  <th>Turns</th>
                  <th>Avg turn</th>
                  <th>Longest monologue</th>
                  <th>Interruptions</th>
                </tr>
              </thead>
              <tbody>
                {speakers.map((spk) => (
                  <tr key={spk}>
                    <td>{label(spk)}</td>
                    <td>{a.talk_seconds?.[spk] ?? "–"} s</td>
                    <td>{pct(a.talk_ratio?.[spk])}</td>
                    <td>{a.turns?.[spk] ?? "–"}</td>
                    <td>{a.avg_turn_s?.[spk] ?? "–"} s</td>
                    <td>{a.longest_monologue_s?.[spk] ?? "–"} s</td>
                    <td>{a.interruptions?.[spk] ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>
        </>
      )}

      {section === "sales" && (
        <>
          <p className="muted">
            Tagged by the LLM from the transcript; the code does the counting and the deal arithmetic.
            {d.consistency?.samples > 1
              ? ` Median of ${d.consistency.samples} samples: pitch ${JSON.stringify(d.consistency.pitch_attempts)}, close ${JSON.stringify(d.consistency.close_attempts)}.`
              : ""}
          </p>
          <div className="stats">
            <Stat label="Pitch attempts" value={s.pitch_attempts} />
            <Stat label="Close attempts" value={s.close_attempts} />
            <Stat label="Stage reached" value={s.stage_reached} />
            <Stat label="Buying signals" value={s.buying_signal_count} />
            <Stat label="Negotiation rounds" value={s.negotiation_rounds} />
            <Stat label="Deal" value={fmtDeal(dealOf(s))} sub={dealSub(dealOf(s), s.product_sold)} />
          </div>
          <div className="grid2">
            <Section title="Selling activity">
              <Table
                rows={[
                  ["Asked for the close", yn(s.asked_for_close)],
                  ["Discovery questions", `${s.discovery_questions?.open ?? 0} open · ${s.discovery_questions?.closed ?? 0} closed`],
                  ["Needs uncovered", yn(s.uncovered_needs)],
                  ["Decision-maker identified", yn(s.decision_maker_identified)],
                  ["Price quoted", s.price_quoted || "–"],
                  ...(dealOf(s)?.computed
                    ? [
                        ["Annual value", fmtMoney(dealOf(s).computed.annual_value, dealOf(s).currency)],
                        ["First-year value", fmtMoney(dealOf(s).computed.first_year_value, dealOf(s).currency)],
                        [
                          `Total over ${dealOf(s).computed.term_months} months${dealOf(s).computed.term_assumed ? " (assumed)" : ""}`,
                          fmtMoney(dealOf(s).computed.total_contract_value, dealOf(s).currency),
                        ],
                      ]
                    : []),
                  ["Discount offered", yn(s.discount_offered)],
                  ["Next step set", nextStep.set ? `${nextStep.detail || "yes"}${nextStep.date ? ` (${nextStep.date})` : ""}` : "no"],
                ]}
              />
            </Section>
            <Section title="Objections by type">
              <Table rows={objByType.map(([k, v]) => [k, v])} />
            </Section>
          </div>
          <div className="grid2">
            <Section title="Pitch examples">
              <List items={s.pitch_examples} />
            </Section>
            <Section title="Close examples">
              <List items={s.close_examples} empty="No close attempt by the salesperson" />
            </Section>
          </div>
        </>
      )}

      {section === "insights" && (
        <>
          <p className="summary">{d.summary || "No summary."}</p>
          <div className="grid2">
            <Section title="Objections">
              {arr(ins.objections).length ? (
                arr(ins.objections).map((o, i) => (
                  <div className="obj" key={i}>
                    <Badge kind={o.resolved ? "good" : "bad"}>{o.resolved ? "resolved" : "open"}</Badge> {o.objection}
                    {o.how ? <div className="muted">{o.how}</div> : null}
                  </div>
                ))
              ) : (
                <p className="muted">None raised</p>
              )}
            </Section>
            <Section title="Buying signals">
              <List items={ins.buying_signals} />
            </Section>
          </div>
          <div className="grid2">
            <Section title="Next steps">
              <List items={ins.next_steps} />
            </Section>
            <Section title="Customer">
              <Table
                rows={[
                  ["Sentiment", sentimentText(ins.customer_sentiment)],
                  ["Trend", sentimentTrend(ins.customer_sentiment)],
                  ["Decision-maker", typeof ins.decision_maker === "string" ? ins.decision_maker : JSON.stringify(ins.decision_maker ?? "–")],
                  ["Competitors", arr(ins.competitors).join(", ") || "none mentioned"],
                ]}
              />
            </Section>
          </div>
          <div className="grid2">
            <Section title="Rep: strengths">
              <List items={ins.rep_performance?.strengths} />
            </Section>
            <Section title="Rep: to improve">
              <List items={ins.rep_performance?.improvements} />
            </Section>
          </div>
          <Section title="Who is who">
            <Table
              rows={Object.entries(d.role_evidence || {}).map(([spk, r]) => [
                `${spk} → ${r.role ?? roles[spk] ?? "?"}`,
                r.confidence != null ? `${pct(r.confidence)} · "${r.evidence || ""}"` : "",
              ])}
            />
          </Section>
        </>
      )}

      {section === "deal" && (
        <>
          <p className="muted">
            What this call was about. The fields below depend on the type of deal (<b>{d.category_label || "unknown"}</b>); the
            qualification block applies to every call.
          </p>
          <div className="stats">
            <Stat label="Type" value={d.category_label} />
            <Stat label="Product" value={s.product_sold} />
            <Stat label="Deal" value={fmtDeal(dealOf(s))} sub={dealSub(dealOf(s))} />
          </div>
          <div className="grid2">
            <Section title={`${d.category_label || "Deal"} details`}>
              {(() => {
                const entries = Object.entries(d.details || {});
                const isEmpty = (v) => v === null || v === undefined || v === "" || (Array.isArray(v) && !v.length);
                const filled = entries.filter(([, v]) => !isEmpty(v));
                const empty = entries.filter(([, v]) => isEmpty(v));
                const fmt = (v) =>
                  typeof v === "boolean" ? (v ? "yes" : "no")
                  : Array.isArray(v) ? v.join(", ")
                  : typeof v === "object" ? JSON.stringify(v) : String(v);
                const rows = filled.map(([k, v]) => [humanize(k), fmt(v)]);
                if (showEmpty) rows.push(...empty.map(([k]) => [humanize(k), <span className="muted">not discussed</span>]));
                return (
                  <>
                    {rows.length ? <Table rows={rows} /> : <p className="muted">Nothing specific was discussed.</p>}
                    {empty.length ? (
                      <button className="link" onClick={() => setShowEmpty(!showEmpty)}>
                        {showEmpty ? "Hide" : "Show"} {empty.length} field{empty.length === 1 ? "" : "s"} not discussed
                      </button>
                    ) : null}
                  </>
                );
              })()}
            </Section>
            <Section title="Qualification (all calls)">
              {Object.entries(d.qualification || {}).map(([k, q]) => (
                <div className="obj" key={k}>
                  <Badge kind={QUAL_KIND[q.status]}>{q.status}</Badge> <b>{k}</b>
                  {q.detail ? <div className="muted">{q.detail}</div> : null}
                </div>
              ))}
            </Section>
          </div>
        </>
      )}

      {section === "compliance" && (
        <>
          {Object.keys(d.compliance || {}).length ? (
            <>
              <p className="muted">
                AI-assisted check of what was said in this recording, for review. It is not legal advice and does not replace
                your compliance team; a clip that starts mid-call will show "unclear" for opening disclosures.
              </p>
              <div className="stats">
                <Stat label="Failed checks" value={Object.values(d.compliance).filter((c) => c.result === "no").length} />
                <Stat label="Unclear" value={Object.values(d.compliance).filter((c) => c.result === "unclear").length} />
                <Stat label="Passed" value={Object.values(d.compliance).filter((c) => c.result === "yes").length} />
              </div>
              <Section title="Checks">
                {Object.entries(d.compliance).map(([k, c]) => (
                  <div className="obj" key={k}>
                    <Badge kind={CHECK_KIND[c.result]}>{c.result.replace("_", " ")}</Badge> {humanize(k)}
                    {c.evidence ? <div className="muted">"{c.evidence}"</div> : null}
                  </div>
                ))}
              </Section>
            </>
          ) : (
            <p className="muted">
              Compliance checks run only for regulated types (insurance, loans). This call is classified as{" "}
              <b>{d.category_label || "unknown"}</b>.
            </p>
          )}
        </>
      )}

      {section === "conversation" && (
        <Section title="Conversation">
          <div className="convo">
            {arr(d.utterances).map((u, i) => (
              <div key={i} className={`utt ${u.role === "salesperson" ? "seller" : u.role === "customer" ? "buyer" : ""}`}>
                <div className="utt-meta">
                  <strong>{u.name || u.role || u.speaker}</strong>
                  <span>{fmtTime(u.start)}</span>
                  {u.intent ? <Badge>{u.intent}</Badge> : null}
                </div>
                <div>{u.text}</div>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- overview */

function Bars({ data }) {
  const entries = Object.entries(data || {}).filter(([, v]) => v > 0);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  if (!entries.length) return <p className="muted">No data</p>;
  return (
    <div>
      {entries.map(([k, v]) => (
        <div className="hbar" key={k}>
          <span>{k}</span>
          <div className="bar">
            <div className="bar-fill" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <b>{v}</b>
        </div>
      ))}
    </div>
  );
}

function Overview({ calls }) {
  const { data: g, error } = usePoll(api.aggregate, 0, [arr(calls).length]);
  if (error) return <p className="muted">{error}</p>;
  if (!g) return <p className="muted">Loading…</p>;
  return (
    <div className="stack">
      <h2>Overview · {g.calls} call{g.calls === 1 ? "" : "s"}</h2>
      {g.calls < 5 ? <p className="muted">Few calls analysed: these aggregates are not meaningful yet.</p> : null}
      <div className="stats">
        <Stat label="Win rate" value={pct(g.win_rate)} />
        {Object.entries(g.avg_deal_won_by_currency || {}).map(([cur, v]) => (
          <Stat key={cur} label={`Avg won deal (${cur})`} value={fmtMoney(v, cur)} />
        ))}
        {!Object.keys(g.avg_deal_won_by_currency || {}).length ? <Stat label="Avg won deal" value="–" /> : null}
        <Stat label="Avg pitch attempts" value={g.avg_pitch_attempts} />
        <Stat label="Avg close attempts" value={g.avg_close_attempts} />
        <Stat label="Avg duration" value={g.avg_duration_s != null ? `${Math.round(g.avg_duration_s / 6) / 10} min` : "–"} />
      </div>
      <div className="grid2">
        <Section title="Outcomes"><Bars data={g.outcomes} /></Section>
        <Section title="Stage reached"><Bars data={g.stage_reached} /></Section>
        <Section title="Deal types"><Bars data={g.by_category} /></Section>
        <Section title="Objections by type"><Bars data={g.objections_by_type} /></Section>
        <Section title="By rep">
          <table className="kv">
            <thead>
              <tr><th>Rep</th><th>Calls</th><th>Win</th><th>Pitch</th><th>Close</th></tr>
            </thead>
            <tbody>
              {Object.entries(g.by_rep || {}).map(([n, r]) => (
                <tr key={n}>
                  <td>{n}</td><td>{r.calls}</td><td>{pct(r.win_rate)}</td>
                  <td>{r.avg_pitch_attempts}</td><td>{r.avg_close_attempts}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- app */

export default function App() {
  const [view, setView] = useState("overview");
  const [selected, setSelected] = useState(null);
  const [section, setSection] = useState("overview");
  const calls = usePoll(api.calls, 0);
  const budget = usePoll(api.budget, 15000);
  const jobs = usePoll(api.jobs, 2000);
  const prevDone = useRef(0);

  // reload the call list when a job finishes
  const done = arr(jobs.data).filter((j) => j.status === "done").length;
  useEffect(() => {
    if (done !== prevDone.current) {
      prevDone.current = done;
      calls.reload();
      budget.reload();
    }
  }, [done]); // eslint-disable-line react-hooks/exhaustive-deps

  const open = (name) => {
    setSelected(name);
    setView("call");
  };

  return (
    <div className="app">
      <header>
        <h1>SalesIQ</h1>
        <nav className="tabs">
          <button className={view === "overview" ? "on" : ""} onClick={() => setView("overview")}>
            Overview
          </button>
          <button className={view === "call" ? "on" : ""} disabled={!selected} onClick={() => setView("call")}>
            Call
          </button>
        </nav>
        <span className="spacer" />
        {budget.data ? (
          <span className="muted" title="LLM spend vs cap">
            LLM ${budget.data.usd.toFixed(4)} / ${budget.data.limit_usd.toFixed(2)}
          </span>
        ) : null}
      </header>

      <main>
        <aside>
          <NewAnalysis onStarted={() => jobs.reload()} />
          <Jobs jobs={jobs.data} onOpen={open} />
          <CallList calls={calls.data} selected={selected} onSelect={open} />
        </aside>
        <div className="content">
          {calls.error ? (
            <p className="error">Cannot reach the API ({calls.error}). Start it with: python api.py</p>
          ) : view === "call" && selected ? (
            <CallDetail key={selected} name={selected} section={section} setSection={setSection} />
          ) : (
            <Overview calls={calls.data} />
          )}
        </div>
      </main>
    </div>
  );
}
