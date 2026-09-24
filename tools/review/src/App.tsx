import { useEffect, useMemo, useRef, useState } from "react";

type Judge = { model: string; label: string; rationale: string };
type Item = {
  id: string; document: string; source: string; proposed_label: string; label_origin: string; judges: Judge[];
  question: { type: string; instructions: string; options: Record<string, string | null> };
  adjudication: { label: string; reason: string } | null;
};
type Verdict = "accept" | "relabel" | "drop";
type Decision = { verdict: Verdict; label: string | null; reviewed_at: string };
type Filter = "undecided" | "all" | "disagree" | "decided";
type Saved = {
  decisions: Record<string, Decision>; notes: Record<string, string>; history: { id: string; prev: Decision | null }[];
  cur: string | null; filter: Filter; spot: boolean; spotN: number;
};

// Option keys: digits, then letters not taken by commands (e j k n u x y).
const KEYS = "1234567890abcdfghilmopqrstvwz";
const SEED = 42;
const FILTERS: Filter[] = ["undecided", "all", "disagree", "decided"];
const EMPTY: Saved = { decisions: {}, notes: {}, history: [], cur: null, filter: "undecided", spot: false, spotN: 50 };
const HELP: [string, string][] = [
  ["Enter / y", "accept proposed label, next"],
  ["1-9 0 a-z", "relabel to that option, next"],
  ["x", "drop (ambiguous / unanswerable), next"],
  ["n", "edit note (Enter saves, Esc cancels)"],
  ["← / j", "previous"],
  ["→ / k", "next"],
  ["u", "undo last decision"],
  ["e", "export reviews.jsonl"],
  ["?", "toggle this help"],
];

// Two independent FNV-1a 32-bit hashes plus the length: plenty to key one person's files.
function hash(s: string): string {
  let a = 0x811c9dc5, b = 0x01000193;
  for (let i = 0; i < s.length; i++) (a = Math.imul(a ^ s.charCodeAt(i), 16777619)), (b = Math.imul(b ^ s.charCodeAt(i), 2654435761));
  return `${(a >>> 0).toString(36)}${(b >>> 0).toString(36)}-${s.length}`;
}

// Fisher-Yates with mulberry32, so a seed gives the same sample everywhere.
function shuffled<T>(xs: T[], seed: number): T[] {
  const out = [...xs];
  let a = seed;
  for (let i = out.length - 1; i > 0; i--) {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const j = Math.floor((((t ^ (t >>> 14)) >>> 0) / 4294967296) * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// Structured fields (e.g. instructions as {question, focus}) become "key: value" lines.
const str = (v: unknown): string =>
  typeof v === "string" ? v : v && typeof v === "object"
    ? Object.entries(v).map(([k, x]) => `${k}: ${typeof x === "string" ? x : JSON.stringify(x)}`).join("\n")
    : v == null ? "" : String(v);

function parse(text: string): { items: Item[]; bad: number } {
  const items: Item[] = [];
  const seen = new Set<string>();
  let bad = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      const r = JSON.parse(line);
      const ok = typeof r.id === "string" && typeof r.document === "string" && typeof r.proposed_label === "string"
        && r.question && typeof r.question.options === "object" && r.question.options && !seen.has(r.id);
      if (!ok) throw new Error();
      seen.add(r.id);
      const options: Record<string, string | null> = {};
      for (const [k, v] of Object.entries(r.question.options)) options[k] = v == null ? null : str(v);
      items.push({
        id: r.id, document: r.document, source: str(r.source), label_origin: str(r.label_origin), proposed_label: r.proposed_label,
        question: { type: str(r.question.type), instructions: str(r.question.instructions), options },
        judges: (Array.isArray(r.judges) ? r.judges : []).map((j: Judge) => ({ model: str(j.model), label: str(j.label), rationale: str(j.rationale) })),
        adjudication: r.adjudication ? { label: str(r.adjudication.label), reason: str(r.adjudication.reason) } : null,
      });
    } catch {
      bad++;
    }
  }
  return { items, bad };
}

const disagree = (it: Item) => it.judges.some((j) => j.label !== it.proposed_label);
const shortModel = (m: string) => m.split("/").pop()!.split("-").slice(0, 2).join("-");

export default function App() {
  const [file, setFile] = useState<{ name: string; items: Item[]; bad: number; key: string } | null>(null);
  const [s, setS] = useState<Saved>(EMPTY);
  const [error, setError] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [help, setHelp] = useState(false);
  const docRef = useRef<HTMLDivElement>(null);
  const proposedRef = useRef<HTMLLIElement>(null);
  const items = file?.items ?? [];

  function load(text: string, name: string) {
    const key = "kev-review:" + hash(text);
    const { items, bad } = parse(text);
    const saved = localStorage.getItem(key);
    setFile({ name, items, bad, key });
    setS(saved ? { ...EMPTY, ...JSON.parse(saved) } : EMPTY);
    setError("");
    try {
      localStorage.setItem("kev-review:last", JSON.stringify({ name, text }));
    } catch {
      localStorage.removeItem("kev-review:last"); // too big to keep; ?file= or re-drop still resumes
    }
  }
  const loadFile = (f: File) => f.text().then((t) => load(t, f.name));

  useEffect(() => {
    const name = new URLSearchParams(location.search).get("file");
    if (name) {
      fetch(name).then((r) => (r.ok && !r.headers.get("content-type")?.includes("html") ? r.text() : Promise.reject(r.ok ? "not found" : r.status)))
        .then((t) => load(t, name), (e) => setError(`Could not fetch ${name}: ${e}`));
      return;
    }
    const last = localStorage.getItem("kev-review:last");
    if (last) {
      const { name, text } = JSON.parse(last);
      load(text, name);
    }
  }, []);

  useEffect(() => {
    if (file) localStorage.setItem(file.key, JSON.stringify(s));
  }, [file, s]);

  const queue = useMemo(() => (s.spot ? shuffled(items, SEED).slice(0, Math.max(1, s.spotN)) : items), [items, s.spot, s.spotN]);
  const matches = (it: Item, d = s.decisions) =>
    s.filter === "all" || (s.filter === "undecided" ? !d[it.id] : s.filter === "decided" ? !!d[it.id] : disagree(it));
  const step = (from: string | null, dir: 1 | -1, d = s.decisions): string | null => {
    const i = queue.findIndex((it) => it.id === from);
    for (let k = 1; k <= queue.length; k++) {
      const it = queue[(((i + dir * k) % queue.length) + queue.length) % queue.length];
      if (matches(it, d)) return it.id;
    }
    return null;
  };
  const curItem = queue.find((it) => it.id === s.cur);
  const item = curItem && matches(curItem) ? curItem : queue.find((it) => it.id === step(s.cur, 1));
  const visible = queue.filter((it) => matches(it));

  useEffect(() => {
    docRef.current?.scrollTo(0, 0);
    proposedRef.current?.scrollIntoView({ block: "center" });
  }, [item?.id]);

  function decide(verdict: Verdict, label: string | null) {
    if (!item) return;
    if (verdict === "relabel" && label === item.proposed_label) verdict = "accept";
    const decisions = { ...s.decisions, [item.id]: { verdict, label, reviewed_at: new Date().toISOString() } };
    const history = [...s.history, { id: item.id, prev: s.decisions[item.id] ?? null }];
    setS({ ...s, decisions, history, cur: step(item.id, 1, decisions) ?? item.id });
  }
  function undo() {
    const last = s.history.at(-1);
    if (!last) return;
    const decisions = { ...s.decisions };
    if (last.prev) decisions[last.id] = last.prev;
    else delete decisions[last.id];
    setS({ ...s, decisions, history: s.history.slice(0, -1), cur: last.id });
  }

  const decided = items.filter((it) => s.decisions[it.id]);
  const count = (xs: Item[], v: Verdict) => xs.filter((it) => s.decisions[it.id]?.verdict === v).length;
  const [acc, rel, drop] = (["accept", "relabel", "drop"] as Verdict[]).map((v) => count(decided, v));
  const qDone = queue.filter((it) => s.decisions[it.id]).length;

  function exportReviews() {
    if (!decided.length) return;
    const lines = decided.map((it) => {
      const d = s.decisions[it.id];
      return JSON.stringify({ id: it.id, verdict: d.verdict, label: d.label, proposed_label: it.proposed_label, note: s.notes[it.id] ?? "", reviewed_at: d.reviewed_at });
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n") + (lines.length ? "\n" : "")], { type: "application/jsonl" }));
    a.download = "reviews.jsonl";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const options = item ? Object.entries(item.question.options) : [];
  const onKey = useRef<(e: KeyboardEvent) => void>(() => {});
  onKey.current = (e) => {
    const t = e.target; // text fields swallow keys; a focused checkbox or file picker must not
    const typing = t instanceof HTMLTextAreaElement || (t instanceof HTMLInputElement && !["checkbox", "file"].includes(t.type));
    if (note !== null || typing || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const idx = KEYS.indexOf(k);
    const run: Record<string, () => void> = {
      "?": () => setHelp(!help),
      Escape: () => setHelp(false),
      e: exportReviews,
      u: undo,
      Enter: () => item && decide("accept", item.proposed_label),
      y: () => item && decide("accept", item.proposed_label),
      x: () => decide("drop", null),
      n: () => item && setNote(s.notes[item.id] ?? ""),
      ArrowLeft: () => item && setS({ ...s, cur: step(item.id, -1) }),
      k: () => item && setS({ ...s, cur: step(item.id, -1) }),
      ArrowRight: () => item && setS({ ...s, cur: step(item.id, 1) }),
      j: () => item && setS({ ...s, cur: step(item.id, 1) }),
    };
    if (run[k]) run[k]();
    else if (k.length === 1 && idx >= 0 && idx < options.length) decide("relabel", options[idx][0]);
    else return;
    e.preventDefault();
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => onKey.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const d = item && s.decisions[item.id];
  const btn = (active = false) => "rounded border px-2 py-0.5 text-xs disabled:opacity-40 "
    + (active ? "border-blue-600 bg-blue-600 text-white" : "border-gray-300 bg-white hover:bg-gray-100");

  return (
    <div
      className="flex h-screen flex-col bg-gray-50 text-gray-900"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (f) loadFile(f);
      }}
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-gray-200 bg-white px-4 py-2 text-xs">
        <span className="font-semibold">Kev review</span>
        <label className={btn() + " cursor-pointer"}>
          {file ? file.name : "Open .jsonl"}
          <input type="file" accept=".jsonl,.json,.txt" className="hidden" onChange={(e) => e.target.files?.[0] && loadFile(e.target.files[0])} />
        </label>
        {file && file.bad > 0 && <span className="text-red-600">{file.bad} bad or duplicate line{file.bad > 1 ? "s" : ""} skipped</span>}
        <span className="ml-2 text-gray-500">Show</span>
        {FILTERS.map((f) => (
          <button key={f} className={btn(s.filter === f)} onClick={() => setS({ ...s, filter: f, cur: item?.id ?? s.cur })}>{f}</button>
        ))}
        <label className="ml-2 flex items-center gap-1">
          <input type="checkbox" checked={s.spot} onChange={(e) => setS({ ...s, spot: e.target.checked, cur: item?.id ?? s.cur })} />
          spot-check first
          <input type="number" min={1} value={s.spotN || ""} className="w-14 rounded border border-gray-300 px-1"
            onChange={(e) => setS({ ...s, spotN: Number(e.target.value) || 0 })}
            onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
          <span className="text-gray-500">(seed {SEED})</span>
        </label>
        <span className="ml-auto text-gray-600">
          accept {acc} · relabel {rel} · drop {drop} · agreement {acc + rel ? ((100 * acc) / (acc + rel)).toFixed(1) + "%" : "–"}
        </span>
        <button className={btn()} onClick={exportReviews} disabled={!decided.length}>Export (e)</button>
        <button className={btn()} onClick={() => setHelp(!help)}>?</button>
      </header>
      {file && (
        <div className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-1 text-xs text-gray-600">
          <div className="h-1.5 w-48 rounded bg-gray-200">
            <div className="h-1.5 rounded bg-blue-600" style={{ width: `${queue.length ? (100 * qDone) / queue.length : 0}%` }} />
          </div>
          <span>
            {qDone} / {queue.length} reviewed, {count(queue, "relabel")} relabelled, {count(queue, "drop")} dropped
          </span>
          {item && <span className="text-gray-400">· item {visible.indexOf(item) + 1} of {visible.length} in "{s.filter}"</span>}
        </div>
      )}
      {!file || !item ? (
        <div className="m-auto text-center text-sm text-gray-500">
          {error || (!file ? "Drop a .jsonl file anywhere, use Open, or add ?file=sample.jsonl to the URL."
            : items.length ? `Nothing left under "${s.filter}". Change the filter, press u to undo, or export.` : "No valid items in this file.")}
        </div>
      ) : (
        <main className="grid min-h-0 flex-1 grid-cols-[1fr_minmax(340px,440px)]">
          <section className="flex min-h-0 flex-col border-r border-gray-200">
            <div className="border-b border-gray-200 bg-white px-6 py-3">
              <div className="text-[11px] text-gray-500">
                {item.source} · <span className="font-mono">{item.id}</span> · {item.question.type} · label {item.label_origin}
              </div>
              <div className="mt-1 whitespace-pre-wrap font-semibold">{item.question.instructions}</div>
              {d && (
                <div className="mt-1 text-xs text-gray-700">
                  Decided: <b>{d.verdict}</b>{d.verdict === "relabel" && <> → <span className="font-mono">{d.label}</span></>}
                </div>
              )}
              {note !== null ? (
                <input autoFocus value={note} placeholder="Note (Enter saves, Esc cancels)" className="mt-2 w-full rounded border border-gray-300 px-2 py-1 text-sm"
                  onChange={(e) => setNote(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") setS({ ...s, notes: { ...s.notes, [item.id]: note.trim() } });
                    if (e.key === "Enter" || e.key === "Escape") setNote(null);
                  }} />
              ) : s.notes[item.id] ? <div className="mt-1 text-xs text-gray-700">Note: {s.notes[item.id]}</div> : null}
            </div>
            <div ref={docRef} className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
              <div className="max-w-[75ch] whitespace-pre-wrap text-[15px] leading-relaxed">{item.document}</div>
            </div>
          </section>
          <aside className="flex min-h-0 flex-col bg-white">
            <ol className="min-h-0 flex-1 overflow-y-auto p-2 text-sm">
              {options.map(([key, desc], i) => {
                const proposed = key === item.proposed_label;
                const chosen = d?.label === key;
                return (
                  <li key={key} ref={proposed ? proposedRef : undefined} onClick={() => decide("relabel", key)}
                    className={`flex cursor-pointer gap-2 rounded border px-2 py-1 ${proposed ? "border-blue-600 bg-blue-50" : chosen ? "border-gray-500" : "border-transparent hover:bg-gray-50"}`}>
                    <kbd className="w-5 shrink-0 text-center font-mono text-xs text-gray-400">{KEYS[i] ?? ""}</kbd>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1">
                        <span className={`font-mono ${proposed ? "font-semibold text-blue-700" : ""}`}>{key}</span>
                        {proposed && <span className="rounded bg-blue-600 px-1 text-[10px] text-white">proposed</span>}
                        {chosen && d?.verdict === "relabel" && <span className="rounded bg-gray-700 px-1 text-[10px] text-white">your label</span>}
                        {item.adjudication?.label === key && <span className="rounded bg-amber-100 px-1 text-[10px] text-amber-800">adjudicated</span>}
                        {item.judges.filter((j) => j.label === key).map((j) => (
                          <span key={j.model} title={j.rationale} className="rounded bg-gray-200 px-1 text-[10px] text-gray-700">{shortModel(j.model)}</span>
                        ))}
                      </div>
                      {desc && <div className="text-xs text-gray-500">{desc}</div>}
                    </div>
                  </li>
                );
              })}
            </ol>
            <div className="space-y-2 border-t border-gray-200 p-3 text-xs">
              {item.adjudication && (
                <div className="rounded bg-amber-50 p-2 text-amber-900">
                  Adjudication: <span className="font-mono font-semibold">{item.adjudication.label}</span> — {item.adjudication.reason}
                </div>
              )}
              <details>
                <summary className="cursor-pointer text-gray-600">Judges ({item.judges.length}){disagree(item) ? " · disagree" : " · agree"}</summary>
                <ul className="mt-1 space-y-1">
                  {item.judges.map((j) => (
                    <li key={j.model}>
                      <span className="font-medium">{j.model}</span> → <span className="font-mono">{j.label}</span>
                      {!(j.label in item.question.options) && <span className="text-red-600"> (not an option)</span>}
                      <div className="text-gray-500">{j.rationale}</div>
                    </li>
                  ))}
                </ul>
              </details>
              <div className="flex gap-1">
                <button className={btn(true)} onClick={() => decide("accept", item.proposed_label)}>Accept (Enter)</button>
                <button className={btn()} onClick={() => decide("drop", null)}>Drop (x)</button>
                <button className={btn()} onClick={() => setNote(s.notes[item.id] ?? "")}>Note (n)</button>
                <button className={btn()} onClick={undo} disabled={!s.history.length}>Undo (u)</button>
              </div>
            </div>
          </aside>
        </main>
      )}
      {help && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/20" onClick={() => setHelp(false)}>
          <table className="rounded bg-white text-sm shadow">
            <tbody>
              {HELP.map(([k, v]) => (
                <tr key={k} className="border-b border-gray-100">
                  <td className="px-3 py-1 font-mono">{k}</td>
                  <td className="px-3 py-1 text-gray-600">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
