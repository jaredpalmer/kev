"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chess, type Square } from "chess.js";
import Link from "next/link";
import { askModel, EVAL_LEVELS, legalMove, loadGames, replay, saveGames, resultText, type Mode, type ModelMove, type SavedGame } from "@/lib/chess";
import { api } from "@/lib/kev";
import { ChessBoard } from "@/components/chess-board";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

const MODES: { value: Mode; label: string }[] = [
  { value: "self", label: "Model vs model" },
  { value: "kev_vs_jev", label: "Kev (White) vs Jev (Black)" },
  { value: "white", label: "You play White" },
  { value: "black", label: "You play Black" },
];

const TOP_N = 8;
type Row = { key: string; p: number; top: boolean };

function topRows(probs: Record<string, number>, chosen: string, n: number): Row[] {
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const rows = sorted.slice(0, n).map(([key, p]) => ({ key, p, top: key === chosen }));
  if (!rows.some((r) => r.top) && probs[chosen] !== undefined) rows[n - 1] = { key: chosen, p: probs[chosen], top: true }; // sampled move outside the top N
  return rows;
}
function topMass(probs: Record<string, number>, n: number) {
  return Object.values(probs).sort((a, b) => b - a).slice(0, n).reduce((s, p) => s + p, 0);
}

// Fixed-height panel: always renders `nRows` bar lanes and one-line header fields, so the column never shifts
// when the number of legal moves changes or while the model is thinking.
function DistributionPanel({ id, title, headline, detail, rows, nRows, mono, footer, dim }: {
  id: string; title: string; headline?: string; detail?: string; rows: Row[]; nRows: number; mono?: boolean; footer?: string; dim?: boolean;
}) {
  const lanes = "grid grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)_2.75rem] items-center gap-x-3";
  return (
    <section aria-labelledby={`p-${id}`} className={`grid gap-3 rounded-md border border-border bg-card px-4 py-3 transition-opacity md:grid-cols-[11rem_minmax(0,1fr)] md:gap-6 ${dim ? "opacity-60" : ""}`}>
      <div className="flex min-w-0 flex-col">
        <h3 id={`p-${id}`} className="font-mono text-[12px] text-muted-foreground">{id}</h3>
        <p className="mt-0.5 h-10 text-[13px] leading-5 text-foreground">{title}</p>
        <p className={`mt-1.5 h-6 truncate text-base font-medium tracking-tight ${mono ? "font-mono" : ""}`}>{headline ?? "\u00a0"}</p>
        <p className="h-5 text-[12px] tabular-nums text-muted-foreground">{detail ?? "\u00a0"}</p>
      </div>
      <div className="flex min-w-0 flex-col">
        {Array.from({ length: nRows }, (_, i) => {
          const r = rows[i];
          return (
            <div key={i} className={`${lanes} h-5 text-[12px]`}>
              <span className={`truncate ${mono ? "font-mono" : ""} ${r?.top ? "text-foreground" : "text-muted-foreground"}`} title={r?.key}>{r?.key ?? "\u00a0"}</span>
              <span className="relative block h-1 min-w-0 rounded-full bg-muted" aria-hidden>
                {r && <span className={`absolute inset-y-0 left-0 rounded-full ${r.top ? "bg-foreground" : "bg-muted-foreground/50"}`} style={{ width: `${Math.round(r.p * 100)}%` }} />}
              </span>
              <span className={`text-right tabular-nums ${r?.top ? "text-foreground" : "text-muted-foreground"}`}>{r ? r.p.toFixed(2) : "\u00a0"}</span>
            </div>
          );
        })}
        {footer !== undefined && <p className="mt-1 h-4 text-[11px] text-muted-foreground">{footer}</p>}
      </div>
    </section>
  );
}

function newGame(mode: Mode): SavedGame {
  return { id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`, startedAt: Date.now(), mode, pgn: "", moves: [] };
}

function rebuild(g: SavedGame) {
  return replay(g.moves).chess;
}

export function ChessGame() {
  const [games, setGames] = useState<SavedGame[]>([]);
  const [game, setGame] = useState<SavedGame | null>(null);
  const [mode, setMode] = useState<Mode>("self");
  const [sample, setSample] = useState(false);
  const [auto, setAuto] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Square | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const autoRef = useRef(false);
  const gameRef = useRef<SavedGame | null>(null); // latest persisted game, so async replies validate against the live position

  // load from localStorage (an external store) after mount; deferred so SSR and first client render match
  useEffect(() => {
    const t = setTimeout(() => {
      const saved = loadGames();
      setGames(saved);
      const last = saved.at(-1);
      const g = last && !last.result ? last : newGame("self");
      if (last && !last.result) setMode(last.mode);
      gameRef.current = g; setGame(g);
    }, 0);
    api.models().then((m) => setModel(m.models[0].run)).catch(() => setModel(null));
    return () => clearTimeout(t);
  }, []);

  const chess = useMemo(() => (game ? rebuild(game) : new Chess()), [game]);
  const lastMove = useMemo(() => { const h = chess.history({ verbose: true }); const m = h.at(-1); return m ? { from: m.from, to: m.to } : null; }, [chess]);
  const humanSide = mode === "white" ? "w" : mode === "black" ? "b" : null;
  const humanToMove = !!game && !game.result && humanSide === chess.turn();
  const modelToMove = !!game && !game.result && humanSide !== chess.turn();
  const lastModel = useMemo(() => [...(game?.moves ?? [])].reverse().find((m) => m.model)?.model, [game]);

  const persist = useCallback((g: SavedGame) => {
    gameRef.current = g;
    setGame(g);
    setGames((prev) => { const next = [...prev.filter((x) => x.id !== g.id), g]; saveGames(next); return next; });
  }, []);

  // The only way a move enters a game. `base` is the game the move was chosen for; it is applied only if that is still
  // the live game (id and ply count) and `san` is legal in that position. Returns the resulting position, or null if rejected.
  const applyMove = useCallback((base: SavedGame, san: string, by: "human" | "model", info?: ModelMove): Chess | null => {
    const cur = gameRef.current;
    if (!cur || cur.id !== base.id || cur.moves.length !== base.moves.length || cur.result) return null; // position changed while the move was being chosen
    const mv = legalMove(cur, san);
    if (!mv) { setError(`Rejected illegal move ${san} for ${by}`); return null; }
    const c = rebuild(cur);
    c.move(mv.san);
    persist({ ...cur, moves: [...cur.moves, { san: mv.san, by, model: info }], pgn: c.pgn(), result: resultText(c) });
    return c;
  }, [persist]);

  const modelMove = useCallback(async () => {
    if (!game || game.result || thinking) return;
    setThinking(true); setError(null);
    try {
      const provider: "kev" | "jev" = game.mode === "kev_vs_jev" && rebuild(game).turn() === "b" ? "jev" : "kev";
      const info = await askModel(rebuild(game), sample, provider);
      const c = applyMove(game, info.san, "model", info);
      if (!c || c.isGameOver()) { setAuto(false); autoRef.current = false; }
    } catch (e) { setError((e as Error).message); setAuto(false); autoRef.current = false; }
    finally { setThinking(false); }
  }, [game, sample, thinking, applyMove]);

  // autoplay loop: after each state change, if it's the model's turn and auto is on, move again
  useEffect(() => { autoRef.current = auto; }, [auto]);
  useEffect(() => {
    if (!auto || !game || game.result || thinking) return;
    if (humanSide !== null && humanSide === chess.turn()) return;
    const t = setTimeout(() => { if (autoRef.current) modelMove(); }, 250);
    return () => clearTimeout(t);
  }, [auto, game, thinking, chess, humanSide, modelMove]);

  // in human-vs-model modes, the model replies automatically (and opens when you play Black)
  useEffect(() => {
    if (humanSide === null || !game || game.result || thinking || chess.turn() === humanSide) return;
    const last = game.moves.at(-1);
    if (game.moves.length === 0 || last?.by === "human") {
      const t = setTimeout(modelMove, 150);
      return () => clearTimeout(t);
    }
  }, [game, chess, humanSide, thinking, modelMove]);

  const targets = useMemo(() => new Set(selected ? chess.moves({ square: selected, verbose: true }).map((m) => m.to) : []), [chess, selected]);

  function onSquare(sq: Square) {
    if (!humanToMove || thinking) return;
    const piece = chess.get(sq);
    if (selected && targets.has(sq)) {
      // several legal moves share from/to only for promotions; promote to a queen
      const candidates = chess.moves({ square: selected, verbose: true }).filter((m) => m.to === sq);
      const mv = candidates.find((m) => !m.promotion || m.promotion === "q") ?? candidates[0];
      if (mv) applyMove(game!, mv.san, "human");
      setSelected(null); return;
    }
    if (piece && piece.color === chess.turn()) setSelected(sq); else setSelected(null);
  }

  function start(m: Mode) {
    setAuto(false); autoRef.current = false; setMode(m); setSelected(null); setError(null);
    persist(newGame(m));
  }

  function undo() {
    if (!game || game.moves.length === 0) return;
    setAuto(false); autoRef.current = false;
    // in human modes undo the model reply too, so it is the human's turn again
    const n = humanSide && game.moves.at(-1)?.by === "model" ? 2 : 1;
    const moves = game.moves.slice(0, -n);
    const c = replay(moves).chess;
    persist({ ...game, moves, pgn: c.pgn(), result: undefined });
  }

  const finished = games.filter((g) => g.result);
  const movesRef = useRef<HTMLOListElement>(null);
  useEffect(() => { movesRef.current?.scrollTo({ top: movesRef.current.scrollHeight }); }, [game?.moves.length]);

  const lastProviderName = lastModel?.provider === "jev" ? "Jev" : lastModel?.provider === "kev" ? "Kev" : null;

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col px-6 pt-8 pb-16 md:px-10">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <nav className="flex items-baseline gap-4 text-[15px]">
          <Link href="/" className="text-muted-foreground hover:text-foreground">kev</Link>
          <span className="font-medium tracking-tight">chess</span>
        </nav>
        <p className="text-[13px] text-muted-foreground">{model ? <span className="font-mono">{model}</span> : "connecting"}</p>
      </header>

      <div className="mt-10 max-w-2xl">
        <h1 className="text-2xl font-medium tracking-tight">Every move is a Choice question.</h1>
        <p className="mt-2 text-[15px] leading-6 text-muted-foreground">
          The legal moves are the options, the board is the state. The model returns a probability for each move and a Score for who is better, in one request. The model has never seen a chess game, so expect the distributions to be more interesting than the play.
        </p>
      </div>

      <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,34rem)_minmax(0,1fr)]">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
            {MODES.map((m) => (
              <button key={m.value} type="button" onClick={() => start(m.value)} aria-current={mode === m.value ? "true" : undefined}
                className={`border-b pb-0.5 ${mode === m.value ? "border-foreground text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
                {m.label}
              </button>
            ))}
          </div>

          <ChessBoard chess={chess} selected={selected} targets={targets} lastMove={lastMove} flipped={mode === "black"} onSquare={onSquare} disabled={!humanToMove || thinking} />

          <div className="flex flex-wrap items-center gap-2">
            {mode === "self" || mode === "kev_vs_jev" ? (
              <>
                <Button onClick={() => setAuto((a) => !a)} disabled={!!game?.result} className="rounded-md">{auto ? "Pause" : "Play"}</Button>
                <Button variant="outline" onClick={modelMove} disabled={auto || thinking || !!game?.result} className="rounded-md shadow-none">Step</Button>
              </>
            ) : (
              <Button variant="outline" onClick={modelMove} disabled={!modelToMove || thinking} className="rounded-md shadow-none">Model moves</Button>
            )}
            <Button variant="ghost" onClick={undo} disabled={!game || game.moves.length === 0 || thinking} className="rounded-md text-muted-foreground">Undo</Button>
            <Button variant="ghost" onClick={() => start(mode)} className="rounded-md text-muted-foreground">New game</Button>
            <div className="ml-auto flex items-center gap-2">
              <Switch id="sample" checked={sample} onCheckedChange={(v) => setSample(!!v)} size="sm" />
              <Label htmlFor="sample" className="text-[13px] text-muted-foreground">Sample from distribution</Label>
            </div>
          </div>

          <p className="min-h-5 text-[13px] text-muted-foreground">
            {game?.result ? <span className="text-foreground">{game.result}</span>
              : thinking ? `${mode === "kev_vs_jev" ? (chess.turn() === "w" ? "Kev (White)" : "Jev (Black)") : "Model"} is choosing`
              : humanToMove ? `Your move (${humanSide === "w" ? "White" : "Black"}). Click a piece, then a square.`
              : mode === "kev_vs_jev" ? `${chess.turn() === "w" ? "Kev (White)" : "Jev (Black)"} to move`
              : `${chess.turn() === "w" ? "White" : "Black"} to move`}
            {chess.inCheck() && !game?.result ? " · check" : ""}
          </p>
          {error && <pre className="whitespace-pre-wrap text-[13px] text-destructive">{error}</pre>}
        </div>

        <div className="flex min-w-0 flex-col gap-3">
          <DistributionPanel id="move" title={lastModel ? `${lastProviderName ? `${lastProviderName}: ` : ""}Best move among ${lastModel.n_legal} legal moves` : "Best move among the legal moves"}
            headline={lastModel?.san} detail={lastModel ? `confidence ${lastModel.confidence.toFixed(2)}` : undefined}
            rows={lastModel ? topRows(lastModel.probabilities, lastModel.san, TOP_N) : []} nRows={TOP_N} mono
            footer={lastModel && lastModel.n_legal > TOP_N ? `${lastModel.n_legal - TOP_N} more moves share the remaining ${(1 - topMass(lastModel.probabilities, TOP_N)).toFixed(2)}` : " "}
            dim={thinking} />
          <DistributionPanel id="evaluation" title="Who is better in this position?"
            headline={lastModel ? `${lastModel.evaluation.toFixed(2)} of 4` : undefined} detail={lastModel?.evalConfidence !== undefined ? `confidence ${lastModel.evalConfidence.toFixed(2)}` : undefined}
            rows={EVAL_LEVELS.map((l, i) => ({ key: `${i}  ${l}`, p: lastModel?.evalProbabilities[String(i)] ?? 0, top: !!lastModel && i === Math.round(lastModel.evaluation) }))} nRows={EVAL_LEVELS.length}
            dim={thinking} />
          <p className="h-5 text-[13px] tabular-nums text-muted-foreground">
            {lastModel ? `${lastProviderName ? `${lastProviderName} · ` : ""}${lastModel.latency_ms.toFixed(0)} ms · ${lastModel.input_tokens} input tokens · ${lastModel.n_legal} options` : "The model's distribution appears here after its first move."}
          </p>

          <div className="rounded-md border border-border bg-card px-4 py-3">
            <p className="text-[12px] text-muted-foreground">Moves</p>
            <ol ref={movesRef} className="mt-1 grid h-40 grid-cols-[2.5rem_1fr_1fr] content-start gap-y-0.5 overflow-y-auto font-mono text-[13px] tabular-nums">
              {Array.from({ length: Math.ceil((game?.moves.length ?? 0) / 2) }, (_, i) => (
                <li key={i} className="contents">
                  <span className="text-muted-foreground">{i + 1}.</span>
                  <span>{game!.moves[2 * i]?.san}</span>
                  <span>{game!.moves[2 * i + 1]?.san ?? ""}</span>
                </li>
              ))}
              {game && game.moves.length === 0 && <li className="col-span-3 font-sans text-muted-foreground">No moves yet.</li>}
            </ol>
          </div>
        </div>
      </div>

      <div className="mt-8 rounded-md border border-border bg-card px-4 py-3">
        <p className="text-[12px] text-muted-foreground">Previous games (stored in this browser)</p>
        {finished.length === 0 ? (
          <p className="mt-1 text-[13px] text-muted-foreground">None yet. Finished games are listed here.</p>
        ) : (
          <ul className="mt-1 grid gap-x-8 text-[13px] md:grid-cols-2">
            {[...finished].reverse().slice(0, 8).map((g) => (
              <li key={g.id} className="flex items-baseline justify-between gap-3 py-0.5">
                <span className="text-muted-foreground">{new Date(g.startedAt).toLocaleString()} · {MODES.find((m) => m.value === g.mode)?.label}</span>
                <span className="tabular-nums">{g.moves.length} plies · {g.result}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
