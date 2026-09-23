// Chess via the decision model: legal moves become the options of one Choice question,
// the position (board, FEN, history) is the state. A Score question rates the position in the same pass.
import { Chess, type Move } from "chess.js";
import { api, type SystemOneRequest, type SystemOneResponse } from "@/lib/kev";

export const PIECE_NAMES: Record<string, string> = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
export const EVAL_LEVELS = ["Black is clearly winning", "Black is better", "Roughly equal", "White is better", "White is clearly winning"];

export function describeMove(m: Move): string {
  const parts = [`${PIECE_NAMES[m.piece]} ${m.from} to ${m.to}`];
  if (m.captured) parts.push(`captures ${PIECE_NAMES[m.captured]}`);
  if (m.promotion) parts.push(`promotes to ${PIECE_NAMES[m.promotion]}`);
  if (m.flags.includes("k") || m.flags.includes("q")) parts[0] = m.flags.includes("k") ? "castles kingside" : "castles queenside";
  if (m.san.endsWith("#")) parts.push("checkmate");
  else if (m.san.endsWith("+")) parts.push("gives check");
  return parts.join(", ");
}

export function positionState(chess: Chess) {
  const side = chess.turn() === "w" ? "White" : "Black";
  const history = chess.history();
  const moves = history.length ? history.map((m, i) => (i % 2 === 0 ? `${i / 2 + 1}. ${m}` : m)).join(" ") : "(game start)";
  return {
    game: "chess",
    side_to_move: side,
    board: chess.ascii(),
    fen: chess.fen(),
    moves_so_far: moves,
    in_check: chess.inCheck(),
  };
}

export function buildRequest(chess: Chess): { req: SystemOneRequest; legal: Move[] } {
  const legal = chess.moves({ verbose: true });
  const side = chess.turn() === "w" ? "White" : "Black";
  const criteria: Record<string, string> = {};
  for (const m of legal) criteria[m.san] = describeMove(m);
  return {
    legal,
    req: {
      state: positionState(chess),
      model: "kev-latest",
      questions: {
        move: {
          type: "choice",
          instructions: `You are playing ${side}. Choose the best legal move for ${side} in this position. Prefer captures of undefended pieces, checks that win material, and moves that develop pieces toward the center.`,
          criteria,
        },
        evaluation: {
          type: "score",
          instructions: "Who is better in this position, before the move is played?",
          criteria: EVAL_LEVELS,
        },
      },
    },
  };
}

export type ModelMove = {
  san: string;
  probabilities: Record<string, number>;
  confidence: number;
  evaluation: number; // expected level 0..4
  evalConfidence?: number; // absent in games saved by older versions
  evalProbabilities: Record<string, number>;
  latency_ms: number;
  input_tokens: number;
  n_legal: number;
  provider?: "kev" | "jev";
};

export async function askModel(chess: Chess, sample = false, provider: "kev" | "jev" = "kev"): Promise<ModelMove> {
  const { req, legal } = buildRequest(chess);
  if (provider === "jev") req.model = "jev-latest";
  const r: SystemOneResponse = provider === "jev" ? await api.systemOneJev(req) : await api.systemOne(req);
  const a = r.answers.move;
  const e = r.answers.evaluation;
  if (a.type !== "choice" || e.type !== "score") throw new Error("unexpected answer types");
  let san = a.choice;
  if (sample) {
    let u = Math.random();
    for (const [k, p] of Object.entries(a.probabilities)) { u -= p; if (u <= 0) { san = k; break; } }
  }
  if (!legal.some((m) => m.san === san)) throw new Error(`model returned ${JSON.stringify(san)}, which is not a legal move here`);
  return { san, probabilities: a.probabilities, confidence: a.confidence, evaluation: e.score, evalConfidence: e.confidence, evalProbabilities: e.probabilities, latency_ms: r.latency_ms, input_tokens: r.usage.input_tokens, n_legal: legal.length, provider };
}

// ---- persistence -------------------------------------------------------------------------------

export type Mode = "self" | "kev_vs_jev" | "white" | "black"; // who the human plays; "self" = Kev vs Kev

export type SavedGame = {
  id: string;
  startedAt: number;
  mode: Mode;
  pgn: string;
  moves: { san: string; by: "human" | "model"; provider?: "kev" | "jev"; model?: ModelMove }[];
  result?: string;
};

const KEY = "kev.chess.v1";

// Replays a game's moves, stopping at the first one that is not legal in its position (chess.js throws on illegal SAN).
export function replay(moves: SavedGame["moves"]): { chess: Chess; moves: SavedGame["moves"] } {
  const chess = new Chess();
  const ok: SavedGame["moves"] = [];
  for (const m of moves) {
    try { chess.move(m.san); ok.push(m); } catch { break; }
  }
  return { chess, moves: ok };
}

// The legal move `san` in the position after `game`, or undefined if it is not legal there.
export function legalMove(game: SavedGame, san: string): Move | undefined {
  return replay(game.moves).chess.moves({ verbose: true }).find((m) => m.san === san);
}

export function loadGames(): SavedGame[] {
  if (typeof window === "undefined") return [];
  let raw: unknown;
  try { raw = JSON.parse(localStorage.getItem(KEY) ?? "[]"); } catch { return []; }
  if (!Array.isArray(raw)) return [];
  // stored games are untrusted: keep only the legal prefix of each move list so the board and the list agree
  return raw.filter((g): g is SavedGame => !!g && typeof g.id === "string" && Array.isArray(g.moves)).map((g) => {
    const { chess, moves } = replay(g.moves);
    return moves.length === g.moves.length ? g : { ...g, moves, pgn: chess.pgn(), result: resultText(chess) };
  });
}

export function saveGames(games: SavedGame[]) {
  localStorage.setItem(KEY, JSON.stringify(games.slice(-20)));
}

export function resultText(chess: Chess): string | undefined {
  if (!chess.isGameOver()) return undefined;
  if (chess.isCheckmate()) return chess.turn() === "w" ? "0-1 (checkmate)" : "1-0 (checkmate)";
  if (chess.isStalemate()) return "½-½ (stalemate)";
  if (chess.isThreefoldRepetition()) return "½-½ (repetition)";
  if (chess.isInsufficientMaterial()) return "½-½ (insufficient material)";
  if (chess.isDrawByFiftyMoves()) return "½-½ (fifty-move rule)";
  return "½-½ (draw)";
}
