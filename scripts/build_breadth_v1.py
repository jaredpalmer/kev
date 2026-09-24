"""breadth-v1: a frozen, eval-only panel for generalisation across the five areas of the community Decision Index, built from
public evaluation splits of datasets that Kev's SFT corpus never trains on.

    uv run python scripts/build_breadth_v1.py --out evals/breadth-v1            # downloads into --raw, ~10 min (BM25 over BRIGHT)
    uv run python scripts/screen_overlap.py --suite evals/breadth-v1 --external /tmp/jevbench/datasets/public --out evals/breadth-v1/overlap.json
    uv run python scripts/breadth_report.py --suite evals/breadth-v1 --results runs/breadth-v1-kev-27b ...

The Decision Index (HF Space multimodalart/jev-decision-index, data/methodology.json, edition 0.2) groups 40 benchmarks into
five equal-weight areas and chance-corrects each benchmark before averaging. breadth-v1 mirrors that structure with the
fifteen held-out datasets it names (DATASETS below is the canonical table: area, licence, pins, label provenance, mapping).
It is built from the original datasets, not from the Index's prepared requests, and maps each one onto Kev's typed
questions (choice / noul) with 2 to MAX_OPTIONS options. Where a source task offers more options than that (CLINC150's 151
intents, ChessBench's legal moves, API-Bank's 53-API catalog, ToolRet / BRIGHT candidate pools, RouterBench's 11 models),
the question is restricted to a candidate subset that always contains the reference answer; the rule for each is in its
`mapping`. That makes these questions easier than the Index's versions, so scores are not comparable to the Index's
numbers, only to other systems read on this panel.

Partitions: development and test, EVAL_RECORDS records per dataset each (ContractNLI: CONTRACTNLI_DOCS documents with up
to CONTRACTNLI_PER_DOC hypotheses each). test is locked: read once per candidate with --allow-test, never for search.
Datasets with a native development/validation and test split keep it (ContractNLI, CLINC150, SGD, Humicroedit); the others
are split into halves by group (story, video, dialogue, query, position, user, ...) in a seeded hash order (deal_groups), so
no group spans the two partitions. Selection inside a partition is round-robin over each dataset's strata (subset, domain,
category, label) in a seeded order. Every state is deduplicated by normalised text across all datasets and both
partitions (test first). Every record is checked with kev.model.fits against the Qwen3.5 tokenizer at ADMISSION (row =
state + one question <= 7,680 tokens, so it also fits the 8,192-token limits of Kev's server and AutoJev's).
Deterministic: the same downloads give byte-identical partitions and manifest (BM25 ties are broken by document id; scores
are rounded before ranking so platform float noise cannot reorder them).
"""
import argparse, ast, csv, hashlib, io, json, math, random, re, struct, sys, tarfile, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.data import materialize  # noqa: E402
from kev.model import fits, load_tokenizer, user_tokens  # noqa: E402
from kev.suite import ADMISSION_TOKENIZER as TOKENIZER, GIT_LIMIT, SERVING_CONTEXT, digest, normalise_text, read_json, read_jsonl, text_digest, write_json, write_jsonl  # noqa: E402
from scripts.build_devtools_v1 import deal_groups, fetch, hub_file, line_safe, parquet_rows, round_robin, state_key  # noqa: E402

VERSION = SEED = "breadth-v1"
PARTITIONS = ("development", "test")
EVAL_RECORDS = 150
CONTRACTNLI_DOCS, CONTRACTNLI_PER_DOC = 40, 4
MAX_OPTIONS = 10            # Kev's supported option count today (kev.api accepts up to 255; AutoJev too)
# stricter than SERVING_CONTEXT (which the manifest records): 1,024 tokens of headroom on the state and 512 on the row, so a
# record also fits servers that count their own template tokens against 8,192 (AutoJev refuses longer requests with a 422)
ADMISSION = {**SERVING_CONTEXT, "max_state": SERVING_CONTEXT["max_state"] - 1024, "max_branch": SERVING_CONTEXT["max_branch"] - 512}
ADMISSION.pop("truncate")
DI_SPACE = ("multimodalart/jev-decision-index", "data/methodology.json")

AREAS = {"knowledge": "Knowledge & Reasoning", "language": "Language Understanding", "retrieval": "Retrieval & Classification",
         "tools": "Tools & Automation", "arts": "Arts & Human Taste"}

MUSR = ("TAUR-Lab/MuSR", "7c365b439a222150f317764d4f16ae6c96d7d94a", ("murder_mystery.csv", "object_placements.csv", "team_allocation.csv"))
SATA = ("sata-bench/sata-bench", "ba43a7ab537adfa3498e3a160a6d1eafbefc95c1", "data_main.json")
CHESS = {"url": "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag", "file": "chessbench_test_action_value_data.bag",
         "sha256": "5f73aac8f60e31734cdbf276ba3fca8d5ba5cb6171ba600e31af4f36327986b0", "code": "google-deepmind/searchless_chess@90ae0e6b121673fc3079aaeffa047580bb600c0a"}
CONTRACTNLI = {"url": "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip", "file": "contract-nli.zip",
               "sha256": "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"}
HELLASWAG = ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data/validation-00000-of-00001.parquet")
CLINC = ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", {"development": "plus/validation-00000-of-00001.parquet", "test": "plus/test-00000-of-00001.parquet"})
CLINC_DOMAINS = {"url": "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/domains.json", "file": "clinc_domains.json",
                 "sha256": "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a"}
SGD = {"url": "https://codeload.github.com/google-research-datasets/dstc8-schema-guided-dialogue/tar.gz/e852981ae34990f4358979625854259302feaa78",
       "file": "dstc8-schema-guided-dialogue-e852981.tar.gz", "sha256": "ff97a9ab52b4cc9f25e1a093c96431512465e8377f9a1f57dc710a10484d2188",
       "commit": "e852981ae34990f4358979625854259302feaa78"}
BRIGHT = ("xlangai/BRIGHT", "3066d29c9651a576c8aba4832d249807b181ecae")
BRIGHT_DOMAINS = ("biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living")
BFCL = ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda")
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance",
                   "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance")
TOOLRET = ("mangopy/ToolRet-Queries", "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445", "mangopy/ToolRet-Tools", "e06c38c75612b6536bd959e08cdd345894aba6a7")
TOOLRET_EXCLUDED = {"apibank": "API-Bank is its own dataset in this panel", "apigen": "Salesforce xLAM-60k, whose card carries a research-only statement"}
APIBANK = ("liminghao1630/API-Bank", "12e8158b7628c168f07e8f31fbbe3445e99f44cf", "test-data/level-1-api.json")
ROUTERBENCH = ("withmartian/routerbench", "784021482c3f320c6619ed4b3bb3b41a21424fcb", "routerbench_0shot.pkl")
ROUTER_MODELS = ("WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2", "gpt-3.5-turbo-1106", "gpt-4-1106-preview",
                 "meta/code-llama-instruct-34b-chat", "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat")
HUMICROEDIT = {"url": "https://cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip", "file": "semeval-2020-task-7-dataset.zip",
               "sha256": "12a6cbf28c8b698ad80be42a65ac867b57e4c71662eedab607805e167ba791ab"}
CFCOLOR = {"url": "https://www.dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip", "file": "cfcolor.zip",
           "sha256": "47c07095642cfab3c2eeab366a5d152b07783cbb5af390cbfda7d7c13db4b54c"}

DATASETS = {
    "musr": {"area": "knowledge", "di": [32, "MuSR"], "metric": "accuracy", "licence": "CC-BY-4.0 (dataset card); code MIT",
             "licence_url": "https://huggingface.co/datasets/TAUR-Lab/MuSR", "attribution": "Sprague et al., MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning (ICLR 2024)",
             "label_provenance": "by construction (the story generator's ground truth), published answer_index", "native_splits": "one 756-item set; split into halves by item",
             "mapping": "state = the narrative; one choice over the published options in published order (keys opt_1..), instructions = the published question; strata = the three subsets (murder mystery, object placements, team allocation)"},
    "sata_bench": {"area": "knowledge", "di": [33, "SATA-Bench"], "metric": "case_exact", "licence": "CC-BY-NC-4.0 (dataset card; the GitHub repo is MIT): the derived records inherit the non-commercial term",
                   "licence_url": "https://huggingface.co/datasets/sata-bench/sata-bench", "attribution": "Xu et al., SATA-Bench: Select All That Apply Benchmark for Multiple Choice Questions (2025)",
                   "label_provenance": "source answer groups (human-labelled multi-answer items pooled by the authors)", "native_splits": "one 1,604-item set; split into halves by item",
                   "mapping": f"state = passage + question (HTML tags removed); one noul per candidate answer ('Is \"X\" a correct answer?'), answers and distractors in a seeded order; scored case-exact (every noul right), as the Index does; items with more than {MAX_OPTIONS} candidates or a candidate that is both answer and distractor are dropped"},
    "chessbench": {"area": "knowledge", "di": [31, "ChessBench"], "metric": "accuracy", "licence": "Apache-2.0 (code and data release); positions from lichess.org (CC0)",
                   "licence_url": "https://github.com/google-deepmind/searchless_chess", "attribution": "Ruoss et al., Amortized Planning with Large-Scale Transformers: A Case Study on Chess (NeurIPS 2024)",
                   "label_provenance": "engine: Stockfish 16 win probability of every legal move (the test action-value file)", "native_splits": "test action-value file only; split into halves by position",
                   "mapping": f"state = FEN, side to move and an ASCII board; one choice over up to {MAX_OPTIONS} legal moves (UCI keys, described as piece, from-to, capture, promotion): the best move plus up to 9 others drawn in a seeded order, leaving out any move within 0.01 win probability of the best so the reference answer is unique; strata = game phase by piece count"},
    "contractnli": {"area": "language", "di": [11, "ContractNLI"], "metric": "accuracy", "licence": "CC-BY-4.0 (Hitachi America terms of use grant CC BY 4.0)",
                    "licence_url": "https://stanfordnlp.github.io/contract-nli/", "attribution": "Koreeda and Manning, ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts (Findings of EMNLP 2021)",
                    "label_provenance": "human (expert annotation of each hypothesis on each NDA)", "native_splits": "development partition from dev.json, test from test.json",
                    "mapping": f"state = the full NDA text (no truncation; longer documents are not admitted); up to {CONTRACTNLI_PER_DOC} of the 17 fixed hypotheses per document, drawn round-robin over the three labels, each a choice entailment / contradiction / not_mentioned"},
    "hellaswag": {"area": "language", "di": [29, "HellaSwag"], "metric": "accuracy", "licence": "MIT (code/data release); ActivityNet Captions contexts only",
                  "licence_url": "https://huggingface.co/datasets/Rowan/hellaswag", "attribution": "Zellers et al., HellaSwag: Can a Machine Really Finish Your Sentence? (ACL 2019)",
                  "label_provenance": "adversarial filtering + human validation (published label)", "native_splits": "validation only (test labels are hidden); split into halves by source video",
                  "mapping": "ActivityNet-derived items only (source_id activitynet~...): the WikiHow-derived half is left out because wikiHow's 2026-09-14 DMCA notice took down the HellaSwag GitHub repository (github/dmca 2026-09-14-wikihow.md); state = activity label + context; one choice over the four endings in published order; strata = in-domain / zero-shot"},
    "clinc150": {"area": "retrieval", "di": [5, "CLINC150+OOS"], "metric": "accuracy", "licence": "CC-BY-3.0",
                 "licence_url": "https://huggingface.co/datasets/clinc/clinc_oos", "attribution": "Larson et al., An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction (EMNLP 2019)",
                 "label_provenance": "crowdsourced utterances written for each intent (published label)", "native_splits": "plus config: development from validation, test from test",
                 "mapping": "state = the utterance; one choice over 10 options: an in-scope utterance gets its intent plus 8 other intents of the same domain (clinc/oos-eval domains.json) plus out_of_scope; an out-of-scope utterance gets 9 intents of one seeded domain plus out_of_scope (always last); strata = the 10 domains and out-of-scope"},
    "sgd": {"area": "retrieval", "di": [10, "SGD/SGD-X"], "metric": "accuracy", "licence": "CC-BY-SA-4.0: the derived records inherit the share-alike term",
            "licence_url": "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue", "attribution": "Rastogi et al., Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset (AAAI 2020); Lee et al., SGD-X (AAAI 2022)",
            "label_provenance": "human-annotated dialogue state (active_intent of the user turn's frame)", "native_splits": "development from dev dialogues, test from test dialogues",
            "mapping": "one user turn per dialogue: state = the service description and the dialogue up to and including that turn; one choice over the service's intents (name: description) plus NONE; the schema wording is the original or one of the five SGD-X paraphrase variants, chosen per dialogue by hash; strata = NONE / intent changed / intent unchanged since the previous user turn; turns with more than one frame are skipped"},
    "bright": {"area": "retrieval", "di": [36, "BRIGHT"], "metric": "accuracy", "licence": "CC-BY-4.0",
               "licence_url": "https://huggingface.co/datasets/xlangai/BRIGHT", "attribution": "Su et al., BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval (ICLR 2025)",
               "label_provenance": "human (StackExchange answers' cited documents, annotated gold chunks)", "native_splits": "examples only; split into halves by query",
               "mapping": "StackExchange domains only (" + ", ".join(BRIGHT_DOMAINS) + "); state = the post; one choice over 5 document chunks: one gold chunk and the 4 highest-BM25 chunks of the same domain that are not gold, not excluded and not from a gold document's page, every option 150-1,500 characters, in a seeded order; strata = domain"},
    "bfcl": {"area": "tools", "di": [1, "BFCL"], "metric": "case_exact", "licence": "Apache-2.0",
             "licence_url": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard", "attribution": "Patil et al., The Berkeley Function Calling Leaderboard (BFCL) (ICML 2025)",
             "label_provenance": "human-verified ground-truth calls (possible_answer); irrelevance categories call nothing", "native_splits": "one set per category; split into halves by item",
             "mapping": f"state = the conversation and every supplied function (name, description, JSON parameters); one noul per function 'Should the assistant call `name`?' (true when the ground truth calls it), at most {MAX_OPTIONS} functions; scored case-exact, as the Index does; strata = the ten single-turn AST categories (simple, multiple, parallel, parallel_multiple, irrelevance and their live_ versions)"},
    "toolret": {"area": "tools", "di": [2, "ToolRet"], "metric": "accuracy", "licence": "Apache-2.0 (ToolRet release); queries and tools re-published from its source benchmarks",
                "licence_url": "https://github.com/mangopy/tool-retrieval-benchmark", "attribution": "Shi et al., Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models (Findings of ACL 2025)",
                "label_provenance": "source benchmarks' target tools (published labels)", "native_splits": "queries only; split into halves by query",
                "mapping": "subsets except " + ", ".join(f"{k} ({v})" for k, v in TOOLRET_EXCLUDED.items()) + "; state = the query; one choice over 5 tools (name: description, description cut to 400 characters): one labelled tool and the 4 highest-BM25 unlabelled tools of the same tool corpus (web / code / customized), in a seeded order; strata = subset"},
    "apibank": {"area": "tools", "di": [3, "API-Bank"], "metric": "accuracy", "licence": "MIT",
                "licence_url": "https://huggingface.co/datasets/liminghao1630/API-Bank", "attribution": "Li et al., API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs (EMNLP 2023)",
                "label_provenance": "human-annotated API call (expected_output)", "native_splits": "level-1 test only; split into halves by dialogue file",
                "mapping": "state = the dialogue so far; one choice over up to 8 APIs (name: description): the APIs the level-1 item itself describes plus the highest-BM25 others from the level-1 catalog, in a seeded order; strata = the reference API"},
    "routerbench": {"area": "tools", "di": [6, "RouterBench"], "metric": "accuracy", "licence": "MIT",
                    "licence_url": "https://github.com/withmartian/routerbench", "attribution": "Hu et al., RouterBench: A Benchmark for Multi-LLM Routing System (2024), Martian",
                    "label_provenance": "recorded correctness of each model's answer (0/1 exact scoring by the RouterBench harness)", "native_splits": "0-shot file only; split into halves by sample",
                    "mapping": "MMLU, Winogrande and MBPP prompts whose 11 scores are all 0/1 (GSM8K is left out because its scores in this file are fractional, HellaSwag for the wikiHow notice, ARC because ARC is in Kev's training data); state = the prompt; one choice 'which of these models answered correctly' over one correct model and up to 9 incorrect ones (drawn in a seeded order), so the answer is unique; strata = source benchmark"},
    "humicroedit": {"area": "arts", "di": [21, "Humicroedit"], "metric": "accuracy", "licence": "none stated (SemEval-2020 Task 7 research release; the README asks users to cite the task paper)",
                    "licence_url": "https://cs.rochester.edu/u/nhossain/humicroedit.html", "attribution": "Hossain et al., SemEval-2020 Task 7: Assessing Humor in Edited News Headlines (SemEval 2020)",
                    "label_provenance": "human (crowd funniness grades; label = the edit with the higher mean grade)", "native_splits": "subtask-2 dev.csv -> development, test.csv -> test",
                    "mapping": "state = the original headline; one choice between the two edited headlines (a / b), ties dropped; one pair per headline; strata = label"},
    "cfcolor": {"area": "arts", "di": [23, "cfcolor"], "metric": "accuracy", "licence": "release notice: 'Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying programs and documents for any purpose' (runPrediction.m), no separate data licence",
                "licence_url": "https://www.dgp.toronto.edu/~donovan/cfcolor/", "attribution": "O'Donovan, Agarwala and Hertzmann, Collaborative Filtering of Color Aesthetics (CAe 2014)",
                "label_provenance": "human (Mechanical Turk 1-5 ratings of 5-colour themes)", "native_splits": "one rating set; split into halves by user",
                "mapping": "one record per user: state = 12 of the user's training-set ratings (five hex colours and a 1-5 rating each); one choice between two palettes from the user's test-set ratings with different ratings, which did this person rate higher (a / b); strata = label"},
}
SKIPPED = {"gpqa_diamond": {"area": "knowledge", "di": [25, "GPQA Diamond"],
                            "reason": "gated on the Hub (Idavidrein/gpqa: accepting the dataset's terms is required, and this account has not); not accepted on the owner's behalf. Its terms also ask that examples not be revealed in plain text online, which a public suite would do"}}

# ---------------------------------------------------------------- shared helpers (pure)


def h(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def seeded(seed):
    return random.Random(h(SEED, seed))


def q_choice(instr, criteria, label, src):
    if label not in criteria: raise ValueError(f"label {label!r} not among options ({src})")
    if not 2 <= len(criteria) <= MAX_OPTIONS: raise ValueError(f"{src}: {len(criteria)} options")
    return {"type": "choice", "instructions": instr, "criteria": dict(criteria), "label": label, "src": src}


def q_noul(instr, label, src):
    return {"type": "noul", "instructions": instr, "label": bool(label), "src": src}


def candidate(dataset, cid, state, questions, group, stratum, split=None, **meta):
    x = {"state": state, "questions": questions, "_stratum": stratum,
         "_meta": {"id": f"{dataset}/{cid}", "source": dataset, "area": DATASETS[dataset]["area"], "group_id": f"{dataset}/{group}", "text_sha256": state_key(state), **meta}}
    if split: x["_split"] = split
    return x


def chance(record, metric):
    """Expected score of uniform random answers on one record: a list of 1/K per question (accuracy), or one product over
    its questions (case_exact: every question right by chance)."""
    ks = [2 if q["type"] == "noul" else len(q["criteria"]) for q in record["questions"].values()]
    return [1 / k for k in ks] if metric == "accuracy" else [math.prod(1 / k for k in ks)]


def split_halves(cands, dataset):
    """Datasets without a native development/test split: groups dealt to the two partitions in a seeded hash order,
    balanced by candidate count (deal_groups)."""
    weights = Counter(x["_meta"]["group_id"] for x in cands)
    where = deal_groups(dict(weights), {"test": 0.5, "development": 0.5}, f"{SEED}:{dataset}")
    for x in cands: x["_split"] = where[x["_meta"]["group_id"]]
    return cands


def clean_html(text):
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())


def truncate_words(text, limit):
    text = " ".join((text or "").split())
    if len(text) <= limit: return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


class BM25:
    """Okapi BM25 (k1 1.2, b 0.75) over a fixed corpus with scikit-learn's word tokenizer (English stop words removed).
    top(query, k, skip) returns up to k document indices by descending score, ties by index, scores rounded to 1e-6."""

    def __init__(self, texts, k1=1.2, b=0.75):
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        self.np = np
        self.vec = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w\w+\b", dtype=np.float64)
        tf = self.vec.fit_transform(texts).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        length = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * length / max(length.mean(), 1e-9))
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.weights = tf

    def top(self, query, k, skip=lambda i: False):
        np = self.np
        q = self.vec.transform([query]); q.data[:] = 1.0
        scores = np.round(np.asarray((self.weights @ q.T).todense()).ravel(), 6)
        order = np.lexsort((np.arange(len(scores)), -scores))
        out = []
        for i in order:
            if scores[i] <= 0: break
            if not skip(int(i)): out.append(int(i))
            if len(out) == k: break
        return out

# ---------------------------------------------------------------- per-dataset mappings (pure; tested in tests/test_breadth_v1.py)


def musr_candidate(subset, index, row):
    choices = ast.literal_eval(row["choices"])
    keys = [f"opt_{i + 1}" for i in range(len(choices))]
    q = q_choice(row["question"].strip(), dict(zip(keys, choices)), keys[int(row["answer_index"])], f"musr_{subset}")
    return candidate("musr", f"{subset}/{index}", row["narrative"].strip(), {"answer": q}, f"{subset}/{index}", subset,
                     provenance={"via": f"hf:{MUSR[0]}@{MUSR[1]}:{subset}.csv", "row": index})


def sata_candidate(index, row, rng):
    answers, distractors = ([str(x).strip() for x in (v if isinstance(v, list) else ast.literal_eval(v))] for v in (row["answer groups"], row["distractor groups"]))
    if not answers or len({normalise_text(x) for x in answers + distractors}) != len(answers) + len(distractors): return None
    if not 2 <= len(answers) + len(distractors) <= MAX_OPTIONS: return None
    options = [(x, True) for x in answers] + [(x, False) for x in distractors]
    rng.shuffle(options)
    question = re.sub(r"^Multi Label Question:\s*", "", clean_html(row["question"]))
    passage = re.sub(r"^Paragraph:\s*", "", clean_html(row["paragraph"]))
    qs = {f"cand_{i + 1}": q_noul(f'Is "{text}" a correct answer to the question?', ok, "sata_bench") for i, (text, ok) in enumerate(options)}
    state = {"passage": passage, "question": question}
    return candidate("sata_bench", index, state, qs, text_digest(question + passage)[:20], f"answers_{min(len(answers), 3)}",
                     provenance={"via": f"hf:{SATA[0]}@{SATA[1]}:{SATA[2]}", "row": index})


PIECES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def fen_board(fen):
    """{square: piece letter} from the placement field of a FEN (uppercase = white)."""
    board = {}
    for r, rank in enumerate(fen.split()[0].split("/")):
        f = 0
        for c in rank:
            if c.isdigit(): f += int(c)
            else: board[f"{'abcdefgh'[f]}{8 - r}"] = c; f += 1
    return board


def ascii_board(fen):
    board = fen_board(fen)
    return "\n".join(f"{rank} " + " ".join(board.get(f"{f}{rank}", ".") for f in "abcdefgh") for rank in range(8, 0, -1)) + "\n  a b c d e f g h"


def describe_move(fen, uci):
    board = fen_board(fen)
    src, dst, promo = uci[:2], uci[2:4], uci[4:]
    piece = board.get(src, "?")
    colour = "white" if piece.isupper() else "black"
    text = f"{colour} {PIECES.get(piece.lower(), 'piece')} {src}-{dst}"
    if piece.lower() == "k" and src[0] == "e" and dst[0] in "cg" and src[1] == dst[1]: text += " (castling)"
    elif dst in board: text += f" capturing the {PIECES.get(board[dst].lower(), 'piece')}"
    if promo: text += f", promoting to {PIECES[promo.lower()]}"
    return text


def chess_options(moves, rng, tie=0.01, limit=MAX_OPTIONS):
    """moves: {uci: win probability}. The best move plus up to limit-1 others drawn in a seeded order, leaving out moves
    within `tie` of the best (so the reference is unique); None when fewer than two options remain. Returns (keys in a
    seeded order, best)."""
    best = max(sorted(moves), key=lambda m: moves[m])
    others = sorted(m for m in moves if m != best and moves[m] <= moves[best] - tie)
    if not others: return None
    rng.shuffle(others)
    keys = [best] + others[:limit - 1]
    rng.shuffle(keys)
    return keys, best


def chess_phase(fen):
    n = sum(1 for c in fen.split()[0] if c.isalpha())
    return "opening" if n >= 26 else "middlegame" if n >= 13 else "endgame"


def chess_candidate(fen, moves, rng):
    picked = chess_options(moves, rng)
    if picked is None: return None
    keys, best = picked
    state = {"fen": fen, "to_move": "white" if fen.split()[1] == "w" else "black", "board": ascii_board(fen)}
    q = q_choice("Which move is strongest for the side to move?", {m: describe_move(fen, m) for m in keys}, best, "chessbench")
    return candidate("chessbench", h(fen)[:16], state, {"best_move": q}, h(fen)[:16], chess_phase(fen),
                     values={m: round(moves[m], 6) for m in keys}, provenance={"via": CHESS["url"], "fen": fen})


def read_varint(buf, i):
    shift = value = 0
    while True:
        b = buf[i]; i += 1
        value |= (b & 0x7F) << shift; shift += 7
        if not b & 0x80: return value, i


def decode_action_value(record):
    """(fen, move, win_prob) from one searchless_chess action-value record: apache_beam TupleCoder of StrUtf8Coder,
    StrUtf8Coder (varint-length-prefixed in the nested context) and FloatCoder (8-byte big-endian double)."""
    n, i = read_varint(record, 0); fen = record[i:i + n].decode(); i += n
    n, i = read_varint(record, i); move = record[i:i + n].decode(); i += n
    return fen, move, struct.unpack(">d", record[i:i + 8])[0]


def bag_records(path):
    """Records of an uncompressed bagz file (searchless_chess src/bagz.py): the records, then one little-endian int64 end
    offset per record; the last offset, the file's final 8 bytes, is also where the offsets start."""
    data = Path(path).read_bytes()
    (start,) = struct.unpack("<Q", data[-8:])
    ends = struct.unpack(f"<{(len(data) - start) // 8}q", data[start:])
    prev = 0
    for end in ends:
        yield data[prev:end]; prev = end


CNLI_LABELS = {"Entailment": "entailment", "Contradiction": "contradiction", "NotMentioned": "not_mentioned"}
CNLI_CRITERIA = {"entailment": "The contract entails the statement", "contradiction": "The contract contradicts the statement",
                 "not_mentioned": "The contract does not mention it"}


def contractnli_questions(annotations, hypotheses, rng, k=CONTRACTNLI_PER_DOC):
    """Up to k hypotheses of one document, round-robin over its labels (sorted), each label's hypotheses in a seeded order."""
    classes = defaultdict(list)
    for key in sorted(annotations):
        classes[CNLI_LABELS[annotations[key]["choice"]]].append(key)
    for v in classes.values(): rng.shuffle(v)
    chosen = round_robin(classes, k, lambda _: True)
    return {key: q_choice(f'Statement: "{hypotheses[key]}" Does the contract entail, contradict or not mention this statement?', CNLI_CRITERIA,
                          CNLI_LABELS[annotations[key]["choice"]], "contractnli") for key in sorted(chosen)}


def hellaswag_candidate(index, row):
    keys = [f"opt_{i + 1}" for i in range(len(row["endings"]))]
    q = q_choice("Which ending most plausibly continues the scene?", dict(zip(keys, row["endings"])), keys[int(row["label"])], "hellaswag")
    return candidate("hellaswag", f"val/{row['ind']}", {"activity": row["activity_label"], "context": row["ctx"]}, {"ending": q}, row["source_id"], row["split_type"],
                     provenance={"via": f"hf:{HELLASWAG[0]}@{HELLASWAG[1]}:{HELLASWAG[2]}", "row": index, "source_id": row["source_id"]})


OOS = "out_of_scope"


def clinc_options(gold, domains, rng):
    """(criteria, label) for one CLINC150 utterance: 9 intents + out_of_scope (last). In scope: the gold intent and 8
    others of its domain; out of scope (gold 'oos'): 9 intents of a seeded domain."""
    domain_of = {i: d for d, intents in domains.items() for i in intents}
    if gold == "oos":
        intents = rng.sample(sorted(domains[rng.choice(sorted(domains))]), 9)
    else:
        intents = rng.sample(sorted(i for i in domains[domain_of[gold]] if i != gold), 8) + [gold]
    rng.shuffle(intents)
    criteria = {**{i: None for i in intents}, OOS: "The request fits none of the listed intents"}
    return criteria, OOS if gold == "oos" else gold


def clinc_candidate(split, index, text, gold, domains, rng):
    criteria, label = clinc_options(gold, domains, rng)
    domain = "oos" if gold == "oos" else next(d for d, v in domains.items() if gold in v)
    q = q_choice("Which intent does this request express?", criteria, label, "clinc150")
    return candidate("clinc150", f"{split}/{index}", text, {"intent": q}, f"{split}/{index}", domain, split=split,
                     provenance={"via": f"hf:{CLINC[0]}@{CLINC[1]}:{CLINC[2][split]}", "row": index})


SGD_VARIANTS = ("original", "v1", "v2", "v3", "v4", "v5")
SGD_NONE = "NONE"


def sgd_variant_services(original, variant):
    """{original service name: variant service} by position; the SGD-X variant schemas list the same services and intents in
    the same order under paraphrased names."""
    if len(original) != len(variant): raise ValueError("SGD-X variant lists a different number of services")
    out = {}
    for o, v in zip(original, variant):
        if not v["service_name"].startswith(o["service_name"]) or len(o["intents"]) != len(v["intents"]):
            raise ValueError(f"SGD-X variant does not line up at {o['service_name']}")
        out[o["service_name"]] = v
    return out


def sgd_turns(dialogue):
    """(turn index, service, active intent, stratum, history) for every single-frame user turn. Stratum: none (no active
    intent), changed (first intent of the service or different from its previous user turn), same."""
    history, previous, out = [], {}, []
    for t, turn in enumerate(dialogue["turns"]):
        history.append(f"{'User' if turn['speaker'] == 'USER' else 'System'}: {turn['utterance']}")
        if turn["speaker"] != "USER": continue
        for frame in turn["frames"]:
            intent, service = frame["state"]["active_intent"], frame["service"]
            if len(turn["frames"]) == 1:
                stratum = "none" if intent == SGD_NONE else "changed" if previous.get(service) != intent else "same"
                out.append((t, service, intent, stratum, list(history)))
            previous[service] = intent
    return out


def sgd_candidate(split, dialogue, turn, schemas):
    t, service, intent, stratum, history = turn
    variant = SGD_VARIANTS[int(h(dialogue["dialogue_id"]), 16) % len(SGD_VARIANTS)]
    original = schemas["original"][service]
    shown = schemas[variant][service]
    names = {o["name"]: v["name"] for o, v in zip(original["intents"], shown["intents"])}
    criteria = {v["name"]: v["description"] for v in shown["intents"]}
    if SGD_NONE in criteria: raise ValueError("intent named NONE")
    criteria[SGD_NONE] = "The user is not pursuing any of these intents right now"
    q = q_choice("Which of this service's intents is the user pursuing right now?", criteria, SGD_NONE if intent == SGD_NONE else names[intent], "sgd")
    state = {"service": shown["description"], "dialogue": history}
    return candidate("sgd", f"{split}/{dialogue['dialogue_id']}/{t}", state, {"intent": q}, f"{split}/{dialogue['dialogue_id']}", stratum, split=split,
                     schema_variant=variant, provenance={"via": f"github:google-research-datasets/dstc8-schema-guided-dialogue@{SGD['commit']}:{split}", "dialogue_id": dialogue["dialogue_id"], "turn": t})


def bfcl_candidate(category, item, answers):
    functions = item["function"]
    if not 1 <= len(functions) <= MAX_OPTIONS: return None
    names = [f["name"] for f in functions]
    if len(set(names)) != len(names): return None
    called = {name for call in answers for name in call} if answers is not None else set()
    if not called <= set(names): return None
    conversation = [f"{m['role']}: {m['content']}" for m in item["question"][0]]
    state = {"conversation": conversation,
             "available_functions": [{"name": f["name"], "description": f.get("description", ""), "parameters": json.dumps(f.get("parameters", {}), ensure_ascii=False, sort_keys=True)} for f in functions]}
    qs = {f"call_{i + 1}": q_noul(f"Should the assistant call the function `{name}` for this request?", name in called, f"bfcl_{category}") for i, name in enumerate(names)}
    return candidate("bfcl", item["id"], state, qs, item["id"], category, provenance={"via": f"hf:{BFCL[0]}@{BFCL[1]}:BFCL_v3_{category}.json", "id": item["id"]})


def tool_summary(documentation, limit=400):
    """(name, description) of a ToolRet tool document (a JSON string in one of several source schemas)."""
    try: doc = json.loads(documentation)
    except (TypeError, ValueError): doc = {"description": str(documentation)}
    if not isinstance(doc, dict): doc = {"description": json.dumps(doc, ensure_ascii=False)}
    name = str(doc.get("name") or doc.get("api_name") or doc.get("tool_name") or doc.get("api_call") or "tool").strip()
    parts = [str(doc[k]) for k in ("functionality", "description", "api_description", "tool_description") if doc.get(k)]
    return name, truncate_words(" ".join(parts) or json.dumps(doc, ensure_ascii=False, sort_keys=True), limit)


def retrieval_choice(dataset, instr, gold_text, negative_texts, rng, src, prefix):
    """A choice over the gold text and negatives with neutral keys in a seeded order; returns (criteria, label)."""
    texts = [(gold_text, True)] + [(t, False) for t in negative_texts]
    rng.shuffle(texts)
    keys = [f"{prefix}_{i + 1}" for i in range(len(texts))]
    criteria = dict(zip(keys, [t for t, _ in texts]))
    label = keys[[ok for _, ok in texts].index(True)]
    return q_choice(instr, criteria, label, src)


API_REQUEST = re.compile(r"API-Request:\s*\[(\w+)\(")


def apibank_parse(row):
    """(dialogue, described APIs {name: description}, gold API name) of one API-Bank level-1 item."""
    described = {}
    tail = row["instruction"].split("API descriptions:", 1)[1]
    for line in tail.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            api = json.loads(line); described[api["name"]] = api["description"]
    dialogue = row["input"].strip()
    dialogue = re.sub(r"\s*Generate API Request:\s*$", "", dialogue)
    m = API_REQUEST.search(row["expected_output"])
    return dialogue, described, m.group(1) if m else None


def router_options(scores, rng, limit=MAX_OPTIONS):
    """scores: {model: 0 or 1}. One correct model and up to limit-1 incorrect ones, in a seeded order; None unless both
    kinds exist. Returns (keys, label)."""
    right = sorted(m for m, s in scores.items() if s == 1)
    wrong = sorted(m for m, s in scores.items() if s == 0)
    if not right or not wrong: return None
    label = rng.choice(right)
    rng.shuffle(wrong)
    keys = [label] + wrong[:limit - 1]
    rng.shuffle(keys)
    return keys, label


def router_family(eval_name):
    return "mmlu" if eval_name.startswith("mmlu-") else eval_name


def humicroedit_pair(row):
    """(original headline, edited headline 1, edited headline 2, label 'a'/'b') or None for a tie."""
    if row["label"] not in ("1", "2"): return None
    tag = re.compile(r"<([^>]*)/>")
    original = tag.sub(lambda m: m.group(1), row["original1"])
    one = tag.sub(row["edit1"], row["original1"], count=1)
    two = tag.sub(row["edit2"], row["original2"], count=1)
    if normalise_text(one) == normalise_text(two): return None
    return " ".join(original.split()), " ".join(one.split()), " ".join(two.split()), "a" if row["label"] == "1" else "b"


def palette_hex(rgb15):
    return " ".join("#" + "".join(f"{round(float(c) * 255):02x}" for c in rgb15[i:i + 3]) for i in range(0, 15, 3))


def cfcolor_record(user, train, test, rgb, rng, history=12):
    """One record for a user: `history` of their training ratings as the state, and a pair of test-set palettes with
    different ratings. train/test: lists of (theme index, rating); rgb: theme index -> 15 floats. None if impossible."""
    test = sorted(test, key=lambda x: h(user, x[0]))
    pair = next(((a, b) for i, a in enumerate(test) for b in test[i + 1:] if a[1] != b[1]), None)
    if pair is None: return None
    shown = [x for x in sorted(train, key=lambda x: h(user, "train", x[0])) if x[0] not in (pair[0][0], pair[1][0])][:history]
    if len(shown) < 8: return None
    a, b = pair if rng.random() < 0.5 else pair[::-1]
    state = {"note": "Ratings this person gave to five-colour palettes, from 1 (dislike) to 5 (like).",
             "earlier_ratings": [{"palette": palette_hex(rgb[t]), "rating": int(r)} for t, r in shown]}
    q = q_choice("Which of these two palettes did this person rate higher?", {"a": palette_hex(rgb[a[0]]), "b": palette_hex(rgb[b[0]])},
                 "a" if a[1] > b[1] else "b", "cfcolor")
    return state, q, {"themes": [int(a[0]), int(b[0])], "ratings": [int(a[1]), int(b[1])]}

# ---------------------------------------------------------------- per-dataset candidates (downloads)


def musr(raw, report):
    out = []
    for fname in MUSR[2]:
        csv.field_size_limit(sys.maxsize)
        rows = list(csv.DictReader(io.StringIO(hub_file(MUSR[0], MUSR[1], fname).read_text(encoding="utf-8"), newline="")))
        out += [musr_candidate(fname[:-4], i, r) for i, r in enumerate(rows)]
    report["musr"] = {"candidates": len(out)}
    return split_halves(out, "musr")


def sata(raw, report):
    rows = read_json(hub_file(*SATA))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        x = sata_candidate(i, r, seeded(f"sata/{i}"))
        if x is None: dropped += 1
        else: out.append(x)
    report["sata_bench"] = {"rows": len(rows), "dropped_options": dropped, "candidates": len(out)}
    return split_halves(out, "sata_bench")


def chessbench(raw, report):
    path = fetch(CHESS["url"], raw / CHESS["file"], sha256=CHESS["sha256"])
    positions = defaultdict(dict)
    for rec in bag_records(path):
        fen, move, p = decode_action_value(rec)
        positions[fen][move] = p
    fens = sorted(positions)
    # the pool: a seeded 4,000-position sample (the full file has ~60k positions; selection needs 2 x EVAL_RECORDS)
    pool = sorted(fens, key=lambda f: h(SEED, "chess", f))[:4000]
    out = [x for x in (chess_candidate(f, positions[f], seeded(f"chess/{f}")) for f in pool) if x is not None]
    report["chessbench"] = {"positions": len(fens), "pool": len(pool), "candidates": len(out)}
    return split_halves(out, "chessbench")


def contractnli(raw, report):
    path = fetch(CONTRACTNLI["url"], raw / CONTRACTNLI["file"], sha256=CONTRACTNLI["sha256"])
    out = []
    with zipfile.ZipFile(path) as z:
        for split, member in (("development", "contract-nli/dev.json"), ("test", "contract-nli/test.json")):
            data = json.loads(z.read(member))
            hyps = {k: v["hypothesis"] for k, v in data["labels"].items()}
            for doc in data["documents"]:
                qs = contractnli_questions(doc["annotation_sets"][0]["annotations"], hyps, seeded(f"cnli/{doc['id']}"))
                out.append(candidate("contractnli", f"{split}/{doc['id']}", doc["text"].strip(), qs, f"doc/{doc['id']}", doc["document_type"], split=split,
                                     provenance={"via": f"{CONTRACTNLI['url']}:{member}", "document_id": doc["id"], "file_name": doc["file_name"]}))
    report["contractnli"] = {"candidates": len(out)}
    return out


def hellaswag(raw, report):
    rows = parquet_rows(hub_file(*HELLASWAG))
    out = [hellaswag_candidate(i, r) for i, r in enumerate(rows) if r["source_id"].startswith("activitynet~")]
    report["hellaswag"] = {"rows": len(rows), "activitynet": len(out), "wikihow_left_out": len(rows) - len(out)}
    return split_halves(out, "hellaswag")


def clinc150(raw, report):
    domains = read_json(fetch(CLINC_DOMAINS["url"], raw / CLINC_DOMAINS["file"], sha256=CLINC_DOMAINS["sha256"]))
    card = hub_file(CLINC[0], CLINC[1], "README.md").read_text(encoding="utf-8")
    block = card[card.index("config_name: plus"):]
    block = block[:block.index("splits:")]
    names = {int(k): v.strip("'\"") for k, v in re.findall(r"'(\d+)': (\S+)", block)}   # YAML quotes 'yes' / 'no'
    if sorted({i for v in domains.values() for i in v}) != sorted(n for n in names.values() if n != "oos"):
        raise ValueError("CLINC150 domains.json and the card's intents differ")
    out = []
    for split, fname in CLINC[2].items():
        for i, r in enumerate(parquet_rows(hub_file(CLINC[0], CLINC[1], fname))):
            out.append(clinc_candidate(split, i, r["text"], names[r["intent"]], domains, seeded(f"clinc/{split}/{i}")))
    report["clinc150"] = {"candidates": len(out)}
    return out


def sgd(raw, report):
    path = fetch(SGD["url"], raw / SGD["file"], sha256=SGD["sha256"])
    files = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".json") and ("/dev/" in m.name or "/test/" in m.name):
                files[m.name.split("/", 1)[1]] = json.loads(t.extractfile(m).read())
    out, stats = [], Counter()
    for split, folder in (("development", "dev"), ("test", "test")):
        original = files[f"{folder}/schema.json"]
        schemas = {"original": {s["service_name"]: s for s in original}}
        for v in SGD_VARIANTS[1:]:
            schemas[v] = sgd_variant_services(original, files[f"sgd_x/data/{v}/{folder}/schema.json"])
        for name in sorted(f for f in files if f.startswith(f"{folder}/dialogues_")):
            for dialogue in files[name]:
                for turn in sgd_turns(dialogue):
                    out.append(sgd_candidate(split, dialogue, turn, schemas)); stats[split, turn[3]] += 1
    report["sgd"] = {"candidates": len(out), "turns_by_stratum": {f"{s}/{k}": n for (s, k), n in sorted(stats.items())}}
    return out


def bright(raw, report):
    out, stats = [], {}
    for domain in BRIGHT_DOMAINS:
        docs = parquet_rows(hub_file(BRIGHT[0], BRIGHT[1], f"documents/{domain}-00000-of-00001.parquet"))
        ids = [d["id"] for d in docs]; texts = [d["content"] for d in docs]
        index = {d: i for i, d in enumerate(ids)}
        ok_len = [150 <= len(t.strip()) <= 1500 for t in texts]
        bm25 = BM25(texts)
        n = 0
        for ex in parquet_rows(hub_file(BRIGHT[0], BRIGHT[1], f"examples/{domain}-00000-of-00001.parquet")):
            gold = [g for g in sorted(ex["gold_ids"]) if g in index and ok_len[index[g]]]
            if not gold: continue
            banned = set(ex["gold_ids"]) | set(ex["excluded_ids"] or []) | set(ex["gold_ids_long"] or [])
            folders = {g.split("/")[0] for g in ex["gold_ids"]}
            skip = lambda i: ids[i] in banned or ids[i].split("/")[0] in folders or not ok_len[i]
            neg = bm25.top(ex["query"], 4, skip)
            if len(neg) < 4: continue
            q = retrieval_choice("bright", "Which document is most helpful for answering this post?", texts[index[gold[0]]].strip(),
                                 [texts[i].strip() for i in neg], seeded(f"bright/{domain}/{ex['id']}"), "bright", "doc")
            out.append(candidate("bright", f"{domain}/{ex['id']}", ex["query"].strip(), {"document": q}, f"{domain}/{ex['id']}", domain,
                                 gold_id=gold[0], negative_ids=[ids[i] for i in neg], provenance={"via": f"hf:{BRIGHT[0]}@{BRIGHT[1]}:examples/{domain}", "query_id": ex["id"]}))
            n += 1
        stats[domain] = {"documents": len(docs), "candidates": n}
    report["bright"] = {"candidates": len(out), "by_domain": stats}
    return split_halves(out, "bright")


def bfcl(raw, report):
    out, stats = [], {}
    for cat in BFCL_CATEGORIES:
        items = read_jsonl(hub_file(BFCL[0], BFCL[1], f"BFCL_v3_{cat}.json"))
        if "irrelevance" in cat: answers = {it["id"]: None for it in items}
        else: answers = {a["id"]: a["ground_truth"] for a in read_jsonl(hub_file(BFCL[0], BFCL[1], f"possible_answer/BFCL_v3_{cat}.json"))}
        kept = [x for x in (bfcl_candidate(cat, it, answers[it["id"]]) for it in items if it["id"] in answers) if x is not None]
        stats[cat] = {"items": len(items), "without_answer": sum(it["id"] not in answers for it in items), "candidates": len(kept)}; out += kept
    report["bfcl"] = {"candidates": len(out), "by_category": stats}
    return split_halves(out, "bfcl")


def toolret(raw, report):
    corpora = {}
    for cat in ("web", "code", "customized"):
        rows = parquet_rows(hub_file(TOOLRET[2], TOOLRET[3], f"{cat}/tools-00000-of-00001.parquet"))
        summaries = [tool_summary(r["documentation"]) for r in rows]
        corpora[cat] = {"ids": [r["id"] for r in rows], "index": {r["id"]: i for i, r in enumerate(rows)},
                        "text": [f"{n}: {d}" for n, d in summaries], "bm25": BM25([f"{n} {r['documentation']}" for (n, _), r in zip(summaries, rows)])}
    from huggingface_hub import HfApi
    files = sorted(f for f in HfApi().list_repo_files(TOOLRET[0], repo_type="dataset", revision=TOOLRET[1]) if f.endswith(".parquet"))
    out, stats = [], {}
    for f in files:
        subset = f.split("/")[0]
        if subset in TOOLRET_EXCLUDED: continue
        n = 0
        for r in parquet_rows(hub_file(TOOLRET[0], TOOLRET[1], f)):
            corpus = corpora.get(r["category"])
            labels = [l for l in json.loads(r["labels"]) if (l.get("relevance") or 0) > 0]
            gold = sorted(l["id"] for l in labels if corpus and l["id"] in corpus["index"])
            if not gold: continue
            gold_text = corpus["text"][corpus["index"][gold[0]]]
            banned = {l["id"] for l in labels}
            seen_text = {normalise_text(gold_text)}

            def skip(i, corpus=corpus, banned=banned, seen_text=seen_text):
                t = normalise_text(corpus["text"][i])
                if corpus["ids"][i] in banned or t in seen_text: return True
                seen_text.add(t); return False
            neg = corpus["bm25"].top(r["query"], 4, skip)
            if len(neg) < 4: continue
            q = retrieval_choice("toolret", "Which tool would be most useful for handling this request?", gold_text,
                                 [corpus["text"][i] for i in neg], seeded(f"toolret/{r['id']}"), "toolret", "tool")
            out.append(candidate("toolret", r["id"], r["query"].strip(), {"tool": q}, r["id"], subset, gold_id=gold[0],
                                 negative_ids=[corpus["ids"][i] for i in neg], provenance={"via": f"hf:{TOOLRET[0]}@{TOOLRET[1]}:{f}", "tools": f"hf:{TOOLRET[2]}@{TOOLRET[3]}:{r['category']}"}))
            n += 1
        stats[subset] = n
    report["toolret"] = {"candidates": len(out), "by_subset": stats, "excluded_subsets": TOOLRET_EXCLUDED}
    return split_halves(out, "toolret")


def apibank(raw, report):
    rows = read_json(hub_file(*APIBANK))
    parsed = [apibank_parse(r) for r in rows]
    catalog = {}
    for _, described, _ in parsed: catalog.update(described)
    names = sorted(catalog)
    bm25 = BM25([f"{n} {catalog[n]}" for n in names])
    out, dropped = [], 0
    for i, (r, (dialogue, described, gold)) in enumerate(zip(rows, parsed)):
        if gold is None or gold not in described: dropped += 1; continue
        fill = bm25.top(dialogue, 8 - len(described), lambda j: names[j] in described)
        keys = sorted(described) + [names[j] for j in fill]
        seeded(f"apibank/{i}").shuffle(keys)
        q = q_choice("Which API should the assistant call next?", {k: catalog[k] for k in keys}, gold, "apibank")
        out.append(candidate("apibank", f"level-1/{i}", dialogue, {"api": q}, f"file/{r['file']}", gold,
                             provenance={"via": f"hf:{APIBANK[0]}@{APIBANK[1]}:{APIBANK[2]}", "row": i, "file": r["file"]}))
    report["apibank"] = {"rows": len(rows), "dropped": dropped, "catalog": len(names), "candidates": len(out)}
    return split_halves(out, "apibank")


def routerbench(raw, report):
    import pandas as pd
    df = pd.read_pickle(hub_file(*ROUTERBENCH))
    out, c = [], Counter()
    for r in df.to_dict("records"):
        fam = router_family(r["eval_name"])
        if fam not in ("mmlu", "winogrande", "grade-school-math", "mbpp"): c["eval_left_out"] += 1; continue   # grade-school-math then fails the 0/1 check
        scores = {m: r[m] for m in ROUTER_MODELS}
        if any(s not in (0, 1, 0.0, 1.0) for s in scores.values()): c["non_binary"] += 1; continue
        picked = router_options({m: int(s) for m, s in scores.items()}, seeded(f"router/{r['sample_id']}"))
        if picked is None: c["all_same"] += 1; continue
        keys, label = picked
        try: prompt = "\n\n".join(str(x) for x in ast.literal_eval(r["prompt"]))
        except (ValueError, SyntaxError): prompt = str(r["prompt"])
        q = q_choice("Which of these models answered this prompt correctly?", {k: None for k in keys}, label, "routerbench")
        out.append(candidate("routerbench", r["sample_id"], prompt.strip(), {"model": q}, r["sample_id"], fam,
                             eval_name=r["eval_name"], correct_models=sorted(m for m, s in scores.items() if int(s) == 1),
                             provenance={"via": f"hf:{ROUTERBENCH[0]}@{ROUTERBENCH[1]}:{ROUTERBENCH[2]}", "sample_id": r["sample_id"]}))
    report["routerbench"] = dict(c, rows=len(df), candidates=len(out))
    return split_halves(out, "routerbench")


def humicroedit(raw, report):
    path = fetch(HUMICROEDIT["url"], raw / HUMICROEDIT["file"], sha256=HUMICROEDIT["sha256"])
    out, c = [], Counter()
    with zipfile.ZipFile(path) as z:
        for split, member in (("development", "semeval-2020-task-7-dataset/subtask-2/dev.csv"), ("test", "semeval-2020-task-7-dataset/subtask-2/test.csv")):
            for i, r in enumerate(csv.DictReader(io.TextIOWrapper(z.open(member), encoding="utf-8"))):
                pair = humicroedit_pair(r)
                if pair is None: c["tie_or_identical"] += 1; continue
                original, one, two, label = pair
                q = q_choice("Which edited headline did readers rate funnier?", {"a": one, "b": two}, label, "humicroedit")
                out.append(candidate("humicroedit", f"{split}/{r['id']}", {"original_headline": original}, {"funnier": q}, f"headline/{r['id'].split('-')[0]}", label, split=split,
                                     grades=[r["meanGrade1"], r["meanGrade2"]], provenance={"via": f"{HUMICROEDIT['url']}:{member}", "id": r["id"]}))
    report["humicroedit"] = dict(c, candidates=len(out))
    return out


def cfcolor(raw, report):
    import scipy.io as sio
    path = fetch(CFCOLOR["url"], raw / CFCOLOR["file"], sha256=CFCOLOR["sha256"])
    with zipfile.ZipFile(path) as z:
        ratings = sio.loadmat(io.BytesIO(z.read("release/allMTurkRatings.mat")), squeeze_me=True)
        themes = sio.loadmat(io.BytesIO(z.read("release/themeData.mat")), squeeze_me=True, struct_as_record=False)["datapoints"]
    rgb = {i + 1: [float(v) for v in row] for i, row in enumerate(themes.rgb)}    # rating files index themes from 1
    train, test = defaultdict(list), defaultdict(list)
    for u, t, r in ratings["train_vec"].tolist(): train[int(u)].append((int(t), int(r)))
    for u, t, r in ratings["test_vec"].tolist(): test[int(u)].append((int(t), int(r)))
    out = []
    for user in sorted(set(train) & set(test)):
        rec = cfcolor_record(user, train[user], test[user], rgb, seeded(f"cfcolor/{user}"))
        if rec is None: continue
        state, q, meta = rec
        out.append(candidate("cfcolor", f"user/{user}", state, {"higher": q}, f"user/{user}", q["label"], **meta,
                             provenance={"via": f"{CFCOLOR['url']}:release/allMTurkRatings.mat (train_vec history, test_vec pair)", "user": user}))
    report["cfcolor"] = {"users": len(set(train) | set(test)), "candidates": len(out)}
    return split_halves(out, "cfcolor")


BUILDERS = {"musr": musr, "sata_bench": sata, "chessbench": chessbench, "contractnli": contractnli, "hellaswag": hellaswag, "clinc150": clinc150,
            "sgd": sgd, "bright": bright, "bfcl": bfcl, "toolret": toolret, "apibank": apibank, "routerbench": routerbench, "humicroedit": humicroedit, "cfcolor": cfcolor}
PER_GROUP = {"sgd": 1, "apibank": 2}      # records per group within a partition (one turn per dialogue; at most two turns per API-Bank dialogue)

# ---------------------------------------------------------------- selection


def target(dataset):
    return CONTRACTNLI_DOCS if dataset == "contractnli" else EVAL_RECORDS


def select(dataset, pool, n, admit, split):
    """Round-robin over the pool's strata (sorted), each stratum in a seeded order."""
    rng = seeded(f"select/{dataset}/{split}")
    items = sorted(pool, key=lambda x: x["_meta"]["id"])
    rng.shuffle(items)
    classes = defaultdict(list)
    for x in items: classes[x["_stratum"]].append(x)
    return round_robin(classes, n, admit)


def build(raw, tok, only=None):
    report, parts = {}, {s: [] for s in PARTITIONS}
    seen, rejected = set(), Counter()
    fit_cache = {}
    for dataset, fn in BUILDERS.items():
        if only and dataset not in only: continue
        cands = fn(raw, report)
        kept, keys = [], set()
        for x in sorted(cands, key=lambda x: x["_meta"]["id"]):      # one candidate per normalised state within a dataset (first by id)
            if x["_meta"]["text_sha256"] in keys: continue
            keys.add(x["_meta"]["text_sha256"]); kept.append(x)
        for split in ("test", "development"):     # test first: a state shared across partitions stays in test
            pool = [x for x in kept if x["_split"] == split]
            groups = Counter()

            def admit(x):
                k = x["_meta"]["text_sha256"]
                if k in seen: rejected[dataset, "duplicate_state"] += 1; return False
                if groups[x["_meta"]["group_id"]] >= PER_GROUP.get(dataset, 10**9): rejected[dataset, "group_limit"] += 1; return False
                if not line_safe(x): rejected[dataset, "line_separator"] += 1; return False
                rid = x["_meta"]["id"]
                if rid not in fit_cache: fit_cache[rid] = fits(materialize(x), tok, **ADMISSION)
                if not fit_cache[rid]: rejected[dataset, "context"] += 1; return False
                seen.add(k); groups[x["_meta"]["group_id"]] += 1
                return True
            chosen = select(dataset, pool, target(dataset), admit, split)
            if len(chosen) < target(dataset): raise SystemExit(f"{dataset}/{split}: only {len(chosen)} of {target(dataset)} records")
            parts[split] += chosen
        report[dataset]["pool"] = {s: sum(1 for x in kept if x["_split"] == s) for s in PARTITIONS}
        print(f"{dataset}: {report[dataset]}", flush=True)
    for split, recs in parts.items():
        for x in recs:
            stratum = x.pop("_stratum"); x.pop("_split", None)
            x["_meta"].update(stratum=stratum, variant="clean", split=split)
            x["_meta"]["row_sha256"] = hashlib.sha256(json.dumps({"state": x["state"], "questions": x["questions"]}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        random.Random(f"{SEED}:shuffle:{split}").shuffle(recs)
    report["rejected"] = {f"{d}/{k}": n for (d, k), n in sorted(rejected.items())}
    return parts, report


def check_invariants(parts):
    """Groups never span partitions; no normalised state or id appears twice; every question has 2..MAX_OPTIONS options."""
    where, keys, ids = {}, set(), set()
    for split, recs in parts.items():
        for r in recs:
            g = r["_meta"]["group_id"]
            if where.setdefault(g, split) != split: raise AssertionError(f"group {g} spans {where[g]} and {split}")
            if r["_meta"]["text_sha256"] in keys: raise AssertionError(f"duplicate state in {r['_meta']['id']}")
            if r["_meta"]["id"] in ids: raise AssertionError(f"duplicate id {r['_meta']['id']}")
            keys.add(r["_meta"]["text_sha256"]); ids.add(r["_meta"]["id"])
            for q in r["questions"].values():
                if q["type"] == "choice" and not 2 <= len(q["criteria"]) <= MAX_OPTIONS: raise AssertionError(f"{r['_meta']['id']}: option count")


def summarise(parts, tok):
    counts, questions, strata, lengths, chances = {}, {}, {}, {}, {}
    for split, recs in parts.items():
        by = defaultdict(list)
        for r in recs: by[r["_meta"]["source"]].append(r)
        counts[split] = {d: len(v) for d, v in sorted(by.items())}
        questions[split] = {d: sum(len(r["questions"]) for r in v) for d, v in sorted(by.items())}
        strata[split] = {d: dict(sorted(Counter(r["_meta"]["stratum"] for r in v).items())) for d, v in sorted(by.items())}
        chances[split] = {d: round(float(sum(c for r in v for c in chance(r, DATASETS[d]["metric"])) / sum(len(chance(r, DATASETS[d]["metric"])) for r in v)), 6) for d, v in sorted(by.items())}
        lengths[split] = {}
        for d, v in sorted(by.items()):
            n = sorted(len(user_tokens(tok, materialize(r)["state"])) + 1 for r in v)
            lengths[split][d] = {"min": n[0], "median": n[len(n) // 2], "p95": n[int(0.95 * (len(n) - 1))], "max": n[-1]}
    return counts, questions, strata, lengths, chances


def pins():
    """Where each dataset came from: Hub revision + sha256 of every file read, or URL + sha256."""
    hub = lambda repo, rev, files: {"hf": f"{repo}@{rev}", "files": {f: digest(hub_file(repo, rev, f)) for f in files}}
    url = lambda spec: {"url": spec["url"], "sha256": spec["sha256"]}
    from huggingface_hub import HfApi
    toolret_files = sorted(f for f in HfApi().list_repo_files(TOOLRET[0], repo_type="dataset", revision=TOOLRET[1]) if f.endswith(".parquet") and f.split("/")[0] not in TOOLRET_EXCLUDED)
    return {"musr": hub(MUSR[0], MUSR[1], MUSR[2]), "sata_bench": hub(*SATA[:2], [SATA[2]]), "chessbench": {**url(CHESS), "code": CHESS["code"]},
            "contractnli": url(CONTRACTNLI), "hellaswag": hub(*HELLASWAG[:2], [HELLASWAG[2]]),
            "clinc150": {**hub(CLINC[0], CLINC[1], sorted(CLINC[2].values()) + ["README.md"]), "domains": url(CLINC_DOMAINS)},
            "sgd": {**url(SGD), "commit": SGD["commit"]},
            "bright": hub(*BRIGHT, [f"{k}/{d}-00000-of-00001.parquet" for d in BRIGHT_DOMAINS for k in ("examples", "documents")]),
            "bfcl": hub(*BFCL, [f"BFCL_v3_{c}.json" for c in BFCL_CATEGORIES] + [f"possible_answer/BFCL_v3_{c}.json" for c in BFCL_CATEGORIES if "irrelevance" not in c]),
            "toolret": {"queries": hub(TOOLRET[0], TOOLRET[1], toolret_files), "tools": hub(TOOLRET[2], TOOLRET[3], [f"{c}/tools-00000-of-00001.parquet" for c in ("web", "code", "customized")])},
            "apibank": hub(*APIBANK[:2], [APIBANK[2]]), "routerbench": hub(*ROUTERBENCH[:2], [ROUTERBENCH[2]]),
            "humicroedit": url(HUMICROEDIT), "cfcolor": url(CFCOLOR)}


def decision_index_reference():
    """The Index's own area, metric and chance for each dataset, from its methodology bundle (for the manifest)."""
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi().space_info(DI_SPACE[0]).sha
    m = read_json(hf_hub_download(DI_SPACE[0], DI_SPACE[1], repo_type="space", revision=revision))
    chance_of = {c["id"]: c["chance"] for c in m["index"]["chance_levels"]}
    bench = {b["id"]: b for b in m["benchmarks"]}
    rows = {name: {"id": spec["di"][0], "name": spec["di"][1], "area": bench[spec["di"][0]]["area"], "metric": bench[spec["di"][0]]["metric"],
                   "chance": chance_of.get(spec["di"][0]), "subset_rule": bench[spec["di"][0]]["subset_rule"]} for name, spec in {**DATASETS, **SKIPPED}.items()}
    for name, row in rows.items():
        expected = {**DATASETS, **SKIPPED}[name]["area"]
        if row["area"] != expected: raise ValueError(f"{name}: the Index puts it in {row['area']}, the table in {expected}")
    return {"space": f"{DI_SPACE[0]}@{revision}", "file": DI_SPACE[1], "edition": m["edition"], "panel_id": m["index"]["panel_id"],
            "formula": "per benchmark k = clip((score - chance) / (1 - chance)) on the coverage-adjusted score (unanswered = wrong); area = mean of its benchmarks; index = 100 x mean of the five areas",
            "datasets": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="suite directory to create, e.g. evals/breadth-v1")
    ap.add_argument("--raw", default="/tmp/breadth-raw", help="download directory for the non-Hub files (never in git)")
    ap.add_argument("--only", help="comma-separated datasets (a dry run: prints counts, writes nothing)")
    a = ap.parse_args()
    out, raw = Path(a.out), Path(a.raw)
    raw.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    tok = load_tokenizer(*TOKENIZER)
    only = set(a.only.split(",")) if a.only else None
    parts, report = build(raw, tok, only)
    check_invariants(parts)
    counts, questions, strata, lengths, chances = summarise(parts, tok)
    if only:
        print(json.dumps({"counts": counts, "questions": questions, "strata": strata, "state_tokens": lengths, "chance": chances}, indent=1)); return
    out.mkdir(parents=True)
    files = {}
    for split in PARTITIONS:
        path = out / f"{split}.jsonl"
        write_jsonl(path, parts[split])
        if read_jsonl(path) != parts[split]: raise AssertionError(f"{path} does not round-trip through kev.suite.read_jsonl")
        files[path.name] = {"sha256": digest(path), "records": len(parts[split]), "questions": sum(len(r["questions"]) for r in parts[split]),
                            "bytes": path.stat().st_size, "in_git": path.stat().st_size <= GIT_LIMIT, "by_source": counts[split]}
    source_pins = pins()
    write_json(out / "manifest.json", {
        "version": VERSION, "seed": SEED, "eval_only": True, "partitions": list(PARTITIONS), "locked": ["test"], "files": files,
        "trainable_sources": [], "eval_only_sources": list(DATASETS), "holdout_sources": [],
        "areas": {area: {"label": label, "datasets": [d for d, s in DATASETS.items() if s["area"] == area]} for area, label in AREAS.items()},
        "datasets": {d: {**s, "pins": source_pins[d], "chance": {sp: chances[sp][d] for sp in PARTITIONS}} for d, s in DATASETS.items()},
        "skipped": SKIPPED, "decision_index": decision_index_reference(),
        "scoring": {"helper": "scripts/breadth_report.py", "metrics": {"accuracy": "fraction of questions answered right", "case_exact": "fraction of records with every question right"},
                    "chance": "expected score of uniform random answers: mean of 1/K over questions (accuracy), mean over records of the product of 1/K (case_exact)",
                    "skill": "clip((score - chance) / (1 - chance)) with unanswered questions scored wrong; area = mean of its datasets; index = 100 x mean of the five areas",
                    "calibration": "ECE (10 bins) and Brier on the returned probabilities of answered questions, per dataset, area and pooled"},
        "counts": counts, "questions": questions, "strata": strata, "state_tokens": lengths,
        "tokenizer": {"model": TOKENIZER[0], "revision": TOKENIZER[1]}, "base_revisions": {TOKENIZER[0]: TOKENIZER[1]},
        "context": {**SERVING_CONTEXT, "admission": ADMISSION,
                    "note": "records are admitted at `admission` under the tokenizer above (row = state + one question <= 7,680 tokens) and scored in the serving context; far longer than Kev's 384 / 1,024 training context"},
        "selection": f"development and test {EVAL_RECORDS} records per dataset ({CONTRACTNLI_DOCS} documents for contractnli); native dev/test splits kept where they exist, otherwise groups dealt to halves in a seeded hash order; "
                     "round-robin over each dataset's strata in a seeded order; normalised-text state dedupe across all datasets and both partitions (test first); no group spans the partitions",
        "label_protocol": "no LLM labels: every label is the source dataset's own (human, engine, recorded execution or by construction); candidate subsets always contain the reference answer",
        "build_report": report, "code_sha256": digest(Path(__file__)),
        "large_partitions": "partitions with in_git false are gitignored; upload them to the Hub mirror (kev.suite.SUITES_DATASET) and bump SUITES_REVISION before load_split can fetch them elsewhere",
    })
    print(json.dumps({"counts": counts, "questions": questions, "files": {k: {kk: v[kk] for kk in ("records", "bytes", "in_git")} for k, v in files.items()}}, indent=1))


if __name__ == "__main__":
    main()
