"""documents-v1 candidates (PLAN_27b B2, at git tag research-archive-2026-09-24): real consumer-complaint narratives
with the complainant's own product and issue labels, stratified by product and length, split by text. Writes unlabelled-by-AI candidates; scripts/label_documents_v1.py
checks the labels and scripts/freeze_documents_v1.py writes the frozen suite.

    uv run python scripts/build_documents_v1.py --out runs/documents-v1-work/candidates

Source: the US CFPB consumer complaint database (a US government work, public domain; narratives are published with the
consumer's consent and scrubbed of personal information, "XXXX"), via the pinned Hugging Face snapshot below, because the
current CFPB export no longer carries narratives. Each document gets two Choice questions whose labels are what the
consumer selected when filing: the kind of product (9 canonical classes) and the main issue (the canonical issues of
that product, merged across CFPB's naming changes, `ISSUES`). Length buckets are by characters: short < 1,200, medium 1,200-4,000, long 4,000-28,000 (about 1k-7k tokens).
"""
import argparse, random, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import digest, text_digest, write_json, write_jsonl  # noqa: E402

REPO, REVISION = "davidheineman/consumer-finance-complaints-large", "44cfa170a402e254407470275ce05d7dcaccde30"
PRODUCTS = {   # canonical key: (description shown as the option, raw CFPB product names over the years)
    "credit_reporting": ("Credit reports, credit scores, or credit repair", ["Credit reporting, credit repair services, or other personal consumer reports", "Credit reporting or other personal consumer reports", "Credit reporting"]),
    "debt_collection": ("Collection of a debt", ["Debt collection"]),
    "mortgage": ("A mortgage or home loan", ["Mortgage"]),
    "bank_account": ("A checking or savings account", ["Checking or savings account", "Bank account or service"]),
    "card": ("A credit card or prepaid card", ["Credit card or prepaid card", "Credit card", "Prepaid card"]),
    "student_loan": ("A student loan", ["Student loan"]),
    "money_transfer": ("A money transfer, money service, or virtual currency", ["Money transfer, virtual currency, or money service", "Money transfers", "Virtual currency"]),
    "vehicle_loan": ("A vehicle loan or lease", ["Vehicle loan or lease"]),
    "personal_loan": ("A payday, title, or personal loan", ["Payday loan, title loan, or personal loan", "Payday loan, title loan, personal loan, or advance loan", "Payday loan"]),
}
RAW = {raw: key for key, (_, raws) in PRODUCTS.items() for raw in raws}
ISSUES = {     # canonical issue options per product: merged across CFPB's naming changes; catch-all and merged-era issues left out
    "credit_reporting": {"incorrect_information": ["Incorrect information on your report", "Incorrect information on credit report"], "improper_use": ["Improper use of your report"],
                         "investigation": ["Problem with a credit reporting company's investigation into an existing problem", "Problem with a company's investigation into an existing problem", "Credit reporting company's investigation"],
                         "unable_to_get_report": ["Unable to get your credit report or credit score"], "fraud_alerts_or_freezes": ["Problem with fraud alerts or security freezes"]},
    "debt_collection": {"debt_not_owed": ["Attempts to collect debt not owed", "Cont'd attempts collect debt not owed"], "written_notification": ["Written notification about debt", "Disclosure verification of debt"],
                        "false_statements": ["False statements or representation"], "communication_tactics": ["Communication tactics"],
                        "negative_or_legal_action": ["Took or threatened to take negative or legal action", "Taking/threatening an illegal action"]},
    "mortgage": {"payment_process": ["Trouble during payment process", "Loan servicing, payments, escrow account"], "struggling_to_pay": ["Struggling to pay mortgage", "Loan modification,collection,foreclosure"],
                 "applying": ["Applying for a mortgage or refinancing an existing mortgage", "Application, originator, mortgage broker"], "closing": ["Closing on a mortgage", "Settlement process and costs"]},
    "bank_account": {"managing": ["Managing an account"], "closing": ["Closing an account"], "opening": ["Opening an account"],
                     "charged_by_another_company": ["Problem with a lender or other company charging your account"], "low_funds": ["Problem caused by your funds being low", "Problems caused by my funds being low"]},
    "card": {"purchase_on_statement": ["Problem with a purchase shown on your statement"], "fees_or_interest": ["Fees or interest"], "getting_a_card": ["Getting a credit card"],
             "making_payments": ["Problem when making payments"], "closing_the_account": ["Closing your account"]},
    "student_loan": {"lender_or_servicer": ["Dealing with your lender or servicer", "Dealing with my lender or servicer"], "struggling_to_repay": ["Struggling to repay your loan", "Can't repay my loan"],
                     "getting_a_loan": ["Getting a loan"]},
    "money_transfer": {"fraud_or_scam": ["Fraud or scam"], "money_not_available": ["Money was not available when promised"], "mobile_wallet_account": ["Managing, opening, or closing your mobile wallet account"],
                       "unauthorized_transactions": ["Unauthorized transactions or other transaction problem"]},
    "vehicle_loan": {"managing": ["Managing the loan or lease"], "end_of_loan": ["Problems at the end of the loan or lease"], "struggling_to_pay": ["Struggling to pay your loan"],
                     "getting_a_loan": ["Getting a loan or lease"], "repossession": ["Repossession"]},
    "personal_loan": {"unexpected_fees": ["Charged fees or interest you didn't expect", "Charged fees or interest I didn't expect"], "struggling_to_pay": ["Struggling to pay your loan"],
                      "making_payments": ["Problem when making payments"], "payoff": ["Problem with the payoff process at the end of the loan"], "getting_the_loan": ["Getting the loan"]},
}
ISSUE_OF = {(product, raw): key for product, table in ISSUES.items() for key, raws in table.items() for raw in raws}
BUCKETS = (("short", 200, 1200), ("medium", 1200, 4000), ("long", 4000, 28000))
SPLITS = {"test": 22, "development": 22, "train": 222}      # documents per (product, bucket) cell
RIGHTS = "US government work (public domain)"


def prepare(row):
    """A source row with the fields a candidate is built from: the narrative "text", the canonical product "key", the
    length "bucket" and "text_sha256" (whitespace- and case-insensitive). None when the row has no narrative or an unknown
    product; "bucket" and "text_sha256" are None when the narrative is outside every length bucket."""
    text, key = (row["complaint_what_happened"] or "").strip(), RAW.get(row["product"])
    if not text or key is None: return None
    bucket = next((b for b, lo, hi in BUCKETS if lo <= len(text) < hi), None)
    return {**row, "text": text, "key": key, "bucket": bucket, "text_sha256": text_digest(text) if bucket else None}


def questions(key, issue):
    """The Choice questions of one document: its product (always) and, when its raw issue maps to a canonical one, the
    issue among that product's canonical issues. The labels are the consumer's own selections."""
    qs = {"product": {"type": "choice", "instructions": "Which kind of financial product is this complaint about?",
                      "criteria": {k: d for k, (d, _) in PRODUCTS.items()}, "label": key, "src": "cfpb_product"}}
    if issue:
        qs["issue"] = {"type": "choice", "instructions": "What is the main problem the consumer describes?",
                       "criteria": {k: raws[0] for k, raws in ISSUES[key].items()}, "label": issue, "src": f"cfpb_issue_{key}"}
    return qs


def record(row, rights):
    """One candidate record from a source row that carries the derived "text", "key", "bucket" and "text_sha256"."""
    return {"state": row["text"], "questions": questions(row["key"], ISSUE_OF.get((row["key"], row["issue"]))),
            "_meta": {"id": f"cfpb/{row['complaint_id']}", "source": "cfpb", "group_id": f"cfpb/{row['text_sha256'][:20]}", "variant": "clean",
                      "length_bucket": row["bucket"], "chars": len(row["text"]), "company": row["company"], "date_received": row["date_received"],
                      "product_raw": row["product"], "issue_raw": row["issue"], "sub_issue_raw": row["sub_issue"], "text_sha256": row["text_sha256"],
                      "provenance": {"source": "CFPB consumer complaint database", "via": f"hf:{REPO}@{REVISION}", "rights": rights}}}


def rows():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO, revision=REVISION).siblings if s.rfilename.endswith(".parquet"))
    cols = ["complaint_id", "product", "sub_product", "issue", "sub_issue", "complaint_what_happened", "company", "date_received"]
    for f in files:
        t = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION), columns=cols)
        yield from (dict(zip(cols, v)) for v in zip(*(t.column(c).to_pylist() for c in cols)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", default="documents-v1")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    cells, seen, issues, stats = defaultdict(list), set(), defaultdict(Counter), Counter()
    for r in map(prepare, rows()):
        stats["rows"] += 1
        if r is None: continue
        if r["bucket"] is None: stats["length_out_of_range"] += 1; continue
        if r["text_sha256"] in seen: stats["duplicate_text"] += 1; continue
        seen.add(r["text_sha256"]); issues[r["key"]][ISSUE_OF.get((r["key"], r["issue"]))] += 1
        cells[r["key"], r["bucket"]].append(r)
    rng, parts, counts = random.Random(a.seed), defaultdict(list), Counter()
    for (key, bucket), pool in sorted(cells.items()):
        rng.shuffle(pool); start = 0
        for split, n in SPLITS.items():
            for r in pool[start:start + n]:
                parts[split].append(record(r, RIGHTS))
                counts[split, key, bucket] += 1
            start += n
    out.mkdir(parents=True)
    for split, recs in parts.items():
        rng.shuffle(recs); write_jsonl(out / f"{split}.jsonl", recs)
    write_json(out / "build.json", {"repo": REPO, "revision": REVISION, "seed": a.seed, "stats": dict(stats), "issue_coverage": {k: {str(i): n for i, n in c.most_common()} for k, c in issues.items()},
                                    "per_split": {s: len(v) for s, v in parts.items()}, "questions": {s: sum(len(r["questions"]) for r in v) for s, v in parts.items()},
                                    "cells": {f"{s}/{k}/{b}": n for (s, k, b), n in sorted(counts.items())}, "buckets": BUCKETS, "code_sha256": digest(__file__)})
    print({s: len(v) for s, v in parts.items()}, dict(stats))


if __name__ == "__main__":
    main()
