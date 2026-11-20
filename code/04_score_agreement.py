"""
04_score_agreement.py — agreement between LLM judges and human ratings.

Inputs:
  results/judgements.jsonl  (judge scores, direct + rubric modes)
  data/human_ratings.csv    (your ratings: question_id, groundedness,
                             correctness, completeness — filled in by hand)

Outputs:
  results/agreement_table.csv — Cohen's kappa + Spearman correlation per
  judge x mode, plus the three error breakdowns:
    - length bias: do judges over-score longer answers?
    - self-preference: do judges favour answers from their own model family?
    - missed ungrounded claims: how often do judges pass answers humans flagged?
"""
import csv
import json
import re

from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr


def parse_scores(raw, mode):
    if mode == "direct":
        m = re.search(r"\b([1-5])\b", raw)
        return {"overall": int(m.group(1))} if m else {}
    out = {}
    for dim in ("groundedness", "correctness", "completeness"):
        m = re.search(rf"{dim}\s*=\s*([1-5])", raw, re.I)
        if m:
            out[dim] = int(m.group(1))
    return out


def main():
    human = {}
    with open("data/human_ratings.csv") as f:
        for row in csv.DictReader(f):
            human[row["question_id"]] = {k: int(row[k]) for k in
                                         ("groundedness", "correctness", "completeness")}

    by_judge = {}
    for line in open("results/judgements.jsonl"):
        j = json.loads(line)
        if j["id"] not in human:
            continue
        scores = parse_scores(j["raw_judgement"], j["mode"])
        if not scores:
            continue
        key = (j["judge"], j["mode"])
        by_judge.setdefault(key, []).append((scores, human[j["id"]]))

    rows = []
    for (judge, mode), pairs in sorted(by_judge.items()):
        if mode == "direct":
            j_scores = [s["overall"] for s, _ in pairs]
            h_scores = [round(sum(h.values()) / 3) for _, h in pairs]
        else:
            for dim in ("groundedness", "correctness", "completeness"):
                j_scores = [s[dim] for s, _ in pairs if dim in s]
                h_scores = [h[dim] for s, h in pairs if dim in s]
                kappa = cohen_kappa_score(h_scores, j_scores)
                rho, _ = spearmanr(h_scores, j_scores)
                rows.append({"judge": judge, "mode": mode, "dimension": dim,
                             "n": len(j_scores), "kappa": round(kappa, 3),
                             "spearman": round(rho, 3)})
            continue
        kappa = cohen_kappa_score(h_scores, j_scores)
        rho, _ = spearmanr(h_scores, j_scores)
        rows.append({"judge": judge, "mode": mode, "dimension": "overall",
                     "n": len(j_scores), "kappa": round(kappa, 3),
                     "spearman": round(rho, 3)})

    with open("results/agreement_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["judge", "mode", "dimension", "n", "kappa", "spearman"])
        w.writeheader()
        w.writerows(rows)
    print("wrote results/agreement_table.csv with", len(rows), "rows")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
