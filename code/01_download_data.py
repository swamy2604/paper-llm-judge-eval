"""
01_download_data.py — download public datasets and draw the fixed full sample.

Datasets: Natural Questions (open-domain QA) and HotpotQA (multi-hop QA).
Sample: 300 questions per dataset (600 total), seed 42.

Method notes (record these in the paper):
- HotpotQA: `hotpotqa/hotpot_qa`, "distractor", train split (90,447 items).
  Same source as the pilot, so the pilot is a strict subset of the design.
- Natural Questions: `google-research-datasets/natural_questions`,
  VALIDATION split (7,830 items), not train. Rationale: the train split is
  ~87k items with multi-MB documents each, which is impractical to stream
  for a 300-item sample on free Colab; the validation split is the split the
  NQ paper itself evaluates on, and it is the standard evaluation split in
  the literature. Only items with a short answer (or a yes/no answer) are
  eligible, matching HotpotQA's answerable-only design.
- NQ passages: the long-answer byte span from the document text, plus a
  500-byte window on each side. Document text is reconstructed from the
  token byte offsets, which is exact (no HTML parsing, no detokenizer).
- Unified record saved to data/full_sample.json:
  {id, source ("hotpotqa" | "natural_questions"), question, answer,
   type, level (HotpotQA only), context: {"title": [...], "sentences": [[...]]}}
  The context dict format matches data/pilot_sample.json, so the existing
  generation/judging code works unchanged.
"""
import json
import random
import re

SEED = 42
N_PER_DATASET = 300
NQ_WINDOW = 500  # bytes of context kept around the long answer on each side


def reconstruct_text(tokens):
    """Rebuild document text from the token byte-offset table."""
    token_list = tokens["token"]
    starts = tokens["start_byte"]
    ends = tokens["end_byte"]
    size = max(ends) if ends else 0
    buf = [" "] * size
    for tok, s, e in zip(token_list, starts, ends):
        buf[s:e] = list(tok)
    return "".join(buf)


def nq_passage(ex):
    """Return (answer, passage) for an NQ example, or None if unanswerable."""
    anns = ex["annotations"]
    long_answers = anns.get("long_answer") or []
    short_answers = anns.get("short_answers") or []
    yes_no = anns.get("yes_no_answer") or []
    if not long_answers:
        return None
    la = long_answers[0]
    if la.get("start_byte", -1) < 0:
        return None
    if short_answers and short_answers[0].get("text"):
        answer = short_answers[0]["text"][0]
    elif yes_no and yes_no[0] in (0, 1):
        answer = "yes" if yes_no[0] == 1 else "no"
    else:
        return None
    text = reconstruct_text(ex["document"]["tokens"])
    start = max(0, la["start_byte"] - NQ_WINDOW)
    end = min(len(text), la["end_byte"] + NQ_WINDOW)
    passage = text[start:end]
    # The token stream includes HTML markup; strip tags, collapse
    # whitespace, and trim the window start to the next word boundary
    # so passages read as plain text.
    passage = re.sub(r"<[^>]+>", " ", passage)
    passage = re.sub(r"\s+", " ", passage).strip()
    return answer, passage


def sample_hotpotqa(n=N_PER_DATASET, seed=SEED):
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ds)), n))
    out = []
    for i in idx:
        row = ds[i]
        out.append({
            "id": f"hotpotqa-{row['id']}",
            "source": "hotpotqa",
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "supporting_facts": row["supporting_facts"],
            "context": row["context"],
        })
    return out


def sample_nq(n=N_PER_DATASET, seed=SEED):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/natural_questions",
                      split="validation", streaming=True)
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    for ex in ds:
        got = nq_passage(ex)
        if got is None:
            continue
        seen += 1
        answer, passage = got
        item = {
            "id": f"nq-{ex['id']}",
            "source": "natural_questions",
            "question": ex["question"]["text"],
            "answer": answer,
            "type": None,
            "level": None,
            "supporting_facts": None,
            "context": {"title": [ex["document"]["title"]],
                        "sentences": [[passage]]},
        }
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            j = rng.randrange(seen)
            if j < n:
                reservoir[j] = item
    print(f"NQ validation: {seen} answerable items seen, sampled {len(reservoir)}")
    return reservoir


def main():
    import os

    os.makedirs("data", exist_ok=True)
    print("sampling HotpotQA...")
    hotpot = sample_hotpotqa()
    print("sampling Natural Questions (validation split, streaming)...")
    nq = sample_nq()
    full = hotpot + nq
    with open("data/full_sample.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"saved {len(full)} questions ({len(hotpot)} HotpotQA + {len(nq)} NQ)"
          " to data/full_sample.json")


if __name__ == "__main__":
    main()
