"""Pilot: load HotpotQA (distractor, train split) and draw a fixed
50-question sample with seed 42. Saves data/pilot_sample.json with the
question, answer, supporting facts, and context passages for each item.
Note: the HF Hub repo moved to the `hotpotqa` namespace."""
import json, random
from datasets import load_dataset

SEED = 42
N = 50

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
print("HotpotQA train size:", len(ds))
rng = random.Random(SEED)
idx = sorted(rng.sample(range(len(ds)), N))
sample = []
for i in idx:
    row = ds[i]
    sample.append({
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": row["supporting_facts"],
        "context": row["context"],
    })
with open("data/pilot_sample.json", "w") as f:
    json.dump(sample, f, indent=2)
print("saved", len(sample), "questions to data/pilot_sample.json")
print("first question:", sample[0]["question"])
