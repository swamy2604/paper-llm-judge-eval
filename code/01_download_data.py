"""
01_download_data.py — download public datasets and draw the fixed sample.

Datasets: Natural Questions (open-domain QA) and HotpotQA (multi-hop QA).
Only the SAMPLING is fixed here (seed 42); raw data is downloaded fresh by
whoever runs this script, so the study is reproducible without committing
large files. Sample: 300 questions per dataset (600 total).
"""
import json
import random

SEED = 42
N_PER_DATASET = 300


def sample_ids(all_ids, n=N_PER_DATASET, seed=SEED):
    rng = random.Random(seed)
    return sorted(rng.sample(all_ids, n))


if __name__ == "__main__":
    print("TODO: load Natural Questions + HotpotQA via `datasets` (Hugging Face),")
    print("draw 300 IDs per dataset with seed 42, save to data/sample_ids.json")
    print("and store the question + passages for the sampled IDs in data/sampled.json")
