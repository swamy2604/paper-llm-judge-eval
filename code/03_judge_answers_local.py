"""
03_judge_answers_local.py — run the LLM judges on a local GPU (free Colab T4).

Loads each judge model one at a time in 4-bit, grades all answers in direct
and rubric modes, then unloads it before the next judge, so a 14B judge still
fits in Colab's free ~15 GB GPU.

Usage:
  python code/03_judge_answers_local.py [--judge Qwen/Qwen2.5-7B-Instruct]
      [--sample data/pilot_sample.json]
      [--answers results/generated_answers.jsonl]
      [--out results/judgements.jsonl]

Same prompts and output format as 03_judge_answers.py, so results are
directly comparable with the API version.

Resumable: judgements already in --out (same id + judge + mode) are skipped
and new ones are appended. Each judgement is written to --out immediately
(flushed per row), so a crash loses at most the one judgement being made.
Run one judge at a time with --judge so a long full-scale run can be split
across Colab sessions.

Memory: same T4 limit as generation — passages are capped at
MAX_PASSAGE_CHARS, and an out-of-memory error retries once with a shorter
context before that judgement is recorded as skipped (raw_judgement="",
judgement_error="oom"). Answers that failed generation (generation_error
set) are skipped entirely.
"""
import json
import os

MAX_PASSAGE_CHARS = 8000
RETRY_PASSAGE_CHARS = 3000

JUDGES = [
    "Qwen/Qwen2.5-3B-Instruct",    # small judge
    "Qwen/Qwen2.5-7B-Instruct",    # medium judge
    "Qwen/Qwen2.5-14B-Instruct",   # large judge
]

DIRECT_PROMPT = (
    "Grade the answer to the question on a scale of 1 to 5, where 1 is completely "
    "wrong and 5 is fully correct and well supported. Reply with only the number.\n\n"
    "Passages:\n{passages}\n\nQuestion: {question}\nAnswer: {answer}\nScore:"
)

RUBRIC_PROMPT = (
    "Grade the answer on three dimensions, each 1 to 5. Reply with only three "
    "numbers in this exact format and nothing else — no explanations, no extra words: "
    "groundedness=X, correctness=Y, completeness=Z\n"
    "Always write a number 1-5 for each dimension; never write X or leave one blank.\n"
    "- groundedness: every claim in the answer is supported by the passages\n"
    "- correctness: the answer matches the correct answer\n"
    "- completeness: the answer covers everything the question asks\n\n"
    "Passages:\n{passages}\n\nQuestion: {question}\nAnswer: {answer}\nScores:"
)


def build_passages(context, max_chars=MAX_PASSAGE_CHARS):
    # pilot_sample.json stores context as a dict {"title": [...], "sentences": [[...]]};
    # accept that, and also a list of [title, [sentences]] pairs.
    if isinstance(context, dict):
        context = zip(context["title"], context["sentences"])
    parts = []
    for title, sentences in context:
        parts.append(f"[{title}] " + " ".join(sentences))
    return chr(10).join(parts)[:max_chars]


def grade(model, tok, prompt):
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "", "oom"
    return tok.decode(
        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip(), None


def main():
    import argparse
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None,
                    help="run a single judge (default: all three)")
    ap.add_argument("--sample", default="data/pilot_sample.json")
    ap.add_argument("--answers", default="results/generated_answers.jsonl")
    ap.add_argument("--out", default="results/judgements.jsonl")
    args = ap.parse_args()
    judges = [args.judge] if args.judge else JUDGES

    sample = {x["id"]: x for x in json.load(open(args.sample))}
    answers = [json.loads(l) for l in open(args.answers)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(json.loads(l)["id"], json.loads(l)["judge"], json.loads(l)["mode"])
                    for l in f if l.strip()}
        print(f"resuming: {len(done)} judgements already in {args.out}")
    out = open(args.out, "a")
    n_new = 0
    n_skipped = 0
    for judge in judges:
        tok = AutoTokenizer.from_pretrained(judge)
        model = AutoModelForCausalLM.from_pretrained(
            judge,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        for ans in answers:
            if ans.get("generation_error"):
                continue
            item = sample[ans["id"]]
            passages = build_passages(item["context"])
            for mode, template in (("direct", DIRECT_PROMPT), ("rubric", RUBRIC_PROMPT)):
                if (ans["id"], judge, mode) in done:
                    continue
                prompt = template.format(
                    passages=passages, question=ans["question"], answer=ans["model_answer"]
                )
                raw, error = grade(model, tok, prompt)
                if error == "oom":
                    prompt = template.format(
                        passages=build_passages(item["context"], RETRY_PASSAGE_CHARS),
                        question=ans["question"], answer=ans["model_answer"])
                    raw, error = grade(model, tok, prompt)
                row = {
                    "id": ans["id"],
                    "judge": judge,
                    "mode": mode,
                    "raw_judgement": raw,
                }
                if error:
                    row["judgement_error"] = error
                    n_skipped += 1
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_new += 1
                torch.cuda.empty_cache()
        print(f"judge done: {judge} ({n_new} new judgements this run, {n_skipped} skipped)", flush=True)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
    out.close()
    print("saved", n_new, "new judgements to", args.out)


if __name__ == "__main__":
    main()
