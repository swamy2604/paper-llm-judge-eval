"""
02_generate_answers_local.py — generate RAG answers with a local open model.

Free-compute backend: runs the model on a local GPU (Google Colab free T4).
Loads the model in 4-bit (bitsandbytes) so a 7B model fits in ~6 GB.
No Hugging Face token or paid credits needed — the model weights are public.

Usage:
  python code/02_generate_answers_local.py [--model Qwen/Qwen2.5-7B-Instruct]
      [--sample data/pilot_sample.json] [--out results/generated_answers.jsonl]

Same prompt, settings, and output format as 02_generate_answers.py, so the
results are directly comparable with the API version.

Resumable: if --out already exists, questions already answered in it are
skipped and new answers are appended, so an interrupted run can be resumed.
Each answer is written to --out immediately (flushed per row), so a crash
loses at most the one answer being generated.

Memory: a free T4 has ~15 GB. A very long context can exhaust it during
attention (the full run hit this at question 297/600). Passages are capped
at MAX_PASSAGE_CHARS (truncation is counted and reported), and an
out-of-memory error retries once with a shorter context before the item is
recorded as skipped (generation_error="oom") instead of crashing the run.
Skipped items are excluded from judging and reported, not silently filled.
"""
import argparse
import json

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_PASSAGE_CHARS = 8000   # ~2,000 tokens; longer contexts are truncated
RETRY_PASSAGE_CHARS = 3000
PROMPT = (
    "Answer the question using ONLY the information in the passages below. "
    "If the passages do not contain the answer, say \"I don't know\".\n\n"
    "Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"
)


def build_passages(context, max_chars=MAX_PASSAGE_CHARS):
    # pilot_sample.json stores context as a dict {"title": [...], "sentences": [[...]]};
    # accept that, and also a list of [title, [sentences]] pairs.
    if isinstance(context, dict):
        context = zip(context["title"], context["sentences"])
    parts = []
    for title, sentences in context:
        parts.append(f"[{title}] " + " ".join(sentences))
    text = chr(10).join(parts)
    truncated = len(text) > max_chars
    return text[:max_chars], truncated


def main():
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sample", default="data/pilot_sample.json")
    ap.add_argument("--out", default="results/generated_answers.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    sample = json.load(open(args.sample))
    total = len(sample)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
        print(f"resuming: {len(done)} answers already in {args.out}")
    sample = [x for x in sample if x["id"] not in done]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_truncated = 0
    n_skipped = 0
    n_new = 0
    with open(args.out, "a") as f:
        for item in sample:
            passages, truncated = build_passages(item["context"])
            n_truncated += int(truncated)
            answer, error = generate_one(model, tok, passages, item["question"])
            if error == "oom":
                passages, _ = build_passages(item["context"], RETRY_PASSAGE_CHARS)
                answer, error = generate_one(model, tok, passages, item["question"])
            if error:
                n_skipped += 1
                print(f"SKIPPED {item['id']}: {error}", flush=True)
            row = {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["answer"],
                "model_answer": answer,
                "model": args.model,
                "prompt_template": PROMPT,
                "temperature": 0.0,
                "passages_truncated": truncated,
            }
            if error:
                row["generation_error"] = error
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_new += 1
            print(f"done {len(done) + n_new}/{total}: {item['id']}", flush=True)
            torch.cuda.empty_cache()
    print(f"saved {n_new} new answers to {args.out} "
          f"({n_truncated} passages truncated, {n_skipped} skipped)")


def generate_one(model, tok, passages, question):
    import torch
    messages = [{
        "role": "user",
        "content": PROMPT.format(passages=passages, question=question),
    }]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "", "oom"
    answer = tok.decode(
        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return answer, None


if __name__ == "__main__":
    main()
