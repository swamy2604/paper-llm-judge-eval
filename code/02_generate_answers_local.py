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
"""
import argparse
import json

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = (
    "Answer the question using ONLY the information in the passages below. "
    "If the passages do not contain the answer, say \"I don't know\".\n\n"
    "Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"
)


def build_passages(context):
    # pilot_sample.json stores context as a dict {"title": [...], "sentences": [[...]]};
    # accept that, and also a list of [title, [sentences]] pairs.
    if isinstance(context, dict):
        context = zip(context["title"], context["sentences"])
    parts = []
    for title, sentences in context:
        parts.append(f"[{title}] " + " ".join(sentences))
    return chr(10).join(parts)


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
    out = []
    for item in sample:
        messages = [{
            "role": "user",
            "content": PROMPT.format(
                passages=build_passages(item["context"]),
                question=item["question"],
            ),
        }]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        answer = tok.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        out.append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "model_answer": answer,
            "model": args.model,
            "prompt_template": PROMPT,
            "temperature": 0.0,
        })
        print(f"done {len(done) + len(out)}/{total}: {item['id']}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print("saved", len(out), "new answers to", args.out)


if __name__ == "__main__":
    main()
