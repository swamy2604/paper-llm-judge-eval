"""
02_generate_answers.py — generate RAG answers with open models.

Backend: Hugging Face Inference API (serverless — no local GPU needed).
Set HF_TOKEN in the environment (free Hugging Face account, read-only token).
Model default: Qwen/Qwen2.5-7B-Instruct (open model, good instruction
following). Swap MODEL to any other open instruct model.

For each sampled question, the model answers using ONLY the retrieved
passages (the context field). The exact prompt template, model name, and
generation settings are recorded in the output file — reviewers will ask
for these.
"""
import json
import os

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
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN first: export HF_TOKEN=hf_... (free Hugging Face account)")
    client = InferenceClient(token=token)
    sample = json.load(open("data/pilot_sample.json"))
    out = []
    for item in sample:
        prompt = PROMPT.format(
            passages=build_passages(item["context"]),
            question=item["question"],
        )
        resp = client.chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,  # deterministic: same input -> same answer
        )
        out.append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "model_answer": resp.choices[0].message.content.strip(),
            "model": MODEL,
            "prompt_template": PROMPT,
            "temperature": 0.0,
        })
        print(f"done {len(out)}/{len(sample)}: {item['id']}")
    with open("results/generated_answers.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print("saved", len(out), "answers to results/generated_answers.jsonl")


if __name__ == "__main__":
    main()
