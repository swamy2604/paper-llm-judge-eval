"""
03_judge_answers.py — run the LLM judges.

Each judge grades every generated answer in TWO modes:
  (a) direct: one overall score (1-5)
  (b) rubric: separate scores for groundedness, correctness, completeness (1-5 each)
Each judge sees the question, the source passages, and the answer — the same
information the human rater sees.

Judges: 3 open models of different sizes, via the HF Inference API.
Swap JUDGES to change the judge set; the paper's design compares sizes.
"""
import json
import os

JUDGES = [
    "Qwen/Qwen2.5-7B-Instruct",    # medium judge
    "Qwen/Qwen2.5-14B-Instruct",   # large judge
    "Qwen/Qwen2.5-3B-Instruct",    # small judge
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
    sample = {x["id"]: x for x in json.load(open("data/pilot_sample.json"))}
    answers = [json.loads(l) for l in open("results/generated_answers.jsonl")]

    out = []
    for ans in answers:
        item = sample[ans["id"]]
        passages = build_passages(item["context"])
        for judge in JUDGES:
            for mode, template in (("direct", DIRECT_PROMPT), ("rubric", RUBRIC_PROMPT)):
                prompt = template.format(
                    passages=passages, question=ans["question"], answer=ans["model_answer"]
                )
                resp = client.chat_completion(
                    model=judge,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                    temperature=0.0,
                )
                out.append({
                    "id": ans["id"],
                    "judge": judge,
                    "mode": mode,
                    "raw_judgement": resp.choices[0].message.content.strip(),
                })
        print(f"judged {len(out) // 6}/{len(answers)}")
    with open("results/judgements.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print("saved", len(out), "judgements to results/judgements.jsonl")


if __name__ == "__main__":
    main()
