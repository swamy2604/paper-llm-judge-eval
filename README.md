# LLM-as-a-Judge Evaluation on Public RAG Benchmarks

Research paper repo for Narasimha Swamy Thoutam (PhD, Information Systems,
University of the Cumberlands). Public data and open-source models only.
No employer data is used anywhere in this study.

**Working title:** "How Reliable Are LLM-as-a-Judge Evaluations? Comparing
Automated and Human Judgments on Public RAG Benchmarks"

**Target venues:** SERA 2027 / ICIS 2027 (deadline Feb 26, 2027); backup AMCIS 2027 (Mar 1, 2027)

## Repo layout

```
paper/      Paper text, section by section (Markdown drafts; final version in LaTeX)
code/       Experiment code — dataset loading, answer generation, judging, scoring
data/       Dataset download scripts and sampling (raw data is NOT committed; see below)
results/    Experiment outputs (tables, figures, agreement scores)
```

## How to run the pilot

```bash
pip install -r code/requirements.txt
python code/01_download_data.py     # downloads Natural Questions + HotpotQA samples
python code/02_generate_answers.py  # generates answers with open models
python code/03_judge_answers.py     # runs the LLM judges (direct + rubric modes)
python code/04_score_agreement.py   # agreement vs human ratings (Cohen's kappa, Spearman)
```

Human ratings go in `data/human_ratings.csv` (columns: question_id, answer_id,
groundedness, correctness, completeness, rater). Rate 600 answers; a second rater
covers a 100-answer subset for inter-rater agreement.

## Research questions

- RQ1: How closely do LLM judges agree with human judges when grading RAG answers?
- RQ2: Which judge design choices matter most — model size, prompting style, or
  access to source documents?
- RQ3: Where do LLM judges fail systematically (length bias, self-preference,
  missed ungrounded claims)?

## Rules for this repo

- Public data only. Never commit employer data, internal metrics, or credentials.
- Every experiment records its prompts, model versions, and random seeds.
- Raw downloaded datasets are NOT committed (they are large and licensed
  separately); only the sampling script and the sampled IDs are committed.
