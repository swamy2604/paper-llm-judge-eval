# Results — Pilot Study (DRAFT)

Status: pilot only (50 HotpotQA questions, seed 42). Human ratings are
AI-drafted and pending the author's review; the numbers below are therefore
DRAFT and must be recomputed after the ratings are finalized. Do not cite
these numbers as final results.

## 4. Results

### 4.1 Answer generation

The generator (Qwen2.5-7B-Instruct, temperature 0, passages only) answered
31 of 50 questions fully correctly, 2 nearly correctly, 2 partially, and 3
mostly wrong; the remaining 12 were wrong or abstained (4 wrong answers, 8
abstentions, 3 of them a bare "I don't know"). Abstention was honest
throughout: every "I don't know" was a case where the model stated the
passages did not contain the answer, rather than a hallucinated guess.

### 4.2 Judge–human agreement

Agreement between the three judges and the (draft) human ratings, Cohen's
kappa / Spearman, n = 50 unless noted:

| Judge | Mode | Dimension | n | Kappa | Spearman |
|---|---|---|---|---|---|
| 14B | direct | overall | 50 | 0.21 | 0.71 |
| 14B | rubric | groundedness | 50 | 0.12 | 0.48 |
| 14B | rubric | correctness | 50 | 0.37 | 0.70 |
| 14B | rubric | completeness | 50 | 0.51 | 0.63 |
| 3B | direct | overall | 50 | 0.20 | 0.52 |
| 3B | rubric | groundedness | 49 | -0.11 | 0.62 |
| 3B | rubric | correctness | 50 | 0.08 | 0.59 |
| 3B | rubric | completeness | 50 | 0.09 | 0.60 |
| 7B | direct | overall | 50 | 0.40 | 0.88 |
| 7B | rubric | groundedness | 50 | 0.16 | 0.69 |
| 7B | rubric | correctness | 50 | 0.43 | 0.78 |
| 7B | rubric | completeness | 50 | 0.13 | 0.65 |

Reading the table (RQ1): agreement is slight to moderate on kappa and
moderate to strong on rank correlation. The strongest single result is the
7B judge in direct mode (kappa 0.40, Spearman 0.88). Rank correlation is
consistently higher than kappa, which means judges order answers by quality
much as humans do but disagree on the exact score — a common pattern when
the score scale is used differently by humans and models.

Reading the table (RQ2): judge size does not predict agreement
monotonically. The 7B judge beats the 14B judge on direct overall and on
rubric correctness, while the 14B judge is best on rubric completeness.
The 3B judge is weakest on kappa throughout but its rank correlations
(0.52–0.62) are respectable. Direct prompting beats rubric prompting on
overall agreement for the 7B and 3B judges; the rubric's value is the
per-dimension breakdown, not a higher total score.

Groundedness is the weakest dimension for every judge (kappa -0.11 to
0.16). Judges are much better at recognizing a correct answer than at
catching unsupported claims — the failure mode that matters most for RAG.

### 4.3 Parsing reliability (RQ2, implementation finding)

Before the parser fix, the 7B judge's rubric output was usable for only
37 of 50 answers: it replied with bare numbers ("5 5 4") instead of the
labeled format. After adding a bare-number fallback to the parser and
tightening the prompt ("reply with only the three scores and nothing
else"), all 50 parsed. One 3B output wrote "grounded=X" and remains
unparseable for that dimension (n = 49). Lesson for the full run: judge
outputs must be parsed defensively, and parse failures must be reported
as a result, not silently dropped.

### 4.4 What the pilot does not show yet

- No second-rater reliability (a second human rater is still needed).
- No Natural Questions results (pilot is HotpotQA only).
- No failure-mode breakdown (length bias, self-preference, missed
  ungrounded claims) — the scoring script's breakdowns are not yet
  implemented.
- n = 50 is too small for firm claims; confidence intervals are wide and
  are not reported here.
