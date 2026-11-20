# Research Questions (draft — for the Introduction and Method)

Research questions are the contract of the paper: everything in the Method,
Results, and Discussion must serve one of them. A good research question is
specific, measurable, and answerable with the data you plan to collect.

## RQ1 — Agreement: How closely do LLM judges agree with human judges when grading RAG answers?

- **What it measures:** For each of the 600 answers, compare the LLM judge's
  scores with the human rater's scores on the same rubric.
- **How it is answered:** Cohen's kappa (agreement beyond chance) and Spearman
  correlation, reported separately for each judge and each judging mode.
- **What "good" looks like:** Kappa above 0.6 is generally called substantial
  agreement; below 0.4 is weak. The paper reports the numbers honestly,
  whatever they are — a low agreement is a finding, not a failure.

## RQ2 — Design: Which judge design choices matter most?

Three choices are tested, one at a time, so each effect can be isolated:

1. **Model size** — does a larger judge model agree with humans more than a
   smaller one?
2. **Prompting style** — does rubric-based scoring (separate scores for
   groundedness, correctness, completeness) agree with humans more than a
   single overall score?
3. **Access to source documents** — does giving the judge the retrieved
   passages improve agreement, or does the judge do just as well from the
   question and answer alone?

- **How it is answered:** Compare agreement scores across the conditions
  (e.g., rubric vs. direct, with vs. without passages) and test whether the
  differences are statistically significant.

## RQ3 — Failure modes: Where do LLM judges fail in predictable ways?

- **Length bias:** Do judges give higher scores to longer answers, even when
  the human rater scored them lower?
- **Self-preference bias:** Does a judge favour answers written by a model
  from its own family?
- **Missed ungrounded claims:** How often does a judge pass an answer that
  the human rater flagged as containing claims not supported by the passages?
- **How it is answered:** For each failure type, count its frequency and give
  concrete examples (anonymised answer excerpts) in the Results section.
  Examples make failure modes believable to reviewers.

## Why three questions and not one

One question would make a thin paper. Three questions that build on each other
— first *whether* judges agree, then *what makes them agree more*, then
*where they fail* — give the paper a story arc: measurement, explanation,
implication. Reviewers expect this shape.
