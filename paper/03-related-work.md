# Related Work (DRAFT — verify every claim against the cited papers before submission)

## 2. Related Work

### 2.1 LLM-as-a-Judge

The practice of using a strong language model to grade the outputs of other
models was systematized by Zheng et al. (2023), who introduced MT-Bench (80
challenging multi-turn questions) and Chatbot Arena (crowdsourced pairwise
human preferences) and measured how well LLM judges agree with human
preferences. Their central finding was that a strong judge (GPT-4) agrees
with human preferences at roughly the same rate that humans agree with each
other — which made automated judging a credible, scalable substitute for
human evaluation of open-ended answers. The same study named the three
biases that now frame every serious discussion of automated judging:
**position bias** (favoring whichever answer is shown first),
**verbosity bias** (favoring longer answers), and **self-enhancement bias**
(favoring the judge's own outputs), plus weaker grading of math and
reasoning questions. They also proposed cheap mitigations: swapping answer
positions, giving the judge a reference answer, and asking for step-by-step
reasoning before the verdict.

This paper builds directly on that foundation but differs in three ways.
First, Zheng et al. studied **chat** evaluation (pairwise preference and
single-answer grading of open-ended responses); we study **RAG** evaluation,
where every answer is judged against retrieved source passages, so
groundedness — whether each claim is supported by the sources — is a
first-class dimension rather than a side concern. Second, they used a
frontier proprietary judge (GPT-4); we use **open models of three sizes**
(3B, 7B, 14B), so the question of whether judge size matters can be answered
without API access or cost. Third, they measured agreement on preference
votes; we measure agreement on **absolute 1–5 scores** in direct and
rubric modes, the format most RAG pipelines actually deploy.

### 2.2 RAG Evaluation Frameworks

RAGAS (Es et al., 2023) is the most widely used reference-free RAG
evaluation framework. It scores generated answers on faithfulness (are the
claims supported by the retrieved context?), answer relevance (does the
answer address the question?), and context relevance (is the retrieved
context itself relevant?). Our rubric dimensions map closely onto this
scheme — groundedness corresponds to faithfulness, completeness to a
combination of answer relevance and coverage — but RAGAS computes its
scores with LLM calls whose prompts and parsing are fixed by the library,
while our study varies the judge and the prompting mode deliberately and
measures the effect of each choice against human ratings.

ARES (Saad-Falcon et al., 2024) takes a different route: it fine-tunes
small, specialized judge models on synthetic data and validates them with
prediction-powered inference, which gives confidence intervals on the
evaluation itself. ARES shows that a fine-tuned judge can beat a prompted
general-purpose judge on RAG evaluation. Our study is complementary: we
measure how far *prompted* open models get without any fine-tuning, which
is the realistic starting point for a team that cannot afford to train a
judge, and our results set the baseline that a fine-tuned judge would need
to beat.

### 2.3 Benchmarks: HotpotQA and Natural Questions

HotpotQA (Yang et al., 2018) is a multi-hop question-answering dataset built
from Wikipedia, where each question requires reasoning over two or more
supporting passages. Its distractor setting — each question ships with
relevant passages mixed with distractors — mirrors a real RAG retriever,
which returns useful and irrelevant passages together. Natural Questions
(Kwiatkowski et al., 2019) is a large-scale dataset of real user queries
against Wikipedia, with long answers (passages) and short answers (spans).
It is single-hop and closer to the everyday questions a production RAG
system faces. Using both datasets lets us test whether judge reliability
depends on question type: multi-hop reasoning (HotpotQA) versus direct
fact lookup (Natural Questions). To our knowledge, no prior study compares
LLM-judge agreement with humans across both datasets in the same design
with the same judges and rubric.

### 2.4 What This Study Adds

Three gaps motivate this work. (1) Judge reliability is usually reported
on chat benchmarks, not on grounded RAG answers where unsupported claims
are the main failure mode. (2) Most studies use one judge; the effect of
judge size and of direct versus rubric prompting is rarely measured in the
same experiment. (3) Failure modes are usually reported as aggregate
scores, not as a breakdown of *how* judges fail — which is what a team
needs in order to fix its evaluation pipeline. Our research questions
address each gap in turn (Section 1.3).

## References (to be completed with full citations)

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
- Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. NAACL 2024. arXiv:2311.09476.
- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018. arXiv:1809.09600.
- Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., & Petrov, S. (2019). Natural Questions: A Benchmark for Question Answering Research. TACL. (Full citation to be verified against the published version.)
