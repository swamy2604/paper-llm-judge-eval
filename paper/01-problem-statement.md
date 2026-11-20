# Problem Statement (draft — Introduction, opening)

Large language models (LLMs) are now routinely used to evaluate the outputs of other LLMs.
This practice, known as LLM-as-a-judge, has become the default evaluation method for
retrieval-augmented generation (RAG) systems, chatbots, and other deployed language
applications, largely because human evaluation is slow, expensive, and difficult to scale.
When an automated judge declares an answer good, teams ship the model behind it; when it
declares an answer bad, they retrain or rewrite prompts. The judge's verdict therefore
directly shapes which systems reach users.

The problem is that the judge itself is rarely evaluated. An LLM judge can be fluent,
confident, and wrong: it can reward long answers over correct ones, prefer answers written
in its own model's style, and overlook claims that are not supported by the retrieved
documents. If such errors are systematic rather than random, then evaluation scores
systematically mislead the teams that rely on them — good systems are held back, bad
systems are released, and in high-stakes settings such as healthcare, users receive
answers that look authoritative but are not grounded in evidence.

Existing research does not settle this question. Studies of LLM-as-a-judge have mostly
tested judges on short, single-turn tasks such as summarisation or general chat, where
grounding in source documents is not the central challenge. Studies of RAG evaluation, in
turn, have mostly proposed new benchmarks or metrics without directly comparing automated
judges against human judgement on the same answers. As a result, we do not know how
closely LLM judges agree with humans on realistic RAG answers, which judge design choices
matter most, or where judges fail in predictable ways.

This paper addresses that gap. We compare three LLM judges of different sizes against
human ratings on 600 RAG answers drawn from two public question-answering benchmarks
(Natural Questions and HotpotQA), testing two judging modes (single overall score versus
rubric-based scoring of groundedness, correctness, and completeness) and whether the
judge sees the source documents. We report agreement with human judgement, identify
systematic failure patterns, and derive practical guidance for teams that use automated
judges to evaluate RAG systems.

## Contributions (what this paper adds)

1. The first direct comparison of LLM judges against human ratings on realistic RAG
   answers from public benchmarks, with a fixed, reproducible sample.
2. Evidence on which judge design choices matter most: model size, prompting style,
   and access to source documents.
3. A taxonomy of systematic judge failures (length bias, self-preference bias,
   missed ungrounded claims), with measured frequencies.
4. Practical guidance for practitioners who rely on automated evaluation of RAG
   systems, plus a fully reproducible code and data release.
