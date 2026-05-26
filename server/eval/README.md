# RAG Evaluation Harness

This folder contains the offline benchmark harness for the resume assistant's RAG pipeline.

The goal is to make retrieval quality measurable instead of relying on manual chat checks. By default, the eval runner measures retrieval quality without generating final LLM answers. When needed, it can also run live answer generation and score answer relevance and faithfulness.

## What This Measures

The eval harness answers four practical questions:

- Did retrieval find context related to the recruiter question?
- Did retrieved context include the expected facts or keywords?
- Did retrieval hit the expected resume sections?
- How long did routing and retrieval take?

When `--generate-answers` is enabled, it also checks:

- Whether the answer is relevant to the user query.
- Whether the answer is grounded in retrieved context.
- Whether hallucination risk is low, medium, or high.

## Folder Structure

```text
server/eval/
├── dataset.json
├── run_eval.py
├── results/
│   └── *.json / *.md
└── README.md
```

`dataset.json` contains benchmark questions and expected retrieval signals.

`run_eval.py` is the CLI runner.

`results/` stores timestamped eval reports for before/after comparisons.

## Quick Start

Run from the backend repo root:

```bash
python -m server.eval.run_eval
```

This runs retrieval-only evaluation using:

```text
dataset: server/eval/dataset.json
label: baseline
output dir: server/eval/results
```

It writes two files:

```text
server/eval/results/{timestamp}_baseline.json
server/eval/results/{timestamp}_baseline.md
```

## CLI Options

```bash
python -m server.eval.run_eval --help
```

Available flags:

| Flag | Purpose | Default |
| --- | --- | --- |
| `--dataset` | Path to the benchmark dataset | `server/eval/dataset.json` |
| `--label` | Name for this eval run | `baseline` |
| `--output-dir` | Directory for JSON and Markdown reports | `server/eval/results` |
| `--generate-answers` | Also generate LLM answers and score answer quality | disabled |

## Common Commands

Run the default retrieval-only eval:

```bash
python -m server.eval.run_eval
```

Run a labeled baseline:

```bash
python -m server.eval.run_eval --label minilm-baseline
```

Run against a custom dataset:

```bash
python -m server.eval.run_eval --dataset server/eval/dataset.json --label custom-dataset
```

Generate live answers and answer-quality metrics:

```bash
python -m server.eval.run_eval --label minilm-live --generate-answers
```

Write reports to a custom output directory:

```bash
python -m server.eval.run_eval --label experiment-1 --output-dir /tmp/rag-eval-results
```

## Dataset Format

Each item in `dataset.json` has this shape:

```json
{
  "id": "skills_backend",
  "question": "What backend skills does this candidate have?",
  "expected_keywords": ["python", "fastapi", "api", "backend"],
  "expected_sections": ["backend"]
}
```

Fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier for the benchmark question |
| `question` | Recruiter-style question sent through the RAG pipeline |
| `expected_keywords` | Facts or terms expected somewhere in retrieved context |
| `expected_sections` | Resume sections expected in retrieved context |

The current dataset is intentionally generic. It is a v1 benchmark that can be refined with more resume-specific facts over time.

## Report Outputs

Each run writes:

- A JSON report for machine-readable comparisons.
- A Markdown report for quick review in GitHub.

The JSON report includes:

- config snapshot
- aggregate metrics
- per-question retrieval results
- optional answer-quality metrics

The Markdown report includes:

- model and retrieval config
- aggregate metrics table
- weakest questions table
- whether live answer generation was enabled

## Metrics

| Metric | Meaning |
| --- | --- |
| `retrieval_precision` | Fraction of retrieved chunks that contain at least one expected keyword or section marker |
| `context_recall` | Fraction of expected keywords found across all retrieved context |
| `section_hit_rate` | Fraction of expected sections found in retrieved context |
| `avg_latency_ms` | Average per-question eval latency |
| `p95_latency_ms` | 95th percentile latency |
| `cache_hit_rate` | Fraction of eval questions served through semantic cache routing |
| `answer_relevance` | Similarity between user query and generated answer, only with `--generate-answers` |
| `faithfulness` | Similarity between generated answer and retrieved context, only with `--generate-answers` |
| `context_score` | Retrieval-context relevance score from the existing `RAGEvaluator` |

## Comparing Experiments

Use labels to make before/after runs easy to compare.

Example workflow:

```bash
python -m server.eval.run_eval --label minilm-baseline
```

Change one retrieval variable, such as embedding model, `RETRIEVAL_TOP_K`, or `RAG_SCORE_THRESHOLD`.

```bash
python -m server.eval.run_eval --label bge-top5
```

Then compare reports under:

```text
server/eval/results/
```

This gives a concrete story:

```text
Baseline retrieval precision: 0.42
After embedding/top-k change: 0.67
```

The exact numbers depend on your resume data and configuration.

## Recruiter and Interview Framing

This harness demonstrates production AI engineering discipline:

- The system has repeatable evals, not just manual demos.
- Retrieval quality is measured separately from answer quality.
- Experiments can be labeled and compared over time.
- Reports are saved as artifacts that can be reviewed in GitHub.

A clear way to explain it:

```text
I built an eval-driven RAG system. The eval harness measures retrieval precision,
context recall, section hit rate, latency, cache behavior, and optional answer
faithfulness. This lets me compare embedding models, chunking, top-k, and
thresholds using evidence instead of guesses.
```

## Limitations

- The default dataset is a v1 generic recruiter benchmark.
- Keyword and section matching are deterministic checks, not a full LLM judge.
- Live answer evaluation can call the configured LLM and may consume API tokens.
- Real `.env` files and resume PDFs are intentionally not committed.
- The benchmark should be refined as the resume content and product goals evolve.

