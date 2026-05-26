import argparse
import asyncio
import json
import math
import re
from datetime import datetime
from pathlib import Path

from server.config import settings
from server.core.assistant import FastAPIPersonalAssistant


DEFAULT_DATASET = "server/eval/dataset.json"
DEFAULT_OUTPUT_DIR = "server/eval/results"


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"Dataset file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset is not valid JSON: {path}") from exc

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")

    required_fields = {"id", "question", "expected_keywords", "expected_sections"}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object")

        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Dataset item {item.get('id', index)!r} is missing: {sorted(missing)}"
            )

        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Dataset item {item['id']!r} has an invalid question")

        for field in ("expected_keywords", "expected_sections"):
            if not isinstance(item[field], list) or not all(
                isinstance(value, str) and value.strip() for value in item[field]
            ):
                raise ValueError(
                    f"Dataset item {item['id']!r} field {field} must be a list of strings"
                )

    return data


def normalize_text(value: str) -> str:
    return value.lower()


def contains_any(text: str, values: list[str]) -> bool:
    haystack = normalize_text(text)
    return any(normalize_text(value) in haystack for value in values)


def retrieval_metrics(context_chunks: list[str], expected_keywords: list[str], expected_sections: list[str]) -> dict:
    context_text = "\n\n".join(context_chunks)
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in normalize_text(context_text)
    ]
    matched_sections = [
        section
        for section in expected_sections
        if normalize_text(section) in normalize_text(context_text)
    ]

    relevant_chunks = [
        chunk
        for chunk in context_chunks
        if contains_any(chunk, expected_keywords) or contains_any(chunk, expected_sections)
    ]

    chunk_count = len(context_chunks)
    return {
        "retrieved_chunks": chunk_count,
        "relevant_chunks": len(relevant_chunks),
        "retrieval_precision": round(len(relevant_chunks) / chunk_count, 3) if chunk_count else 0,
        "context_recall": round(len(matched_keywords) / len(expected_keywords), 3)
        if expected_keywords
        else 0,
        "section_hit_rate": round(len(matched_sections) / len(expected_sections), 3)
        if expected_sections
        else 0,
        "matched_keywords": matched_keywords,
        "matched_sections": matched_sections,
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = math.ceil((percentile_value / 100) * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return round(ordered[index], 2)


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def aggregate_results(results: list[dict]) -> dict:
    precisions = [item["metrics"]["retrieval_precision"] for item in results]
    recalls = [item["metrics"]["context_recall"] for item in results]
    section_hits = [item["metrics"]["section_hit_rate"] for item in results]
    latencies = [item["latency_ms"] for item in results]
    cache_hits = [item["cache_hit"] for item in results]
    answer_metrics = [
        item["rag_eval"]
        for item in results
        if item.get("answer") and item.get("rag_eval") and "error" not in item["rag_eval"]
    ]

    aggregate = {
        "questions": len(results),
        "retrieval_precision": average(precisions),
        "context_recall": average(recalls),
        "section_hit_rate": average(section_hits),
        "avg_latency_ms": average(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "cache_hit_rate": round(sum(1 for hit in cache_hits if hit) / len(cache_hits), 3)
        if cache_hits
        else 0,
    }

    if answer_metrics:
        aggregate["answer_relevance"] = average(
            [metric.get("answer_relevance", 0) for metric in answer_metrics]
        )
        aggregate["faithfulness"] = average(
            [metric.get("faithfulness", 0) for metric in answer_metrics]
        )
        aggregate["context_score"] = average(
            [metric.get("context_score", 0) for metric in answer_metrics]
        )

    return aggregate


def config_snapshot() -> dict:
    return {
        "model_name": settings.MODEL_NAME,
        "embedding_model": settings.CLOUDFLARE_EMBEDDING_MODEL,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "rag_score_threshold": settings.RAG_SCORE_THRESHOLD,
        "redis_enabled": settings.REDIS_ENABLED,
    }


def safe_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label.strip())
    return cleaned.strip("-") or "eval"


def write_json_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def score_for_sorting(result: dict) -> float:
    metrics = result["metrics"]
    return average([
        metrics["retrieval_precision"],
        metrics["context_recall"],
        metrics["section_hit_rate"],
    ])


def markdown_table(rows: list[list[object]]) -> str:
    if not rows:
        return ""

    header = rows[0]
    separator = ["---"] * len(header)
    table_rows = [header, separator, *rows[1:]]
    return "\n".join(
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in table_rows
    )


def write_markdown_report(path: Path, report: dict) -> None:
    aggregate = report["aggregate"]
    config = report["config"]
    weakest = sorted(report["results"], key=score_for_sorting)[:5]

    aggregate_rows = [
        ["Metric", "Value"],
        ["Questions", aggregate["questions"]],
        ["Retrieval precision", aggregate["retrieval_precision"]],
        ["Context recall", aggregate["context_recall"]],
        ["Section hit rate", aggregate["section_hit_rate"]],
        ["Avg latency ms", aggregate["avg_latency_ms"]],
        ["P95 latency ms", aggregate["p95_latency_ms"]],
        ["Cache hit rate", aggregate["cache_hit_rate"]],
    ]

    if "answer_relevance" in aggregate:
        aggregate_rows.extend([
            ["Answer relevance", aggregate["answer_relevance"]],
            ["Faithfulness", aggregate["faithfulness"]],
            ["Context score", aggregate["context_score"]],
        ])

    weakest_rows = [["ID", "Precision", "Recall", "Section Hit", "Latency ms"]]
    for item in weakest:
        metrics = item["metrics"]
        weakest_rows.append([
            item["id"],
            metrics["retrieval_precision"],
            metrics["context_recall"],
            metrics["section_hit_rate"],
            item["latency_ms"],
        ])

    content = "\n\n".join([
        f"# RAG Eval Report: {report['label']}",
        f"Generated: {report['generated_at']}",
        f"Live answer generation: {'enabled' if report['generate_answers'] else 'disabled'}",
        "## Config",
        markdown_table([
            ["Setting", "Value"],
            ["Model", config["model_name"]],
            ["Embedding", config["embedding_model"]],
            ["Top K", config["retrieval_top_k"]],
            ["Threshold", config["rag_score_threshold"]],
            ["Redis enabled", config["redis_enabled"]],
        ]),
        "## Aggregate Metrics",
        markdown_table(aggregate_rows),
        "## Weakest Questions",
        markdown_table(weakest_rows),
        "",
    ])

    path.write_text(content, encoding="utf-8")


async def run_eval(args: argparse.Namespace) -> dict:
    dataset = load_dataset(Path(args.dataset))
    assistant = FastAPIPersonalAssistant(
        pdf_path=settings.PDF_PATH,
        assistant_name=settings.ASSISTANT_NAME,
        model_name=settings.MODEL_NAME,
    )

    results = []
    for item in dataset:
        eval_result = await assistant.run_eval_query(
            item["question"],
            generate_answer=args.generate_answers,
        )
        metrics = retrieval_metrics(
            eval_result["context_chunks"],
            item["expected_keywords"],
            item["expected_sections"],
        )

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_keywords": item["expected_keywords"],
            "expected_sections": item["expected_sections"],
            "route_type": eval_result["route_type"],
            "intent": eval_result["intent"],
            "retrieval_queries": eval_result["retrieval_queries"],
            "context_chunks": eval_result["context_chunks"],
            "answer": eval_result["answer"],
            "cache_hit": eval_result["cache_hit"],
            "latency_ms": eval_result["latency_ms"],
            "rag_eval": eval_result["rag_eval"],
            "metrics": metrics,
            "error": eval_result.get("error"),
        })

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    label = safe_label(args.label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "label": label,
        "generated_at": timestamp,
        "dataset": args.dataset,
        "generate_answers": args.generate_answers,
        "config": config_snapshot(),
        "aggregate": aggregate_results(results),
        "results": results,
    }

    json_path = output_dir / f"{timestamp}_{label}.json"
    markdown_path = output_dir / f"{timestamp}_{label}.md"
    write_json_report(json_path, report)
    write_markdown_report(markdown_path, report)

    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {markdown_path}")
    print(json.dumps(report["aggregate"], indent=2))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generate-answers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_eval(args))
    except ValueError as exc:
        raise SystemExit(f"Eval configuration error: {exc}") from exc


if __name__ == "__main__":
    main()
