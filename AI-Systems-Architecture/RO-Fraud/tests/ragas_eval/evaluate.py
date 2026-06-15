"""
RAGAS evaluation runner for the RO-Fraud RAG pipeline.

Evaluates three layers:
  1. Retrieval quality  — context_precision, context_recall
  2. Generation quality — faithfulness, answer_relevancy
  3. End-to-end         — answer_correctness (requires ground_truth)

Usage (from the RO-Fraud root with .venv active):
    # Offline — score the golden dataset against stored answers (no live GCP calls):
    python -m tests.ragas_eval.evaluate --mode offline

    # Online — run live retrieval + generation then score (requires GCP credentials):
    python -m tests.ragas_eval.evaluate --mode online

Environment variables required for online mode (same as the main app):
    GCP_PROJECT_ID, GCP_REGION, GCS_BUCKET_NAME,
    VERTEX_INDEX_ID, VERTEX_ENDPOINT_ID, GEMINI_MODEL
"""

import argparse
import json
import logging
import os
import sys

# Allow running as `python -m tests.ragas_eval.evaluate` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ragas_dataset(records: list) -> "Dataset":
    """Convert our golden-dataset dicts into a HuggingFace Dataset for RAGAS."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "question":     [r["question"]     for r in records],
            "contexts":     [r["contexts"]     for r in records],
            "answer":       [r["answer"]       for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        }
    )


def _run_offline_eval(records: list) -> dict:
    """
    Score the golden dataset using pre-written answers.
    No live GCP calls — safe to run in CI without credentials.
    RAGAS uses an LLM-as-judge internally; configure it to use VertexAI
    so the judge itself is also on GCP (or swap for any LangChain LLM).
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )
    from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Use the same Gemini model as the main app for the RAGAS judge
    from src.config import settings

    judge_llm = LangchainLLMWrapper(
        VertexAI(
            model_name=settings.GEMINI_MODEL,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_REGION,
            temperature=0.0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_REGION,
        )
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    # Inject the judge into every metric
    for m in metrics:
        m.llm = judge_llm
        if hasattr(m, "embeddings"):
            m.embeddings = judge_embeddings

    dataset = _build_ragas_dataset(records)
    logger.info("Running RAGAS evaluation on %d golden samples...", len(records))
    result = evaluate(dataset, metrics=metrics)
    return result


def _run_online_eval() -> dict:
    """
    Live mode: call the real retriever + generator for each golden question,
    capture the pipeline outputs, then score them with RAGAS.
    """
    from src.retrieval import FraudPatternRetriever
    from src.generation import RiskAnalyzerLLM
    from tests.ragas_eval.golden_dataset import GOLDEN_DATASET

    retriever = FraudPatternRetriever()
    generator = RiskAnalyzerLLM()

    live_records = []
    for entry in GOLDEN_DATASET:
        question = entry["question"]
        logger.info("Online eval — querying pipeline for: %s", question[:60])

        # Step 1: real retrieval
        retrieved_contexts = retriever.get_similar_historical_claims(question)

        # Step 2: real generation
        result = generator.analyze_claim(question, retrieved_contexts)
        answer = json.dumps(result)

        live_records.append(
            {
                "question":     question,
                "contexts":     retrieved_contexts,
                "answer":       answer,
                "ground_truth": entry["ground_truth"],
            }
        )

    return _run_offline_eval(live_records)  # reuse scoring logic with live data


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(result: dict):
    """Pretty-print RAGAS scores and flag any metric below threshold."""
    THRESHOLDS = {
        "faithfulness":       0.80,  # hallucination gate — critical for fraud compliance
        "answer_relevancy":   0.75,
        "context_precision":  0.70,
        "context_recall":     0.65,
        "answer_correctness": 0.70,
    }

    print("\n" + "=" * 60)
    print("  RO-Fraud RAGAS Evaluation Report")
    print("=" * 60)

    failed = []
    for metric, threshold in THRESHOLDS.items():
        score = result.get(metric)
        if score is None:
            continue
        status = "✅" if score >= threshold else "❌ BELOW THRESHOLD"
        print(f"  {metric:<25} {score:.3f}   (min: {threshold})  {status}")
        if score < threshold:
            failed.append(metric)

    print("=" * 60)
    if failed:
        print(f"\n⚠️  {len(failed)} metric(s) below threshold: {', '.join(failed)}")
        print("  → Review retrieval corpus freshness and prompt grounding.\n")
    else:
        print("\n✅  All metrics passed thresholds.\n")

    return failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for RO-Fraud")
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        default="offline",
        help="offline: score golden answers (no GCP retrieval/generation calls). "
             "online: run live pipeline then score.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results (e.g. reports/ragas_results.json).",
    )
    args = parser.parse_args()

    if args.mode == "offline":
        from tests.ragas_eval.golden_dataset import GOLDEN_DATASET
        result = _run_offline_eval(GOLDEN_DATASET)
    else:
        result = _run_online_eval()

    failed = _print_report(result)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(dict(result), f, indent=2)
        logger.info("Results written to %s", args.output)

    # Non-zero exit code lets CI fail the build if quality gates are not met
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
