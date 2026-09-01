from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .evaluation import RetrievalEvaluation, evaluate_retrieval
from .models import FaithfulnessReport, HallucinationReport
from .pipeline import RagPipeline
from .text import meaningful_terms


@dataclass(slots=True)
class EvaluationCase:
    name: str
    question: str
    relevant_chunk_ids: list[str]
    expected_answer_contains: str | None = None
    expects_no_answer: bool = False


@dataclass(slots=True)
class EvaluationResult:
    case: EvaluationCase
    retrieved_ids: list[str]
    retrieval: RetrievalEvaluation | None
    answer: str
    faithfulness: FaithfulnessReport
    hallucination: HallucinationReport
    context_relevance: float
    answer_correctness: float
    latency_ms: float
    estimated_token_usage: int
    passed: bool
    reason: str


@dataclass(slots=True)
class EvaluationSummary:
    total: int
    passed: int
    failed: int
    mean_recall_at_1: float
    mean_recall_at_5: float
    mean_mrr: float
    mean_ndcg_at_5: float
    faithfulness_pass_rate: float
    hallucination_rate: float
    mean_context_relevance: float
    answer_correctness_rate: float
    mean_latency_ms: float
    mean_estimated_token_usage: float


DEFAULT_EVALUATION_CASES = [
    EvaluationCase(
        name="annual_leave_entitlement",
        question="What is the annual leave entitlement?",
        relevant_chunk_ids=["employee_handbook_sample_chunk_0001"],
        expected_answer_contains="20 days of annual leave",
    ),
    EvaluationCase(
        name="unused_annual_leave",
        question="What happens if an employee doesn't use their annual leave?",
        relevant_chunk_ids=["employee_handbook_sample_chunk_0001"],
        expected_answer_contains="carried over up to 5 days",
    ),
    EvaluationCase(
        name="mars_policy",
        question="What is the company's policy for employees working on Mars?",
        relevant_chunk_ids=[],
        expects_no_answer=True,
    ),
]


def load_cases_from_jsonl(path: str | Path) -> list[EvaluationCase]:
    case_path = Path(path)
    cases: list[EvaluationCase] = []
    with case_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            cases.append(
                EvaluationCase(
                    name=str(payload["name"]),
                    question=str(payload["question"]),
                    relevant_chunk_ids=list(payload.get("relevant_chunk_ids", [])),
                    expected_answer_contains=payload.get("expected_answer_contains"),
                    expects_no_answer=bool(payload.get("expects_no_answer", False)),
                )
            )
    return cases


def run_evaluation(pipeline: RagPipeline, cases: Sequence[EvaluationCase]) -> tuple[list[EvaluationResult], EvaluationSummary]:
    results: list[EvaluationResult] = []
    recall_at_1_total = 0.0
    recall_at_5_total = 0.0
    mrr_total = 0.0
    ndcg_total = 0.0
    retrieval_case_count = 0
    faithfulness_passes = 0
    hallucination_hits = 0
    context_relevance_total = 0.0
    answer_correctness_total = 0.0
    latency_total = 0.0
    token_usage_total = 0

    for case in cases:
        response = pipeline.answer(case.question)
        retrieved_ids = [citation.chunk_id for citation in response.retrieval]
        retrieval = None
        if case.relevant_chunk_ids:
            retrieval = evaluate_retrieval(retrieved_ids, case.relevant_chunk_ids)
            recall_at_1_total += retrieval.recall_at_1
            recall_at_5_total += retrieval.recall_at_5
            mrr_total += retrieval.mrr
            ndcg_total += retrieval.ndcg_at_5
            retrieval_case_count += 1

        query_terms = set(meaningful_terms(case.question))
        context_terms = set()
        prompt_context = response.prompt
        if "Context:\n" in prompt_context and "\n\nQuestion:\n" in prompt_context:
            prompt_context = prompt_context.split("Context:\n", 1)[1].split("\n\nQuestion:\n", 1)[0]
        context_terms.update(meaningful_terms(prompt_context))
        context_relevance = len(query_terms & context_terms) / len(query_terms) if query_terms else 0.0
        answer_correctness = 1.0
        if case.expected_answer_contains is not None:
            answer_correctness = float(case.expected_answer_contains.lower() in response.answer.lower())
        elif case.expects_no_answer:
            answer_correctness = float("i don't have enough information" in response.answer.lower())
        latency_total += response.latency_ms
        estimated_token_usage = len(response.prompt.split()) + len(response.answer.split())
        token_usage_total += estimated_token_usage
        context_relevance_total += context_relevance
        answer_correctness_total += answer_correctness

        passed = True
        reason_parts: list[str] = []
        if case.expected_answer_contains is not None and case.expected_answer_contains.lower() not in response.answer.lower():
            passed = False
            reason_parts.append("answer text did not contain the expected phrase")
        if case.expects_no_answer and "i don't have enough information" not in response.answer.lower():
            passed = False
            reason_parts.append("model did not decline to answer")
        if not response.faithfulness.passed:
            passed = False
            reason_parts.append(f"faithfulness failed: {response.faithfulness.reason}")

        if response.faithfulness.passed:
            faithfulness_passes += 1
        if response.hallucination.detected:
            hallucination_hits += 1

        results.append(
            EvaluationResult(
                case=case,
                retrieved_ids=retrieved_ids,
                retrieval=retrieval,
                answer=response.answer,
                faithfulness=response.faithfulness,
                hallucination=response.hallucination,
                context_relevance=context_relevance,
                answer_correctness=answer_correctness,
                latency_ms=response.latency_ms,
                estimated_token_usage=estimated_token_usage,
                passed=passed,
                reason="; ".join(reason_parts) if reason_parts else "passed",
            )
        )

    total = len(cases)
    summary = EvaluationSummary(
        total=total,
        passed=sum(1 for result in results if result.passed),
        failed=sum(1 for result in results if not result.passed),
        mean_recall_at_1=(recall_at_1_total / retrieval_case_count) if retrieval_case_count else 0.0,
        mean_recall_at_5=(recall_at_5_total / retrieval_case_count) if retrieval_case_count else 0.0,
        mean_mrr=(mrr_total / retrieval_case_count) if retrieval_case_count else 0.0,
        mean_ndcg_at_5=(ndcg_total / retrieval_case_count) if retrieval_case_count else 0.0,
        faithfulness_pass_rate=(faithfulness_passes / total) if total else 0.0,
        hallucination_rate=(hallucination_hits / total) if total else 0.0,
        mean_context_relevance=(context_relevance_total / total) if total else 0.0,
        answer_correctness_rate=(answer_correctness_total / total) if total else 0.0,
        mean_latency_ms=(latency_total / total) if total else 0.0,
        mean_estimated_token_usage=(token_usage_total / total) if total else 0.0,
    )
    return results, summary


def format_evaluation_results(results: Sequence[EvaluationResult], summary: EvaluationSummary) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"Case: {result.case.name}")
        lines.append(f"Question: {result.case.question}")
        lines.append(f"Passed: {'yes' if result.passed else 'no'}")
        if result.retrieval is not None:
            lines.append(
                "Retrieval: "
                f"recall@1={result.retrieval.recall_at_1:.2f} "
                f"recall@5={result.retrieval.recall_at_5:.2f} "
                f"mrr={result.retrieval.mrr:.2f} "
                f"ndcg@5={result.retrieval.ndcg_at_5:.2f}"
            )
        lines.append(f"Faithfulness: {'passed' if result.faithfulness.passed else 'failed'} ({result.faithfulness.reason})")
        lines.append(
            f"Hallucination: {'detected' if result.hallucination.detected else 'not detected'} "
            f"(severity={result.hallucination.severity:.2f}, coverage={result.hallucination.coverage:.2f})"
        )
        lines.append(f"Hallucination reason: {result.hallucination.reason}")
        lines.append(
            f"Quality: context relevance={result.context_relevance:.2f} "
            f"answer correctness={result.answer_correctness:.2f}"
        )
        lines.append(f"Performance: latency={result.latency_ms:.1f}ms estimated tokens={result.estimated_token_usage}")
        lines.append(f"Answer: {result.answer}")
        lines.append(f"Reason: {result.reason}")
        lines.append("")

    lines.append("Summary")
    lines.append(
        f"Passed {summary.passed}/{summary.total} | failed {summary.failed} | faithfulness pass rate {summary.faithfulness_pass_rate:.2f}"
    )
    lines.append(f"Hallucination detection rate: {summary.hallucination_rate:.2f}")
    lines.append(
        f"Mean quality: context relevance={summary.mean_context_relevance:.2f} "
        f"answer correctness={summary.answer_correctness_rate:.2f}"
    )
    lines.append(
        f"Mean performance: latency={summary.mean_latency_ms:.1f}ms "
        f"estimated tokens={summary.mean_estimated_token_usage:.1f}"
    )
    lines.append(
        f"Mean retrieval: recall@1={summary.mean_recall_at_1:.2f} recall@5={summary.mean_recall_at_5:.2f} "
        f"mrr={summary.mean_mrr:.2f} ndcg@5={summary.mean_ndcg_at_5:.2f}"
    )
    return "\n".join(lines).strip()
