from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved_ids[:k] if item in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for item in retrieved_ids[:k] if item in relevant)
    return hits / k


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    relevant = set(relevant_ids)
    for index, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def normalized_discounted_cumulative_gain(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    dcg = 0.0
    for index, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant), k)
    if not ideal_hits:
        return 0.0

    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@dataclass(slots=True)
class RetrievalEvaluation:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float


def evaluate_retrieval(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        recall_at_1=recall_at_k(retrieved_ids, relevant_ids, 1),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
        ndcg_at_5=normalized_discounted_cumulative_gain(retrieved_ids, relevant_ids, 5),
    )

