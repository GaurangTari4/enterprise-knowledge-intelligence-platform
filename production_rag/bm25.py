from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Callable, Sequence

from .text import tokenize


class BM25Index:
    def __init__(
        self,
        documents: Sequence[str],
        tokenizer: Callable[[str], list[str]] = tokenize,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.documents = list(documents)
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        self.document_frequency: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for index, document in enumerate(self.documents):
            terms = self._tokenizer(document)
            frequencies = Counter(terms)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(terms))
            for term, frequency in frequencies.items():
                self.document_frequency[term] += 1
                self.postings[term].append((index, frequency))

        self.document_count = len(self.documents)
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count if self.document_count else 0.0
        )

    def score_all(self, query: str) -> list[float]:
        scores = [0.0] * self.document_count
        if not self.document_count:
            return scores

        query_terms = Counter(term for term in self._tokenizer(query) if term)
        if not query_terms:
            return scores

        for term, query_frequency in query_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue

            document_frequency = self.document_frequency[term]
            idf = math.log(1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for document_index, term_frequency in postings:
                length = self.document_lengths[document_index] or 1
                normalization = self.k1 * (
                    1 - self.b + self.b * (length / self.average_document_length if self.average_document_length else 1.0)
                )
                denominator = term_frequency + normalization
                scores[document_index] += idf * ((term_frequency * (self.k1 + 1)) / denominator) * query_frequency

        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        scores = self.score_all(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:top_k] if score > 0]

