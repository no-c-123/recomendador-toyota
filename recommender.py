"""Toyota KMS - Document recommender (Path A: TF-IDF + cosine similarity).

Usage:
    python recommender.py                      # interactive menu
    python recommender.py --id DOC001          # Top-5 similar to a seed document
    python recommender.py --text "free query"  # Top-5 for a query text
    python recommender.py --demo               # run the three evidence tests
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_PATH = Path(__file__).with_name("corpus.csv")
TOP_K = 5
REQUIRED_COLUMNS = {"id", "title", "text"}
LINE = "=" * 72


def clean_text(text: str) -> str:
    """Lowercase and keep only letters, digits and spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Recommendation:
    rank: int
    doc_id: str
    title: str
    category: str
    score: float


class DocumentRecommender:
    """Content-based recommender over a small document corpus."""

    def __init__(self, corpus: pd.DataFrame):
        missing = REQUIRED_COLUMNS - set(corpus.columns)
        if missing:
            raise ValueError(f"corpus is missing columns: {sorted(missing)}")
        self.corpus = corpus.reset_index(drop=True)
        if "category" not in self.corpus:
            self.corpus["category"] = "-"
        # Title is included because it summarizes the document's topic.
        documents = (self.corpus["title"] + ". " + self.corpus["text"]).map(clean_text)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(documents)
        self.similarity = cosine_similarity(self.matrix)

    @classmethod
    def from_csv(cls, path: Path = CORPUS_PATH) -> "DocumentRecommender":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run build_corpus.py first.")
        return cls(pd.read_csv(path))

    def _rank(self, scores, exclude_index: int | None = None, k: int = TOP_K):
        order = scores.argsort()[::-1]
        results = []
        for idx in order:
            if idx == exclude_index:
                continue  # never recommend the seed document itself
            row = self.corpus.iloc[idx]
            results.append(Recommendation(len(results) + 1, row["id"], row["title"],
                                          row["category"], float(scores[idx])))
            if len(results) == k:
                break
        return results

    def _index_of(self, doc_id: str) -> int:
        matches = self.corpus.index[self.corpus["id"].str.upper() == doc_id.strip().upper()]
        if len(matches) == 0:
            raise KeyError(f"document '{doc_id.strip()}' does not exist")
        return int(matches[0])

    def recommend_by_id(self, doc_id: str, k: int = TOP_K) -> list[Recommendation]:
        idx = self._index_of(doc_id)
        return self._rank(self.similarity[idx], exclude_index=idx, k=k)

    def recommend_by_text(self, query: str, k: int = TOP_K) -> list[Recommendation]:
        cleaned = clean_text(query)
        if not cleaned:
            raise ValueError("query text is empty")
        query_vector = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        return self._rank(scores, k=k)

    def describe(self, doc_id: str) -> str:
        row = self.corpus.iloc[self._index_of(doc_id)]
        return f"{row['id']} | {row['category']} | {row['title']}"


def print_results(results: list[Recommendation]) -> None:
    print(f"{'#':<3}{'ID':<8}{'Score':<8}{'Category':<19}Title")
    print("-" * 72)
    for r in results:
        title = r.title if len(r.title) <= 40 else r.title[:37] + "..."
        print(f"{r.rank:<3}{r.doc_id:<8}{r.score:<8.3f}{r.category:<19}{title}")


def run_id(rec: DocumentRecommender, doc_id: str) -> None:
    print(f"Input (seed ID): {rec.describe(doc_id)}\n")
    print_results(rec.recommend_by_id(doc_id))


def run_text(rec: DocumentRecommender, text: str) -> None:
    print(f'Input (query text): "{text}"\n')
    print_results(rec.recommend_by_text(text))


def demo(rec: DocumentRecommender) -> None:
    tests = [
        ("TEST 1 - Seed document from Policy/Process", "id", "DOC001"),
        ("TEST 2 - Seed document from Support/Incidents", "id", "DOC011"),
        ("TEST 3 - Free text query", "text",
         "the supplier sent parts late and the line may stop, what should I do"),
    ]
    for title, mode, value in tests:
        print(LINE)
        print(title)
        print(LINE)
        run_id(rec, value) if mode == "id" else run_text(rec, value)
        print()


def interactive(rec: DocumentRecommender) -> None:
    while True:
        print("\n1) Recommend by document ID   2) Recommend by text   3) Exit")
        choice = input("Option: ").strip()
        try:
            if choice == "1":
                run_id(rec, input("Document ID (e.g. DOC001): "))
            elif choice == "2":
                run_text(rec, input("Paste a short text: "))
            elif choice == "3":
                break
            else:
                print("Invalid option.")
        except (KeyError, ValueError) as err:
            print(f"Error: {err.args[0]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Toyota KMS document recommender (TF-IDF)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--id", help="seed document ID")
    group.add_argument("--text", help="free text query")
    group.add_argument("--demo", action="store_true", help="run the three evidence tests")
    args = parser.parse_args(argv)

    rec = DocumentRecommender.from_csv()
    print(LINE)
    print("TOYOTA KMS - DOCUMENT RECOMMENDER (TF-IDF + COSINE SIMILARITY)")
    print(LINE)
    counts = rec.corpus["category"].value_counts()
    print(f"Documents loaded: {len(rec.corpus)} | Vocabulary: {rec.matrix.shape[1]} terms")
    for cat, n in counts.items():
        print(f"  {cat:<18}{n} documents")
    print()

    try:
        if args.id:
            run_id(rec, args.id)
        elif args.text:
            run_text(rec, args.text)
        elif args.demo:
            demo(rec)
        else:
            interactive(rec)
    except (KeyError, ValueError) as err:
        print(f"Error: {err.args[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
