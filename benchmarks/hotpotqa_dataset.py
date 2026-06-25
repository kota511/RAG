from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping

# dataset repo to exact Git commit for consistent benchmark
DATASET_REPOSITORY = "hotpotqa/hotpot_qa"
DATASET_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
DATASET_FILE_PATH = "distractor/validation-00000-of-00001.parquet"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
    f"{DATASET_REVISION}/{DATASET_FILE_PATH}"
)
DEFAULT_CASE_COUNT = 50
DEFAULT_SELECTION_SEED = 5114


@dataclass(frozen=True)
class BenchmarkDocument:
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class HotpotQACase:
    dataset_id: str
    question: str
    answer: str
    documents: tuple[BenchmarkDocument, ...]
    supporting_document_titles: tuple[str, ...]


def load_hotpotqa_cases(
    count: int = DEFAULT_CASE_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
) -> list[HotpotQACase]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install benchmark dependencies with 'uv sync --group benchmark'."
        ) from error

    rows = load_dataset(
        "parquet",
        data_files={"validation": DATASET_URL},
        split="validation",
    )
    return select_hotpotqa_cases(rows, count=count, seed=seed)


def select_hotpotqa_cases(
    rows: Iterable[Mapping[str, Any]],
    count: int,
    seed: int,
) -> list[HotpotQACase]:
    if count <= 0:
        raise ValueError("count must be greater than zero")

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            sha256(f"{seed}:{row['id']}".encode()).digest(),
            row["id"],
        ),
    )
    if count > len(ordered_rows):
        raise ValueError("count cannot exceed the number of dataset rows")

    return [
        create_hotpotqa_case(row)
        for row in ordered_rows[:count]
    ]


def create_hotpotqa_case(row: Mapping[str, Any]) -> HotpotQACase:
    context = row["context"]
    documents = tuple(
        BenchmarkDocument(
            title=title,
            sentences=tuple(sentences),
        )
        for title, sentences in zip(
            context["title"],
            context["sentences"],
            strict=True,
        )
    )
    supporting_document_titles = tuple(
        dict.fromkeys(row["supporting_facts"]["title"])
    )

    return HotpotQACase(
        dataset_id=row["id"],
        question=row["question"],
        answer=row["answer"],
        documents=documents,
        supporting_document_titles=supporting_document_titles,
    )


def main() -> None:
    cases = load_hotpotqa_cases()
    print(
        f"Loaded {len(cases)} deterministic HotpotQA cases "
        f"from revision {DATASET_REVISION}."
    )
    for case in cases:
        print(case.dataset_id)


if __name__ == "__main__":
    main()
