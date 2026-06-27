import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_REPORT_PATH = Path("benchmark_results/hotpotqa_retrieval.json")


def summarize_report(report: dict[str, Any], example_count: int = 5) -> str:
    cases = report["cases"]
    if not cases:
        raise ValueError("report must contain at least one case")

    complete = 0
    partial = 0
    zero_recall = 0
    duplicate_cases = 0
    incomplete_duplicate_cases = 0
    incomplete_cases = []

    for case in cases:
        metrics = case["metrics"]
        titles = retrieved_titles(case)
        has_duplicates = bool(duplicate_titles(titles))

        if has_duplicates:
            duplicate_cases += 1

        if metrics["complete_support"]:
            complete += 1
        else:
            incomplete_cases.append(case)
            if has_duplicates:
                incomplete_duplicate_cases += 1
            if metrics["matched_supporting_document_count"] == 0:
                zero_recall += 1
            else:
                partial += 1

    total = len(cases)
    incomplete_total = partial + zero_recall
    lines = [
        "HotpotQA retrieval failure analysis",
        f"Cases: {total}",
        f"Retrieval limit: {report['aggregate']['limit']}",
        f"Complete: {complete} ({rate(complete, total)})",
        f"Partial: {partial} ({rate(partial, total)})",
        f"Zero recall: {zero_recall} ({rate(zero_recall, total)})",
        (
            "Cases with duplicate retrieved document titles: "
            f"{duplicate_cases} ({rate(duplicate_cases, total)})"
        ),
        (
            "Incomplete cases with duplicate retrieved document titles: "
            f"{incomplete_duplicate_cases} "
            f"({rate(incomplete_duplicate_cases, incomplete_total)} "
            "of incomplete cases)"
        ),
    ]

    worst_cases = sorted(
        incomplete_cases,
        key=lambda case: (
            case["metrics"]["matched_supporting_document_count"],
            case["metrics"]["reciprocal_rank"],
            case["dataset_id"],
        ),
    )[:example_count]
    if worst_cases:
        lines.extend(["", "Worst incomplete examples:"])

    for index, case in enumerate(worst_cases, start=1):
        metrics = case["metrics"]
        support = case["supporting_document_titles"]
        titles = retrieved_titles(case)
        missed = [title for title in support if title not in set(titles)]
        duplicates = duplicate_titles(titles)
        lines.extend(
            [
                (
                    f"{index}. {case['dataset_id']}: "
                    f"matched {metrics['matched_supporting_document_count']}/"
                    f"{metrics['supporting_document_count']}, "
                    f"mrr={metrics['reciprocal_rank']:.3f}"
                ),
                f"   question: {case['question']}",
                f"   expected support: {'; '.join(support)}",
                f"   missed support: {'; '.join(missed)}",
                f"   retrieved titles: {' | '.join(titles)}",
                (
                    "   duplicate retrieved titles: "
                    f"{'; '.join(duplicates) or 'none'}"
                ),
            ]
        )

    return "\n".join(lines)


def retrieved_titles(case: dict[str, Any]) -> list[str]:
    return [chunk["document_title"] for chunk in case["retrieved_chunks"]]


def duplicate_titles(titles: list[str]) -> list[str]:
    counts = Counter(titles)
    seen = set()
    duplicates = []
    for title in titles:
        if counts[title] > 1 and title not in seen:
            seen.add(title)
            duplicates.append(title)
    return duplicates


def rate(count: int, total: int) -> str:
    return "n/a" if total == 0 else f"{count / total:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze saved HotpotQA retrieval benchmark failures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to a saved hotpotqa_retrieval JSON report.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of incomplete cases to print.",
    )
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    print(summarize_report(report, args.examples))


if __name__ == "__main__":
    main()
