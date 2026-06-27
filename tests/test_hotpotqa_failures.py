from benchmarks.hotpotqa_failures import (
    duplicate_titles,
    summarize_report,
)


def report_case(
    dataset_id: str,
    supporting_titles: list[str],
    retrieved_titles: list[str],
    matched_count: int,
    complete: bool,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "question": f"Question for {dataset_id}?",
        "supporting_document_titles": supporting_titles,
        "metrics": {
            "supporting_document_count": len(supporting_titles),
            "matched_supporting_document_count": matched_count,
            "complete_support": complete,
            "reciprocal_rank": 1.0 if matched_count else 0.0,
        },
        "retrieved_chunks": [
            {"document_title": title}
            for title in retrieved_titles
        ],
    }


def test_analyzes_saved_retrieval_report() -> None:
    report = {
        "aggregate": {"limit": 5},
        "cases": [
            report_case("complete", ["A", "B"], ["A", "B"], 2, True),
            report_case("partial", ["C", "D"], ["C", "C", "X"], 1, False),
            report_case("zero", ["E", "F"], ["X", "Y"], 0, False),
        ],
    }

    output = summarize_report(report, example_count=2)

    assert "Complete: 1 (33.3%)" in output
    assert "Partial: 1 (33.3%)" in output
    assert "Zero recall: 1 (33.3%)" in output
    assert "Cases with duplicate retrieved document titles: 1 (33.3%)" in output
    assert (
        "Incomplete cases with duplicate retrieved document titles: "
        "1 (50.0% of incomplete cases)"
    ) in output
    assert "1. zero: matched 0/2, mrr=0.000" in output
    assert "missed support: E; F" in output
    assert "2. partial: matched 1/2, mrr=1.000" in output
    assert "duplicate retrieved titles: C" in output


def test_duplicate_titles_are_reported_once_in_rank_order() -> None:
    assert duplicate_titles(["A", "B", "A", "A", "C", "B"]) == ["A", "B"]
