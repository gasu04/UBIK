#!/usr/bin/env python3
"""
UBIK Somatic Node - Golden Set Score Calculator

After human scoring, calculates aggregate metrics and determines pass/fail
against Phase 3 checkpoint criteria.

Usage:
    python golden_set_scorer.py <results_file.json>
    python golden_set_scorer.py <results_file.json> --weak
    python golden_set_scorer.py <results_file.json> --export metrics.json

Scoring Rubric:
    5 - Excellent: Sounds exactly like Gines would say it
    4 - Good: Captures voice well with minor imperfections
    3 - Acceptable: Generic but not wrong, missing personal authenticity
    2 - Poor: Doesn't sound like Gines, wrong tone or themes
    1 - Failure: Completely wrong, shows reasoning, or system error

Author: UBIK Project
Version: 2.1.0 (Phase 3)
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def calculate_scores(results_path: str, export_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate Golden Set evaluation metrics from scored results.

    Args:
        results_path: Path to JSON file with human-scored results
        export_path: Optional path to export metrics JSON

    Returns:
        Dictionary with metrics and pass/fail status
    """
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_file) as f:
        data = json.load(f)

    results: List[Dict[str, Any]] = data.get("results", [])
    if not results:
        raise ValueError("No results found in file")

    # Check for unscored items
    unscored = [r for r in results if r.get("score") is None]
    if unscored:
        print(f"\n⚠ WARNING: {len(unscored)} prompts not yet scored!")
        print("  Unscored IDs:", [r.get("id") for r in unscored[:5]])
        if len(unscored) > 5:
            print(f"  ... and {len(unscored) - 5} more")
        print("\nComplete scoring before calculating final metrics.\n")
        return {"status": "incomplete", "unscored_count": len(unscored)}

    # Extract scores
    scores = [r["score"] for r in results]

    # Group by category
    by_category: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        category = r.get("category", "unknown")
        by_category[category].append(r["score"])

    # Calculate metrics
    total = len(scores)
    avg_score = sum(scores) / total
    hard_failures = sum(1 for s in scores if s == 1)
    high_scores = sum(1 for s in scores if s >= 4)
    high_score_pct = (high_scores / total) * 100

    # Family & Legacy category specifically (critical for Phase 3)
    # Check multiple possible category names
    family_scores = (
        by_category.get("family_legacy", []) or
        by_category.get("family", []) or
        by_category.get("Family & Legacy", [])
    )
    if family_scores:
        family_high = sum(1 for s in family_scores if s >= 4)
        family_high_pct = (family_high / len(family_scores)) * 100
    else:
        family_high_pct = 0.0
        print("⚠ No 'family' or 'family_legacy' category found in results")

    # Per-category averages
    category_metrics = {}
    for cat, cat_scores in by_category.items():
        category_metrics[cat] = {
            "count": len(cat_scores),
            "average": round(sum(cat_scores) / len(cat_scores), 2),
            "high_score_count": sum(1 for s in cat_scores if s >= 4),
            "high_score_pct": round(sum(1 for s in cat_scores if s >= 4) / len(cat_scores) * 100, 1)
        }

    # Determine pass/fail against Phase 3 criteria
    criterion_1 = avg_score >= 3.5
    criterion_2 = hard_failures == 0
    criterion_3 = family_high_pct >= 70.0

    all_passed = criterion_1 and criterion_2 and criterion_3

    # Build report
    metrics = {
        "status": "complete",
        "calculated_at": datetime.now().isoformat(),
        "source_file": str(results_path),
        "overall": {
            "total_prompts": total,
            "average_score": round(avg_score, 2),
            "hard_failures": hard_failures,
            "high_scores": high_scores,
            "high_score_pct": round(high_score_pct, 1)
        },
        "by_category": category_metrics,
        "phase3_criteria": {
            "avg_gte_3_5": {"required": 3.5, "actual": round(avg_score, 2), "passed": criterion_1},
            "no_hard_failures": {"required": 0, "actual": hard_failures, "passed": criterion_2},
            "family_high_pct": {"required": 70.0, "actual": round(family_high_pct, 1), "passed": criterion_3}
        },
        "checkpoint_7_passed": all_passed
    }

    # Print report
    print(f"\n{'=' * 60}")
    print(" GOLDEN SET EVALUATION RESULTS")
    print(f"{'=' * 60}")

    print(f"\n Overall Metrics")
    print(f"{'─' * 40}")
    print(f"  Total Prompts:     {total}")
    print(f"  Average Score:     {avg_score:.2f} / 5.0")
    print(f"  Hard Failures:     {hard_failures}")
    print(f"  High Scores (4-5): {high_scores} ({high_score_pct:.0f}%)")

    print(f"\n By Category")
    print(f"{'─' * 40}")
    for cat, cat_data in sorted(category_metrics.items()):
        print(f"  {cat}:")
        print(f"    Average: {cat_data['average']:.2f}  |  "
              f"High: {cat_data['high_score_count']}/{cat_data['count']} "
              f"({cat_data['high_score_pct']:.0f}%)")

    print(f"\n{'=' * 60}")
    print(" CHECKPOINT 7 CRITERIA")
    print(f"{'=' * 60}")

    def status_icon(passed: bool) -> str:
        return "✓" if passed else "✗"

    print(f"\n  [{status_icon(criterion_1)}] Average ≥ 3.5")
    print(f"      Required: 3.5  |  Actual: {avg_score:.2f}")

    print(f"\n  [{status_icon(criterion_2)}] No Hard Failures (score = 1)")
    print(f"      Required: 0    |  Actual: {hard_failures}")

    print(f"\n  [{status_icon(criterion_3)}] Family & Legacy ≥ 70% High Scores")
    print(f"      Required: 70%  |  Actual: {family_high_pct:.1f}%")

    print(f"\n{'=' * 60}")
    if all_passed:
        print(" ✓ CHECKPOINT 7 PASSED")
        print("   Phase 3 voice evaluation criteria met.")
    else:
        print(" ✗ CHECKPOINT 7 FAILED")
        print("   Review low-scoring responses and iterate.")
    print(f"{'=' * 60}\n")

    # Export if requested
    if export_path:
        export_file = Path(export_path)
        with open(export_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics exported to: {export_file}\n")

    return metrics


def identify_weak_areas(results_path: str, threshold: int = 3) -> List[Dict[str, Any]]:
    """
    Identify prompts scoring below threshold for targeted improvement.

    Args:
        results_path: Path to scored results JSON
        threshold: Score threshold (items below this are flagged)

    Returns:
        List of weak responses
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    weak = [r for r in results if r.get("score") is not None and r["score"] < threshold]

    if not weak:
        print(f"\n✓ No responses scored below {threshold}")
        return []

    print(f"\n{'=' * 60}")
    print(f" WEAK AREAS (score < {threshold})")
    print(f"{'=' * 60}")

    for r in sorted(weak, key=lambda x: x.get("score", 0)):
        print(f"\n[Score: {r['score']}] {r.get('category', 'unknown')} - {r.get('id', 'unknown')}")
        print(f"  Prompt: {r.get('prompt', '')[:70]}...")
        print(f"  Rationale: {r.get('score_rationale', 'No rationale provided')}")
        if r.get("response"):
            print(f"  Response preview: {r['response'][:100]}...")

    print(f"\n{'=' * 60}")
    print(f" Total weak responses: {len(weak)}")
    print(f"{'=' * 60}\n")

    return weak


def score_distribution(results_path: str) -> Dict[int, int]:
    """
    Show score distribution histogram.

    Args:
        results_path: Path to scored results JSON

    Returns:
        Dictionary mapping scores to counts
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    distribution = defaultdict(int)

    for r in results:
        score = r.get("score")
        if score is not None:
            distribution[score] += 1

    print(f"\n{'=' * 60}")
    print(" SCORE DISTRIBUTION")
    print(f"{'=' * 60}\n")

    max_count = max(distribution.values()) if distribution else 1
    bar_width = 40

    for score in range(5, 0, -1):
        count = distribution.get(score, 0)
        bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "█" * bar_length
        label = {5: "Excellent", 4: "Good", 3: "Acceptable", 2: "Poor", 1: "Failure"}[score]
        print(f"  {score} ({label:10}) │{bar:<{bar_width}} {count}")

    print(f"\n{'=' * 60}\n")

    return dict(distribution)


def compare_runs(path1: str, path2: str) -> None:
    """
    Compare metrics between two evaluation runs.

    Args:
        path1: Path to first results file
        path2: Path to second results file
    """
    with open(path1) as f:
        data1 = json.load(f)
    with open(path2) as f:
        data2 = json.load(f)

    results1 = [r for r in data1.get("results", []) if r.get("score") is not None]
    results2 = [r for r in data2.get("results", []) if r.get("score") is not None]

    if not results1 or not results2:
        print("⚠ One or both files have no scored results")
        return

    avg1 = sum(r["score"] for r in results1) / len(results1)
    avg2 = sum(r["score"] for r in results2) / len(results2)

    failures1 = sum(1 for r in results1 if r["score"] == 1)
    failures2 = sum(1 for r in results2 if r["score"] == 1)

    high1 = sum(1 for r in results1 if r["score"] >= 4)
    high2 = sum(1 for r in results2 if r["score"] >= 4)

    print(f"\n{'=' * 60}")
    print(" RUN COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n  Run 1: {Path(path1).name}")
    print(f"  Run 2: {Path(path2).name}")

    print(f"\n{'─' * 40}")
    print(f"  {'Metric':<20} {'Run 1':>10} {'Run 2':>10} {'Delta':>10}")
    print(f"{'─' * 40}")

    def delta_str(v1: float, v2: float) -> str:
        d = v2 - v1
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.2f}"

    print(f"  {'Average Score':<20} {avg1:>10.2f} {avg2:>10.2f} {delta_str(avg1, avg2):>10}")
    print(f"  {'Hard Failures':<20} {failures1:>10} {failures2:>10} {delta_str(failures1, failures2):>10}")
    print(f"  {'High Scores (4-5)':<20} {high1:>10} {high2:>10} {delta_str(high1, high2):>10}")

    print(f"\n{'=' * 60}\n")


def print_usage():
    """Print usage information."""
    print("""
Golden Set Score Calculator

Usage:
    python golden_set_scorer.py <results_file.json>
        Calculate metrics and check Phase 3 criteria

    python golden_set_scorer.py <results_file.json> --weak
        Identify responses scoring below threshold

    python golden_set_scorer.py <results_file.json> --distribution
        Show score distribution histogram

    python golden_set_scorer.py <results_file.json> --export <output.json>
        Export metrics to JSON file

    python golden_set_scorer.py --compare <file1.json> <file2.json>
        Compare metrics between two runs

Scoring Rubric:
    5 - Excellent: Sounds exactly like Gines would say it
    4 - Good: Captures voice well with minor imperfections
    3 - Acceptable: Generic but not wrong
    2 - Poor: Doesn't sound like Gines
    1 - Failure: Completely wrong or system error
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    if sys.argv[1] == "--compare":
        if len(sys.argv) < 4:
            print("Usage: python golden_set_scorer.py --compare <file1.json> <file2.json>")
            sys.exit(1)
        compare_runs(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print_usage()

    else:
        results_path = sys.argv[1]

        if "--weak" in sys.argv:
            threshold = 3
            # Check for custom threshold
            for i, arg in enumerate(sys.argv):
                if arg == "--threshold" and i + 1 < len(sys.argv):
                    threshold = int(sys.argv[i + 1])
            identify_weak_areas(results_path, threshold)

        elif "--distribution" in sys.argv:
            score_distribution(results_path)

        elif "--export" in sys.argv:
            export_idx = sys.argv.index("--export")
            if export_idx + 1 < len(sys.argv):
                export_path = sys.argv[export_idx + 1]
                calculate_scores(results_path, export_path)
            else:
                print("Error: --export requires output path")
                sys.exit(1)

        else:
            calculate_scores(results_path)
