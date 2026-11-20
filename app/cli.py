"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import json
from pathlib import Path
from typing import List

from app.core import (
    milestone_1,
    milestone_2,
    milestone_4,
    milestone_5,
    verbalize_transaction_sequence,
)


def _parse_prefix(prefix_arg: str | None) -> List[str] | None:
    if not prefix_arg:
        return None
    # Support comma, semicolon, or JSON array format
    if prefix_arg.startswith("[") and prefix_arg.endswith("]"):
        try:
            return json.loads(prefix_arg)
        except Exception:
            pass
    delimiter = ";" if ";" in prefix_arg else ","
    return [item.strip() for item in prefix_arg.split(delimiter) if item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="transaction-pattern-simulator",
        description="Financial analysts simulate sequences of customer transactions to test fraud detection systems or model customer behavior patterns.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    # generate command (Milestone 5)
    gen_parser = sub.add_parser("generate", help="Generate plausible transaction sequences from a starting set of transactions")
    gen_parser.add_argument("--prefix", "-p", type=str, default=None, help="Starting set of transactions (comma or semicolon separated)")
    gen_parser.add_argument("--count", "-n", type=int, default=1, help="Number of customer sequences to generate")
    gen_parser.add_argument("--min-length", type=int, default=4, help="Constraint: minimum length of generated sequences")
    gen_parser.add_argument("--max-length", type=int, default=None, help="Constraint: maximum length of generated sequences")
    gen_parser.add_argument("--must-include", type=str, default=None, help="Constraint: required transaction type in sequence")
    gen_parser.add_argument("--temperature", "-t", type=float, default=1.0, help="Sampling temperature")
    gen_parser.add_argument("--output", "-o", type=str, default=None, help="Save generated sequences to a JSON file")
    gen_parser.add_argument("--verbalized", action="store_true", help="Print in LM verbalized format")

    # mine command (Milestone 2)
    mine_parser = sub.add_parser("mine", help="Discover frequent transaction patterns using thilimarketbasket")
    mine_parser.add_argument("--min-support", type=float, default=0.08, help="Minimum support threshold")
    mine_parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum confidence threshold for rules")

    # evaluate command (Milestone 4)
    eval_parser = sub.add_parser("evaluate", help="Evaluate generated sequences against real transaction patterns")
    eval_parser.add_argument("--min-support", type=float, default=0.08, help="Minimum support threshold")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Transaction Pattern Simulator is ready. Open README.md for your milestones.")

    elif args.command == "generate":
        prefix = _parse_prefix(args.prefix)
        sequences = milestone_5(
            prefix=prefix,
            num_sequences=args.count,
            min_length=args.min_length,
            max_length=args.max_length,
            must_include=args.must_include,
            output_path=args.output,
            temperature=args.temperature,
        )
        print(f"Generated {len(sequences)} sequence(s):")
        for i, seq in enumerate(sequences, start=1):
            if args.verbalized:
                print(f"[{i}] {verbalize_transaction_sequence(seq)}")
            else:
                print(f"[{i}] {' -> '.join(seq)}")
        if args.output:
            print(f"Saved generated sequence(s) to: {args.output}")

    elif args.command == "mine":
        results = milestone_2(min_support=args.min_support, min_confidence=args.min_confidence)
        summary = results["summary"]
        print(f"Total sequences analyzed: {summary['total_sequences']}")
        print(f"Frequent patterns found: {summary['num_patterns']} (Multi-event patterns: {summary['num_multi_event_patterns']})")
        print(f"Association rules discovered: {summary['num_rules']}")
        print("\nTop frequent transaction patterns:")
        for pat in summary["top_patterns"][:5]:
            print(f"  - Support {pat['support']:.3f}: {' + '.join(pat['pattern'])}")

    elif args.command == "evaluate":
        res = milestone_4(min_support=args.min_support)
        metrics = res["metrics"]
        print("=== Pattern Similarity Evaluation ===")
        print(f"Overall Similarity Score:       {metrics['overall_similarity_score']:.4f}")
        print(f"Pattern Jaccard Similarity:     {metrics['jaccard_similarity']:.4f}")
        print(f"Pattern F1 Score:               {metrics['f1_score']:.4f}")
        print(f"Support Correlation:            {metrics['support_correlation']:.4f}")
        print(f"Mean Absolute Support Error:    {metrics['mean_absolute_support_error']:.4f}")
        print(f"Distribution Similarity (TVD):  {metrics['distribution_similarity']:.4f}")
        print(f"Shared Patterns Count:          {res['counts']['shared_patterns']}")


if __name__ == "__main__":
    main()
