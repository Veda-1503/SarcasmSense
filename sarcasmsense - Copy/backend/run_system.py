import os
import sys
import argparse

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.utils import logging
logging.set_verbosity_error()


def _check_and_finetune():
    from config import FINE_TUNED_MODELS, SENTIMENT_MODEL_PATH, DOMAINS
    from finetune import finetune_missing
    sentiment_missing = not os.path.isdir(SENTIMENT_MODEL_PATH)
    domain_missing = any(
        not os.path.isdir(FINE_TUNED_MODELS[d]["path"]) for d in DOMAINS
    )
    if sentiment_missing or domain_missing:
        print("[AUTO] One or more models are missing. Running fine-tuning...")
        finetune_missing()


def _print_realtime_output(result: dict):
    print("\n" + "=" * 120)
    print("🔍 REALTIME OUTPUT")
    print("=" * 120)

    for i, sr in enumerate(result["sentence_results"], 1):
        print(f"\nSentence {i}:")
        print(f"  {'Clause':<15}: {sr['clause']}")
        sarc = sr["sarcasm"]
        print(f"  {'Sarcasm':<15}: {sarc['label']} ({sarc['confidence']:.2f})")
        print(f"  {'Sarcasm Level':<15}: {sarc['level']}")
        sent = sr["sentiment"]
        print(f"  {'Sentiment':<15}: {sent['base_label']} ({sent['base_confidence']:.2f})")
        print(f"  {'Explanation':<15}: {sr['explanation']}")

    print(f"\n{'Overall Sarcasm':<20}: {'Yes' if result['overall_sarcasm'] else 'No'}")
    print(f"{'Final Sentiment':<20}: {result['final_sentiment']} ({result['final_confidence']:.2f})")
    print(f"{'Model Used':<20}: {result['model_used']}")
    print("=" * 120)


def run_realtime():
    _check_and_finetune()
    from main_pipeline import run_pipeline
    from domain_classifier import train_domain_classifier

    print("\n" + "=" * 120)
    print("  MULTI-STAGE SARCASM DETECTION WITH CLAUSE BASED SENTIMENT CLASSIFICATION")
    print("  Real-Time Mode | Type 'quit' to exit")
    print("=" * 120)

    while True:
        try:
            user_input = input("\nEnter review (or 'quit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[EXIT] Goodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("[EXIT] Goodbye!")
            break

        if not user_input:
            continue

        domain_input = input("Enter domain (apps/movies/hotels/restaurants/ecommerce/healthcare) [auto]: ").strip().lower()
        if domain_input not in ("apps", "movies", "hotels", "restaurants", "ecommerce", "healthcare", ""):
            print("[WARN] Invalid domain. Using auto-detection.")
            domain_input = None
        elif domain_input == "":
            domain_input = None

        try:
            result = run_pipeline(user_input, domain=domain_input)
            _print_realtime_output(result)
        except Exception as e:
            print(f"[ERROR] {e}")


def run_evaluation_mode():
    _check_and_finetune()
    from evaluation import run_evaluation
    run_evaluation()


def main():
    parser = argparse.ArgumentParser(
        description="Sarcasm-Aware Sentiment Analysis System"
    )
    parser.add_argument(
        "mode",
        choices=["realtime", "evaluate", "finetune"],
        nargs="?",
        default="realtime",
        help="Mode to run: realtime (default), evaluate, finetune"
    )
    parser.add_argument(
        "--domain",
        choices=["all", "sentiment", "apps", "movies", "hotels", "restaurants", "ecommerce", "healthcare"],
        default="all",
        help="Domain to fine-tune (only for finetune mode)"
    )
    args = parser.parse_args()

    if args.mode == "realtime":
        run_realtime()
    elif args.mode == "evaluate":
        run_evaluation_mode()
    elif args.mode == "finetune":
        from finetune import (
            finetune_all, finetune_sarcasm_model, finetune_sentiment_model
        )
        if args.domain == "all":
            finetune_all()
        elif args.domain == "sentiment":
            finetune_sentiment_model()
        else:
            finetune_sarcasm_model(args.domain)


if __name__ == "__main__":
    main()
