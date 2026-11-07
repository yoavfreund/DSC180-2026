import argparse
import json
import os
from pathlib import Path

from agents.evaluator import Evaluator
from agents.learner_agent import LearnerAgent
from common.audit import AuditLog
from common.images import load_images
from config import Config
from coordinator.coordinator import Coordinator


def main():
    parser = argparse.ArgumentParser(description="Train a face detector with weighted validation boosting")
    parser.add_argument("--data", default=os.environ.get("BOOSTING_DATA_DIR", "."),
                        help="dataset root containing train/ and validate/")
    parser.add_argument("--library", default=os.environ.get("BOOSTING_LIBRARY", "learner/libweak.so"),
                        help="path to the compiled learner shared library")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, help="small smoke test only")
    parser.add_argument("--llm-adapter", help="path to a Python provider adapter")
    parser.add_argument("--no-augmentation", action="store_true", help="train only on original images")
    args = parser.parse_args()
    cfg = Config(T=args.rounds, K=args.candidates, max_refine=args.attempts,
                 time_limit=args.time_limit, seed=args.seed, llm_adapter=args.llm_adapter)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    X, y, paths, variants = load_images(Path(args.data).resolve(), args.library, args.limit_per_class,
                                        augment=not args.no_augmentation)
    (run / "dataset.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    learners = LearnerAgent(str(Path(args.library).resolve()), X, y, variants)
    coordinator = Coordinator(cfg, learners, Evaluator(learners._lib),
                              AuditLog(run / "audit.jsonl"), X, y, None, None,
                              str(Path(args.data).resolve()), str(run))
    print(json.dumps(coordinator.run()))


if __name__ == "__main__":
    main()
