"""Apply a saved weighted ensemble to the supplied train/validate image tree."""
import argparse
from pathlib import Path

from agents.combiner import combine, load_ensemble
from agents.learner_agent import LearnerAgent
from common.images import load_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--library", default="learner/libweak.so")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-class", type=int, help="small smoke test only")
    args = parser.parse_args()
    X, y, paths, _ = load_images(Path(args.data).resolve(), args.library, args.limit_per_class)
    learner = LearnerAgent(str(Path(args.library).resolve()), X, y)
    labels = combine(load_ensemble(args.run_dir), learner, X, "adaboost")
    with open(args.output, "x", encoding="utf-8") as stream:
        for path, label in zip(paths, labels):
            stream.write(f"{path}\t{int(label)}\n")


if __name__ == "__main__":
    main()
