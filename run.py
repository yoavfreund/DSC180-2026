"""Build script: one command per target.

    python run.py test     unit tests + a 2-round pipeline run (no LLM, under a minute)
    python run.py all      Haar baseline: train, cascade, score, evaluate (no LLM, same result every run)
    python run.py agents   the main agent comparison (needs a Claude sign-in; results vary run to run)
    python run.py clean    delete everything the targets above built
"""
import shutil
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent


def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def pipeline(name, rounds):
    """Train with the Haar search only, slice the cascade, score validate, evaluate."""
    run = Path("runs") / name
    sh(PY, "main.py", "--run-name", name, "--rounds", rounds, "--agent", "base=haar")
    sh(PY, "cascade.py", "--run-dir", run, "--face-loss", "0.01")
    sh(PY, "predict.py", "--run-dir", run, "--images", "validate",
       "--scores", run / "scores.txt", "--key", run / "key.txt", "--cascade")
    sh("make", "-C", "eval", "eval")
    sh("eval/eval", run / "scores.txt", run / "key.txt", "--fp-rates", "0.01,0.05,0.1")


def test():
    sh(PY, "-m", "pytest", "-q", "tests")
    sh("eval/tests/run_tests.sh")
    pipeline("test-smoke", 2)


def all_():
    pipeline("haar-baseline", 10)


def agents():
    sh(PY, "compare.py", "run", "--name", "main", "--rounds", 25, "--max-sessions", 100)


def clean():
    shutil.rmtree(ROOT / "runs", ignore_errors=True)
    (ROOT / "eval" / "eval").unlink(missing_ok=True)
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    print("removed runs/, eval/eval and __pycache__/")


TARGETS = {"test": test, "all": all_, "agents": agents, "clean": clean}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        sys.exit(__doc__)
    TARGETS[sys.argv[1]]()
