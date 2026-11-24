"""Freeze and compare rule-chain prefixes and their soft cascades on labeled patches."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from cascade import fit_cascade
from common.data import load_labeled_folder
from common.metrics import FP_RATES
from ensemble import apply_cascade
from harness.build import CC, write_images
from harness.detector import build_detector, score_detector, time_detector
from predict import write_predictions


def evaluate(binary, images_file, labels, ids, evaluator, out, cascade):
    mode = "cascade" if cascade else "full"
    scores, used = score_detector(binary, images_file, len(ids), out / "predictions.bin", cascade)
    scores_path, key_path = out / f"{mode}-scores.txt", out / f"{mode}-key.txt"
    write_predictions(scores_path, key_path, scores, labels, ids)
    text = subprocess.check_output(
        [evaluator, scores_path, key_path, "--fp-rates", ",".join(map(str, FP_RATES))], text=True,
    )
    (out / f"{mode}-eval.txt").write_text(text)
    records = [dict(token.split("=", 1) for token in line.split()) for line in text.splitlines()]
    return records, scores, used


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--model", action="append", required=True, metavar="RUN_DIR:RULES")
    args.add_argument("--validation", type=Path, default=Path("validate"))
    args.add_argument("--test", type=Path, required=True)
    args.add_argument("--out", type=Path, required=True)
    args.add_argument("--face-loss", type=float, default=0.01)
    cfg = args.parse_args()

    specs = []
    for model in cfg.model:
        try:
            path, rules = model.rsplit(":", 1)
            specs.append((Path(path), int(rules)))
        except ValueError:
            args.error(f"expected RUN_DIR:RULES, got {model!r}")
    cfg.out.mkdir(parents=True, exist_ok=False)
    validation, validation_y, _ = load_labeled_folder(cfg.validation)
    catalog = []
    with TemporaryDirectory(prefix="dsc180-compare-") as scratch:
        scratch = Path(scratch)
        validation_file = scratch / "validation.bin"
        write_images(validation, validation_file)
        evaluator = scratch / "eval"
        source = Path(__file__).resolve().parent / "eval" / "eval.c"
        subprocess.run([CC, "-std=c99", "-O2", str(source), "-lm", "-o", str(evaluator)],
                       check=True, timeout=30)
        for run, count in specs:
            out = cfg.out / f"{run.name}-{count}"
            out.mkdir()
            config, entries, trajectories = fit_cascade(run, validation, validation_y, cfg.face_loss, count)
            (out / "cascade.json").write_text(json.dumps(config, indent=2) + "\n")
            binary = build_detector(run, entries, config["stages"], scratch / out.name)
            alphas = [e["alpha"] for e in entries]
            reference, reference_used = apply_cascade(trajectories, config["stages"], alphas, rank_exits=False)
            scratch_output = scratch / "validation-scores.bin"
            full, used_full = score_detector(binary, validation_file, len(validation), scratch_output)
            soft, used = score_detector(binary, validation_file, len(validation), scratch_output, cascade=True)
            np.testing.assert_array_equal(full, trajectories[-1])
            np.testing.assert_array_equal(used_full, count)
            np.testing.assert_array_equal(soft, reference)
            np.testing.assert_array_equal(used, reference_used)
            face_exits = int(np.count_nonzero((validation_y > 0) & (used < count)))
            assert face_exits <= int(cfg.face_loss * int(np.count_nonzero(validation_y > 0)))
            print(f"Frozen {out.name}: cuts {[stage['after'] for stage in config['stages']]}, "
                  f"{face_exits} validation faces exited", flush=True)
            catalog.append((run, count, out, config, binary, face_exits))

        # Load the test set only after all stages and thresholds are frozen.
        test, labels, ids = load_labeled_folder(cfg.test)
        test_file, timing_file = scratch / "test.bin", scratch / "timing.bin"
        write_images(test, test_file)
        permutation = np.random.default_rng(0).permutation(len(test))
        write_images(test[permutation], timing_file)

        results = []
        for run, count, out, config, binary, face_exits in catalog:
            timing = time_detector(binary, timing_file, len(test))
            item = {
                "run": run.name,
                "rules": count,
                "stages": config["stages"],
                "validation_face_exits": face_exits,
                "modes": {},
            }
            for index, mode in enumerate(("full", "cascade")):
                records, scores, used = evaluate(binary, test_file, labels, ids, evaluator, out, mode == "cascade")
                early = used < count
                if early.any() and any(float(row["threshold"]) <= float(scores[early].max()) for row in records[1:]):
                    raise ValueError("a reported operating point admitted an early-rejected patch")
                item["modes"][mode] = {
                    "summary": records[0],
                    "operating_points": records[1:],
                    "mean_rules": float(used.mean()),
                    "face_exits": int(np.count_nonzero(early & (labels > 0))),
                    "nonface_exits": int(np.count_nonzero(early & (labels < 0))),
                    "ns_per_patch": float(np.median(timing[:, index])),
                    "timing_samples": timing[:, index].tolist(),
                }
            results.append(item)
            print(f"Evaluated {out.name}", flush=True)
        (cfg.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
