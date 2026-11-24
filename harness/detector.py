"""Build a native detector from frozen rules, with real per-patch early exits."""
import math
from pathlib import Path
import subprocess

import numpy as np

from ensemble import read_rule
from harness.build import CC, CFLAGS, COMPILE_TIMEOUT_S, compile_rule


def _literal(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("detector parameters must be finite")
    return value.hex()


def _scorer(name, entries, stages, rank_exits):
    offset = 2.0 * float(np.sum(np.abs([e["alpha"] for e in entries]))) + 2.0
    checks = {stage["after"]: (index, stage["theta"]) for index, stage in enumerate(stages)
              if stage["theta"] is not None}
    lines = [f"double {name}(const vj_image *im, int *used) {{", "    double total = 0.0;"]
    for count, entry in enumerate(entries, 1):
        alpha = _literal(entry["alpha"] * entry["polarity"])
        opposite = _literal(-entry["alpha"] * entry["polarity"])
        theta = _literal(entry["theta"])
        lines.append(f"    double raw_{count} = weak_score_{count}(im);")
        lines.append(f"    if (!isfinite(raw_{count})) return NAN;")
        lines.append(f"    total += raw_{count} > {theta} ? {alpha} : {opposite};")
        if count in checks:
            index, threshold = checks[count]
            exit_score = f"{_literal(index * offset)} + total" if rank_exits else _literal(-offset)
            lines.extend([f"    if (total < {_literal(threshold)}) {{",
                          f"        *used = {count};",
                          f"        return {exit_score};",
                          "    }"])
    final_score = f"{_literal(len(stages) * offset)} + total" if rank_exits else "total"
    lines.extend([f"    *used = {len(entries)};",
                  f"    return {final_score};", "}"])
    return "\n".join(lines)


def build_detector(run_dir, entries, stages, build_dir, *, rank_exits=False):
    """Compile learners separately and link full and early-exit modes together."""
    run_dir, build_dir = Path(run_dir), Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    runtime = Path(__file__).resolve().parent
    header = run_dir / "harness" / "vj_learner.h"
    if not header.is_file():
        header = runtime / "vj_learner.h"
    cuts = [s["after"] for s in stages]
    if cuts != sorted(set(cuts)) or any(not 1 <= cut < len(entries) for cut in cuts):
        raise ValueError("invalid checkpoints for this chain length")
    objects = []
    declarations = []
    for count, entry in enumerate(entries, 1):
        name = f"weak_score_{count}"
        obj, _ = compile_rule(read_rule(run_dir, entry), build_dir / name, header, symbol=name)
        objects.append(obj)
        declarations.append(f"double {name}(const vj_image *);")
    chain = build_dir / "chain.c"
    chain.write_text("\n\n".join(["\n".join(declarations), _scorer("score_full", entries, [], False),
                                   _scorer("score_cascade", entries, stages, rank_exits)]) + "\n")
    binary = build_dir / "detector"
    result = subprocess.run(
        [CC, *CFLAGS, "-include", str(header), str(chain),
         str(runtime / "detector.c"), *map(str, objects), "-lm", "-o", str(binary)],
        capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
    )
    if result.returncode:
        raise RuntimeError(f"detector build failed:\n{result.stderr}")
    return binary


def score_detector(binary, images_file, count, output_file, cascade=False):
    mode = "cascade" if cascade else "full"
    subprocess.run([binary, "score", images_file, str(count), mode, output_file],
                   check=True, timeout=60)
    data = np.fromfile(output_file, dtype=np.float64).reshape(count, 2)
    return data[:, 0], data[:, 1].astype(np.int32)


def time_detector(binary, images_file, count, reps=15):
    result = subprocess.run([binary, "time", images_file, str(count), "both", str(reps)],
                            check=True, capture_output=True, text=True, timeout=60)
    return np.fromstring(result.stdout, dtype=np.float64, sep=" ").reshape(reps, 2)
