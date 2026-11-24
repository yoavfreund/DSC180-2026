"""Check, compile, run and time one agent-written weak rule.

An agent submits C source that defines `double weak_score(const vj_image *im)`.
This module is the only path from that source to a number:

1. check_source: size limit; no #include, #pragma, asm, __attribute__ or _Pragma.
2. compile with vj_learner.h forced in (-include), then read the object's
   undefined symbols with nm. Only math functions (and the few symbols compilers
   emit on their own) may be called. This blocks file, network and process calls
   even if the source declares them by hand.
3. link with runner.c and run it in a child process with a timeout. A crash, a
   hang or a non-finite score fails the candidate; it never stops the caller.

The checks keep honest mistakes and accidental leaks out. They are not a full
sandbox against code written to attack the machine.
"""
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import platform

import numpy as np

# A run builds from its own snapshot of the harness (FACE_HARNESS_DIR), so editing the
# repository while a run is going cannot change that run. See use_harness().
HARNESS = Path(os.environ.get("FACE_HARNESS_DIR") or Path(__file__).resolve().parent)
HEADER = HARNESS / "vj_learner.h"
SCAN_HEADER = HARNESS / "vj_scan.h"
MAX_SCAN = 1000         # parameter combinations per scan
RUNNER_C = HARNESS / "runner.c"
HARNESS_FILES = ("vj_learner.h", "vj_scan.h", "vj_image.h", "runner.c")


def use_harness(directory):
    """Build from an existing harness snapshot (for example a finished run's harness/ folder)."""
    global HARNESS, HEADER, SCAN_HEADER, RUNNER_C
    d = Path(directory)
    HARNESS, HEADER, SCAN_HEADER, RUNNER_C = d, d / "vj_learner.h", d / "vj_scan.h", d / "runner.c"


def snapshot_harness(dest):
    """Copy the harness sources to dest and build from there in this process. Returns dest."""
    global HARNESS, HEADER, SCAN_HEADER, RUNNER_C
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    for name in HARNESS_FILES:
        (dest / name).write_bytes((src / name).read_bytes())
    HARNESS, HEADER, SCAN_HEADER, RUNNER_C = dest, dest / "vj_learner.h", dest / "vj_scan.h", dest / "runner.c"
    return dest
CC = os.environ.get("CC", "cc")
CFLAGS = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Wno-unused-parameter",
          "-Werror=implicit-function-declaration", "-Werror=return-type"]
MAX_SOURCE_BYTES = 8000
COMPILE_TIMEOUT_S = 30
MAX_MESSAGE_CHARS = 1500
# Timed alongside every candidate. Its time (the call and loop overhead, about 3.5 ns)
# is subtracted, so a rule's ns is the extra time it costs over an empty rule.
BASELINE_SOURCE = "/* rule: empty (timing baseline) */\ndouble weak_score(const vj_image *im) { return 0.0; }\n"

BANNED = [
    (re.compile(r"^\s*#\s*(include|import|pragma|line)\b", re.M), "#include, #import, #pragma and #line are not allowed"),
    (re.compile(r"\b(asm|__asm|__asm__)\b"), "inline assembly is not allowed"),
    (re.compile(r"\b__attribute__\b"), "__attribute__ is not allowed"),
    (re.compile(r"\b_Pragma\b"), "_Pragma is not allowed"),
]

MATH = {
    "sqrt", "cbrt", "fabs", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "pow",
    "floor", "ceil", "round", "lround", "trunc", "fmod", "fmin", "fmax", "hypot",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "erf",
}
ALLOWED_UNDEFINED = (MATH | {m + "f" for m in MATH}
                     | {"memset", "memcpy", "memmove", "bzero", "__bzero",
                        "__chkstk_darwin", "__stack_chk_fail", "__stack_chk_guard"})


class LearnerError(Exception):
    """The candidate cannot be used. The message is safe to show the agent."""


@dataclass(frozen=True)
class Build:
    binary: Path
    sha256: str
    warnings: str


def _clip(text, limit=MAX_MESSAGE_CHARS):
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"\n...[{len(text) - limit} more characters]"


def check_source(source):
    """Return a list of problems with the source; an empty list means it may be compiled."""
    problems = []
    size = len(source.encode("utf-8"))
    if size > MAX_SOURCE_BYTES:
        problems.append(f"source is {size} bytes; the limit is {MAX_SOURCE_BYTES}")
    for pattern, message in BANNED:
        if pattern.search(source):
            problems.append(message)
    if "weak_score" not in source:
        problems.append("source must define double weak_score(const vj_image *im)")
    return problems


def _nm(obj, flag):
    out = subprocess.run(["nm", flag, str(obj)], capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    if out.returncode != 0:
        raise LearnerError("nm failed: " + _clip(out.stderr))
    return [line.split() for line in out.stdout.splitlines() if line.split()]


def _c_name(name):
    if sys.platform == "darwin" and name.startswith("_"):
        return name[1:]  # Mach-O adds one leading underscore to C names
    return name


def _undefined_symbols(obj):
    return {_c_name(parts[-1]) for parts in _nm(obj, "-u")}


def _defined_symbols(obj):
    """Global names the object defines (functions and variables), without the undefined ones."""
    return {_c_name(parts[-1]) for parts in _nm(obj, "-g") if len(parts) >= 3 and parts[-2] not in "Uw"}


def runner_object(cache_dir):
    """Compile runner.c once per cache directory (atomic, safe for parallel workers)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_header = RUNNER_C.with_name("vj_image.h")
    image_code = image_header.read_bytes() if image_header.exists() else b""
    compiler = f"{platform.system()}:{platform.machine()}:{CC}"
    key = hashlib.sha256(RUNNER_C.read_bytes() + HEADER.read_bytes() + SCAN_HEADER.read_bytes() + image_code
                         + compiler.encode()).hexdigest()[:16]
    obj = cache_dir / f"runner-{key}.o"
    if obj.exists():
        return obj
    fd, tmp = tempfile.mkstemp(suffix=".o", dir=cache_dir)
    os.close(fd)
    out = subprocess.run([CC, *CFLAGS, "-c", str(RUNNER_C), "-o", tmp],
                         capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    if out.returncode != 0:
        os.unlink(tmp)
        raise RuntimeError("runner.c failed to compile:\n" + out.stderr)
    os.replace(tmp, obj)
    return obj


def build_learner(source, build_dir, cache_dir):
    """Check, compile and link one weak rule. Raises LearnerError with an agent-readable reason."""
    return _build(source, build_dir, cache_dir, template=False)


def build_template(source, build_dir, cache_dir):
    """Like build_learner, but P0..P3 (vj_scan.h) are allowed. Only for scan; never submitted."""
    if not re.search(r"\bP[0-3]\b", source):
        raise LearnerError("a scan template must use at least one of P0, P1, P2, P3")
    return _build(source, build_dir, cache_dir, template=True)


def _build(source, build_dir, cache_dir, template):
    obj, warnings = compile_rule(source, build_dir, HEADER, SCAN_HEADER if template else None)
    binary = obj.with_name("learner")
    out = subprocess.run([CC, str(runner_object(cache_dir)), str(obj), "-lm", "-o", str(binary)],
                         capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    if out.returncode != 0:
        raise LearnerError("link error (is weak_score defined exactly once, and no main?):\n" + _clip(out.stderr))
    return Build(binary, hashlib.sha256(source.encode("utf-8")).hexdigest(), warnings)


def compile_rule(source, build_dir, header, scan_header=None, symbol="weak_score", rename=None):
    """Compile and check an accepted learner, optionally using a unique score symbol.

    With a unique symbol (several rules linked into one detector), any other global name the
    rule defines, such as a non-static helper function, gets the suffix __<symbol> so two rules
    with a helper of the same name still link."""
    problems = check_source(source)
    if problems:
        raise LearnerError("source rejected: " + "; ".join(problems))
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / "learner.c"
    src.write_text(source, encoding="utf-8")
    obj = build_dir / "learner.o"
    includes = ["-include", str(header)]
    if symbol != "weak_score":
        includes += [f"-Dweak_score={symbol}"]
    includes += [f"-D{name}={name}__{symbol}" for name in sorted(rename or ())]
    if scan_header is not None:
        includes += ["-include", str(scan_header)]
    try:
        out = subprocess.run([CC, *CFLAGS, *includes, "-c", str(src), "-o", str(obj)],
                             capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise LearnerError(f"compiling took longer than {COMPILE_TIMEOUT_S} s") from None
    if out.returncode != 0:
        msg = out.stderr.replace(str(build_dir) + "/", "")
        if scan_header is None and re.search(r"\bP[0-3]\b", source):
            msg = "P0..P3 exist only in scan templates; submit concrete numbers.\n" + msg
        raise LearnerError("compile error:\n" + _clip(msg))
    warnings = _clip(out.stderr.replace(str(build_dir) + "/", ""))
    allowed = ALLOWED_UNDEFINED | ({"vj_p"} if scan_header is not None else set())
    bad = sorted(_undefined_symbols(obj) - allowed)
    if bad:
        raise LearnerError("calls functions that are not allowed: " + ", ".join(bad)
                           + ". Only <math.h> functions may be called.")
    if symbol != "weak_score" and rename is None:
        extra = _defined_symbols(obj) - {symbol}
        if extra:
            return compile_rule(source, build_dir, header, scan_header, symbol, rename=extra)
    return obj, warnings


def expand_ranges(ranges):
    """{"P0": [lo, hi] or [lo, hi, step], ...} -> list of (p0, p1, p2, p3). Unnamed parameters are 0."""
    if not isinstance(ranges, dict) or not ranges:
        raise LearnerError('ranges must look like {"P0": [0, 15], "P1": [2, 8, 2]}')
    axes = []
    for k in ("P0", "P1", "P2", "P3"):
        r = ranges.get(k)
        if r is None:
            axes.append([0])
            continue
        if not isinstance(r, (list, tuple)) or len(r) not in (2, 3) or not all(isinstance(v, int) for v in r):
            raise LearnerError(f"{k} must be [lo, hi] or [lo, hi, step] with integers")
        lo, hi, step = r[0], r[1], (r[2] if len(r) == 3 else 1)
        if step < 1 or hi < lo or hi - lo > 40:
            raise LearnerError(f"{k}: need lo <= hi, hi - lo <= 40 and step >= 1")
        axes.append(list(range(lo, hi + 1, step)))
    unknown = set(ranges) - {"P0", "P1", "P2", "P3"}
    if unknown:
        raise LearnerError("unknown parameters: " + ", ".join(sorted(unknown)))
    total = 1
    for a in axes:
        total *= len(a)
    if total > MAX_SCAN:
        raise LearnerError(f"{total} combinations; the limit is {MAX_SCAN}. Use a step or a smaller range.")
    return [(a, b, c, d) for a in axes[0] for b in axes[1] for c in axes[2] for d in axes[3]]


def run_scan(binary, images_path, n, combos, timeout_s=30):
    """Scores (len(combos), n) for a template binary."""
    params = Path(binary).parent / "params.txt"
    params.write_text("".join(f"{a} {b} {c} {d}\n" for a, b, c, d in combos))
    try:
        proc = subprocess.run([str(binary), "scan", str(images_path), str(n), str(params)],
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise LearnerError(f"the scan took longer than {timeout_s} s; use fewer combinations") from None
    if proc.returncode != 0:
        raise LearnerError(f"the template crashed during the scan (exit code {proc.returncode})")
    out = Path(str(params) + ".out")
    scores = np.fromfile(out, dtype=np.float64)
    out.unlink()
    if scores.size != len(combos) * n:
        raise LearnerError("the scan wrote the wrong number of scores")
    return scores.reshape(len(combos), n)


def substitute(source, values):
    """Replace P0..P3 with concrete integers, giving a rule that can be submitted."""
    return re.sub(r"\bP([0-3])\b", lambda m: str(values[int(m.group(1))]), source)


def write_images(images, path):
    """Write uint8 images (n, 19, 19) in the runner's format."""
    images = np.ascontiguousarray(images, dtype=np.uint8)
    if images.ndim != 3 or images.shape[1:] != (19, 19):
        raise ValueError("images must have shape (n, 19, 19)")
    Path(path).write_bytes(images.tobytes())
    return len(images)


def run_scores(binary, images_path, n, timeout_s=20):
    """Return weak_score for each of the n images. Raises LearnerError on crash, hang or non-finite output."""
    fd, out_path = tempfile.mkstemp(suffix=".scores", dir=Path(binary).parent)
    os.close(fd)
    try:
        proc = subprocess.run([str(binary), "score", str(images_path), str(n), out_path],
                              capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            raise LearnerError(f"weak_score crashed (exit code {proc.returncode}). " + _clip(proc.stderr, 300))
        scores = np.fromfile(out_path, dtype=np.float64)
    except subprocess.TimeoutExpired:
        raise LearnerError(f"scoring {n} images took longer than {timeout_s} s") from None
    finally:
        os.unlink(out_path)
    if scores.shape != (n,):
        raise LearnerError("runner wrote the wrong number of scores")
    bad = int(np.count_nonzero(~np.isfinite(scores)))
    if bad:
        raise LearnerError(f"weak_score returned NaN or infinity for {bad} of {n} images")
    return scores


def time_ns(binary, images_path, n, reps=7, timeout_s=20):
    """Median nanoseconds per weak_score call over n images, repeated reps times."""
    try:
        proc = subprocess.run([str(binary), "time", str(images_path), str(n), str(reps)],
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise LearnerError(f"timing took longer than {timeout_s} s") from None
    if proc.returncode != 0:
        raise LearnerError(f"weak_score crashed while timed (exit code {proc.returncode})")
    return float(proc.stdout.strip())


def baseline_binary(cache_dir):
    """The empty rule, built once per cache folder (atomic, safe for parallel workers)."""
    final = Path(cache_dir) / "baseline-learner"
    if not final.exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="baseline-", dir=cache_dir))
        built = build_learner(BASELINE_SOURCE, tmp_dir, cache_dir).binary
        os.replace(built, final)
    return final


def marginal_ns(binaries, images_path, n, baseline, rounds=3, reps=7):
    """Extra ns per call over the empty baseline, for each binary.

    The baseline and the candidates are timed in turn, `rounds` times, and the
    minimum is kept for each, so a burst of load on the machine hits all of them
    alike and the fastest (least disturbed) run counts. Use a small image set
    (a few hundred images) so it stays in cache; otherwise memory traffic, not
    the rule, decides the time. Returns [(marginal_ns, raw_ns), ...].
    """
    raw = [float("inf")] * len(binaries)
    base = float("inf")
    for _ in range(rounds):
        base = min(base, time_ns(baseline, images_path, n, reps))
        for i, b in enumerate(binaries):
            raw[i] = min(raw[i], time_ns(b, images_path, n, reps))
    return [(max(r - base, 0.0), r) for r in raw]
