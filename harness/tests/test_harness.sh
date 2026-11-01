#!/usr/bin/env bash
# End-to-end harness tests against the real staged data:
#   1. an honest baseline learner gets real validation metrics back
#   2. a red-team learner's attempts to read the key, the repo, its own run
#      log, or the network (from make and from predict) are all blocked
#   3. a learner/ containing a symlink is rejected before anything runs
#   4. the call budget is enforced
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PRIVATE=${DSC180_PRIVATE:-$HOME/dsc180-private}
WORK=${DSC180_WORK:-$HOME/dsc180-work}/_harness_tests
PY=${PYTHON:-python3}
fail=0
pass() { echo "PASS  $1"; }
flunk() { echo "FAIL  $1"; fail=1; }

[ -d "$PRIVATE/staged/validate" ] || "$PY" "$REPO/pipeline/stage.py" prepare || exit 1
rm -rf "$WORK" "$PRIVATE"/runs/_test_*
mkdir -p "$WORK"

# 1. Baseline
mkdir -p "$WORK/baseline" && cp -R "$HERE/baseline_learner" "$WORK/baseline/learner"
out=$("$PY" "$REPO/harness/run_validation.py" "$WORK/baseline" --run-name _test_baseline)
echo "$out" | sed 's/^/      /'
if grep -q "AUC" <<<"$out" && grep -q "364 faces, 682 non-faces" <<<"$out"; then
    pass "baseline learner returns validation metrics"
else
    flunk "baseline learner returns validation metrics"
fi
if grep -qE '"status": "ok"' "$PRIVATE/runs/_test_baseline/runs.jsonl"; then
    pass "run logged privately"
else
    flunk "run logged privately"
fi
if grep -qE '^timing: train [0-9.]+ s wall .*predict [0-9.]+ us CPU per image' <<<"$out" \
    && grep -q '"cpu_us_per_image"' "$PRIVATE/runs/_test_baseline/runs.jsonl"; then
    pass "train and predict timings reported and logged"
else
    flunk "train and predict timings reported and logged"
fi

# 1b. Cascade requirement: a 1-stage learner is rejected, a 2-stage one passes.
out=$("$PY" "$REPO/harness/run_validation.py" "$WORK/baseline" --run-name _test_onestage --require-cascade)
if grep -q "cascade required" <<<"$out" && ! grep -q "AUC" <<<"$out"; then
    pass "single-stage learner rejected when a cascade is required"
else
    flunk "single-stage learner rejected when a cascade is required"
fi
mkdir -p "$WORK/twostage" && cp -R "$HERE/baseline_learner" "$WORK/twostage/learner"
sed -i '' 's/^CFLAGS ?= \(.*\)$/CFLAGS ?= \1 -DTWO_STAGE/' "$WORK/twostage/learner/Makefile"
out=$("$PY" "$REPO/harness/run_validation.py" "$WORK/twostage" --run-name _test_twostage --require-cascade)
if grep -q "cascade: 2 stages, weak classifiers per stage: 1,1" <<<"$out" && grep -q "AUC" <<<"$out" \
    && grep -qE "weak classifiers evaluated per image: non-faces [0-9.]+, faces [0-9.]+ \(of 2\)" <<<"$out"; then
    pass "two-stage learner accepted and its layout reported"
else
    flunk "two-stage learner accepted and its layout reported"
fi

# 2. Red team
mkdir -p "$WORK/redteam" && cp -R "$HERE/redteam_learner" "$WORK/redteam/learner"
printf 'PRIVATE_DIR=%s\nREPO_DIR=%s\nHOME_DIR=%s\n' "$PRIVATE" "$REPO" "$HOME" > "$WORK/redteam/learner/paths.mk"
echo probe > /Users/Shared/.dsc180_probe  # readable by anyone outside the sandbox
"$PY" "$REPO/harness/run_validation.py" "$WORK/redteam" --run-name _test_redteam >/dev/null
rm -f /Users/Shared/.dsc180_probe
build="$PRIVATE/runs/_test_redteam/build_001"
for f in probes_make.txt probes.txt; do
    if [ ! -s "$build/$f" ]; then flunk "red team $f was written"; continue; fi
    while read -r verdict what; do
        if [ "$verdict" = BLOCKED ]; then pass "blocked: $what"; else flunk "LEAKED: $what"; fi
    done < "$build/$f"
done

# 3. Symlink rejection
mkdir -p "$WORK/symlink" && cp -R "$HERE/baseline_learner" "$WORK/symlink/learner"
ln -s "$PRIVATE/staged/validate/labels.txt" "$WORK/symlink/learner/key.txt"
out=$("$PY" "$REPO/harness/run_validation.py" "$WORK/symlink" --run-name _test_symlink)
if grep -q "symlinks are not allowed" <<<"$out"; then pass "symlink in learner/ rejected"; else flunk "symlink in learner/ rejected"; fi

# 4. Call budget
out=$(cd "$REPO/harness" && "$PY" -c "
from run_validation import Validator
v = Validator('$WORK/baseline', '_test_budget', max_calls=1)
v.run()
print(v.run())")
if grep -q "budget used up" <<<"$out"; then pass "call budget enforced"; else flunk "call budget enforced"; fi

rm -rf "$WORK" "$PRIVATE"/runs/_test_*
[ $fail -eq 0 ] && echo "ALL HARNESS TESTS PASSED" || echo "SOME HARNESS TESTS FAILED"
exit $fail
