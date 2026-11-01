#!/usr/bin/env bash
# Tests for eval: hand-checked mock classifiers, malformed inputs, and a
# random cross-check against a brute-force reference. Run from anywhere.
set -u
cd "$(dirname "$0")/.."
make -s eval || exit 1
EVAL=./eval
fail=0

# 1. Mock classifiers with hand-computed expected output.
for d in tests/cases/*/; do
    name=$(basename "$d")
    # shellcheck disable=SC2046
    got=$($EVAL "$d/scores.txt" "$d/key.txt" $(cat "$d/args") 2>&1)
    if [ "$got" == "$(cat "$d/expected.txt")" ]; then
        echo "PASS  mock/$name"
    else
        echo "FAIL  mock/$name"
        diff <(echo "$got") "$d/expected.txt" | sed 's/^/      /'
        fail=1
    fi
done

# 2. Malformed inputs must exit 1 with a specific message and print no metrics.
for d in tests/bad/*/; do
    name=$(basename "$d")
    out=$($EVAL "$d/scores.txt" "$d/key.txt" --fp-counts 1 2>/dev/null)
    err=$($EVAL "$d/scores.txt" "$d/key.txt" --fp-counts 1 2>&1 >/dev/null)
    code=$?
    if [ $code -eq 1 ] && [ -z "$out" ] && grep -qF "$(cat "$d/expect_err")" <<<"$err"; then
        echo "PASS  reject/$name"
    else
        echo "FAIL  reject/$name (exit $code, stderr: $err, stdout: $out)"
        fail=1
    fi
done

# 3. Bad usage exits 2.
$EVAL tests/cases/perfect/scores.txt tests/cases/perfect/key.txt --fp 1 >/dev/null 2>&1
if [ $? -eq 2 ]; then echo "PASS  usage"; else echo "FAIL  usage"; fail=1; fi

# 4. Random inputs (with heavy ties) against the brute-force reference.
if python3 tests/reference.py crosscheck "$EVAL" --trials 500; then
    echo "PASS  crosscheck"
else
    echo "FAIL  crosscheck"
    fail=1
fi

[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail
