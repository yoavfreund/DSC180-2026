"""Fake agents for tests. They behave like real workers but never call an LLM.

modes:
  good          submit a fixed Haar rule at once and exit
  hang          never submit; sleep until killed
  submit_hang   submit a fixed rule, then sleep until killed (the submission still counts)
  forbidden     submit a rule that calls fopen (the coordinator must reject it)
  crash         exit with an error and no submission
"""
import argparse
import sys
import time
from pathlib import Path

from agents.common import write_result, write_submission

GOOD = ("/* rule: eye band darker than cheeks */\n"
        "double weak_score(const vj_image *im) { return vj_haar2_tb(im, 2, 5, 15, 3); }\n")
FORBIDDEN = ("void *fopen(const char *, const char *);\n"
             "double weak_score(const vj_image *im) { return fopen(\"/etc/hosts\", \"r\") != 0; }\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == "good":
        write_submission(out, GOOD)
        write_result(out, status="finished")
    elif args.mode in ("hang", "submit_hang"):
        if args.mode == "submit_hang":
            write_submission(out, GOOD)
        time.sleep(3600)
    elif args.mode == "forbidden":
        write_submission(out, FORBIDDEN)
        write_result(out, status="finished")
    elif args.mode == "crash":
        sys.exit(3)
    else:
        sys.exit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()
