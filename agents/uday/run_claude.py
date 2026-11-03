#!/usr/bin/env python3
"""run_claude.py - run the agent against a Claude model, entirely from Python.

Replaces run_local.py (the Ollama launcher). Edit the settings below, then run this
file from your IDE with the project's virtualenv (the one that has claude-agent-sdk).

Before the first run:
  1. Someone with a paid Anthropic API account (your sister) creates an API key in the
     Anthropic Console, ideally with a spend limit set on it or its workspace.
     Use a key for you to use. Don't share her login.
  2. Save the key as the only line of the file KEY_FILE (below), which is outside the
     repo, so it can never be committed to GitHub. (Or set ANTHROPIC_API_KEY.)
  3. `pipeline/stage.py prepare` must have been run once (the agent needs staged data).

What this script does: loads the key, makes one tiny test request to check the key and
model work (costs a fraction of a cent), then runs the isolation self-test
(MODE = "redteam") or one agent run per entry in VARIANTS_TO_RUN (MODE = "variants").
Each run stops when it reaches MAX_TURNS or MAX_BUDGET_USD. Results go to
agents/<agent>/runs/<run_id>/, and console.log there has a [usage] line with the cost.
"""
import datetime
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ----------------------------- settings: edit these -----------------------------
MODE = "variants"                  # "redteam": isolation self-test only. "variants": real runs.
MODEL = "claude-sonnet-5-5"       # e.g. "claude-haiku-5-5" (cheapest), "claude-sonnet-5-5", "claude-opus-5-5"
VARIANTS_TO_RUN = ["waldboost"]  # names from variants.py ("free" = none assigned)
MAX_TURNS = 40
MAX_VALIDATIONS = 5               # validation calls per run
MAX_BUDGET_USD = 2.0              # per run; the run stops when its cost reaches this
REDTEAM_TURNS = 20
REDTEAM_BUDGET_USD = 1.0
KEY_FILE = Path.home() / ".config" / "dsc180" / "anthropic_api_key"
# Only needed if the API says "This API key is not scoped to a workspace". Either have the key's owner create a
# key inside one workspace (then leave this empty), or paste that workspace's ID here (it starts with "wrkspc_").
WORKSPACE_ID = "wrkspc_01B5QGPKrobTtamddepQoBk5"
# --------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
API_URL = "https://api.anthropic.com/v1/messages"

KEY_HELP = f"""\
No API key found. To set one up:
  1. Your sister creates an API key in the Anthropic Console (console.anthropic.com, API keys),
     and sets a spend limit if the Console offers one.
  2. Create the file {KEY_FILE}
     containing only the key (it starts with "sk-ant-"). Make the folder if needed.
  3. Run this again.
Alternatively set the ANTHROPIC_API_KEY environment variable."""


def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    source = "the ANTHROPIC_API_KEY environment variable"
    if not key and KEY_FILE.exists():
        resolved = KEY_FILE.resolve()
        if REPO in resolved.parents:
            sys.exit(f"{KEY_FILE} is inside the repo, where it could be committed. Move it outside the repo.")
        key = KEY_FILE.read_text().strip()
        source = str(KEY_FILE)
        try:
            os.chmod(KEY_FILE, 0o600)  # owner-only; harmless if it fails (e.g. on Windows)
        except OSError:
            pass
    if not key:
        sys.exit(KEY_HELP)
    if not key.startswith("sk-ant-"):
        sys.exit(f"The key from {source} does not look like an Anthropic API key (it should start with sk-ant-).")
    print(f"Using the API key from {source}.")
    return key



def check_key_and_model(key):
    """Make a tiny request to verify the API key, workspace, and model."""
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    if WORKSPACE_ID:
        headers["anthropic-workspace-id"] = WORKSPACE_ID

    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": MODEL,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "Say OK."}],
        }).encode(),
        headers=headers,
    )

    try:
        urllib.request.urlopen(req, timeout=60).read()

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")

        if "workspace" in body.lower():
            hint = (
                "WORKSPACE ERROR: Anthropic cannot find this workspace "
                "or this API key does not have access to it. Ask the key "
                "owner to verify the workspace and key permissions."
            )
        elif e.code == 401:
            hint = "API key rejected. Check whether it is correct or revoked."
        elif e.code == 403:
            hint = "API key lacks permission to use this model or feature."
        elif e.code == 404:
            hint = f"Model '{MODEL}' not found or unavailable to this key."
        elif e.code == 429:
            hint = "Rate limit or spending limit reached."
        elif "credit" in body.lower():
            hint = "API account has insufficient credit."
        else:
            hint = "Unexpected API error."

        sys.exit(f"Test request failed (HTTP {e.code}). {hint}\n{body}")

    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"Could not reach Anthropic API: {e}")

    print(f"Key, workspace, and model OK ({MODEL}).")


def clean_env(key):
    """Use the real Anthropic API: remove any redirect left over from the Ollama experiments."""
    for var in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_CUSTOM_HEADERS"):
        os.environ.pop(var, None)
    os.environ["ANTHROPIC_API_KEY"] = key
    if WORKSPACE_ID:  # Claude Code adds this header to every request it makes
        os.environ["ANTHROPIC_CUSTOM_HEADERS"] = f"anthropic-workspace-id: {WORKSPACE_ID}"


def run_agent(argv):
    """Run agent.main() with the given command-line arguments. Returns 0 on success."""
    import agent  # imported late so the environment is already set
    sys.argv = ["agent.py"] + argv
    try:
        agent.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def main():
    stamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    results = []
    if WORKSPACE_ID and not WORKSPACE_ID.startswith("wrkspc_"):
        sys.exit('WORKSPACE_ID should start with "wrkspc_" (copy it from the Console).')
    key = load_api_key()
    check_key_and_model(key)
    clean_env(key)
    sys.path.insert(0, str(HERE))
    from variants import VARIANTS
    try:
        if MODE == "redteam":
            run_id = f"redteam-{stamp}"
            code = run_agent([run_id, "--model", MODEL, "--redteam", "--max-turns", str(REDTEAM_TURNS),
                              "--max-budget-usd", str(REDTEAM_BUDGET_USD)])
            results.append((run_id, code))
        elif MODE == "variants":
            unknown = [v for v in VARIANTS_TO_RUN if v not in VARIANTS]
            if unknown:
                sys.exit(f"Unknown variants {unknown}; choose from {sorted(VARIANTS)}")
            for variant in VARIANTS_TO_RUN:
                run_id = f"{variant}-{stamp}"
                print(f"\n===== run {run_id} =====")
                code = run_agent([run_id, "--model", MODEL, "--variant", variant, "--max-turns", str(MAX_TURNS),
                                  "--max-validations", str(MAX_VALIDATIONS), "--max-budget-usd", str(MAX_BUDGET_USD)])
                results.append((run_id, code))
        else:
            sys.exit('MODE must be "redteam" or "variants"')
    except KeyboardInterrupt:
        print("\nInterrupted.")
    print("\nSummary")
    for run_id, code in results:
        print(f"  {run_id}: {'ok' if code == 0 else f'exit {code}'}  -> {HERE / 'runs' / run_id}")
    print("Cost and turns for each run are in its console.log, on the [usage] and [done] lines.")


if __name__ == "__main__":
    main()