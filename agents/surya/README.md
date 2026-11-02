# Surya's learner

`agent.py` follows the launcher lifecycle: prepare training data and an
empty learner, run an agent with a validation tool, and save logs. It uses
native Linux Codex inside Ubuntu WSL.

Your implementation lives directly in `agents/surya/runs/<run-id>/learner/`.
Each run also holds its training copy, instructions, configuration, and logs.
The trusted validation data and build snapshots stay in Ubuntu's private root.

From PowerShell at the repository root:

```powershell
$repoLinux = (wsl -d Ubuntu -- wslpath -u (Get-Location).Path).Trim()
# One-time native runtime installation, then sign in to the isolated client.
wsl -d Ubuntu -- python3 "$repoLinux/agents/surya/install_runtime.py"
wsl -d Ubuntu -- python3 "$repoLinux/agents/surya/agent.py" --login
# Prepare only; no model session or validation call.
wsl -d Ubuntu -- python3 "$repoLinux/agents/surya/agent.py" run-01 --prepare-only
# Start Codex with the already prepared run.
wsl -d Ubuntu -- python3 "$repoLinux/agents/surya/agent.py" run-01 --reuse
# Validate an existing learner once.
wsl -d Ubuntu -- python3 "$repoLinux/agents/surya/agent.py" run-01 --validate
```

Use `--check` to probe real agent access restrictions and inspect the validation
MCP configuration without a model call. Every model launch repeats the access
checks and refuses to start if they fail. The native runtime is installed in
Ubuntu's private root; an existing Windows sign-in is not copied into it.
Use `--model <id>` to select a model and `--max-validations <n>` to set a new
run's budget. Reusing a run keeps the same budget and existing source. A run
that used a nondefault budget must pass that same value when reused.
Each run also freezes its own `system_prompt_runNN.md` and records its SHA-256.
Reusing a run preserves that prompt even when the default system_prompt.md
changes; editing a frozen prompt requires a new run ID.

Codex JSON events are saved to `console.log`, item events to `tool_log.jsonl`,
and the final response to `final.md`. Validation logs are copied to `runs.jsonl`
after the agent exits. Logs are append-only across launcher invocations;
`final.md` contains the latest response. There is no Claude-style turn limit.

The editing agent now has a deny-by-default Linux filesystem profile and no
tool network, browsing, apps, plugins, or subagents. Training data is read-only;
only learner/ is writable. See [SECURITY.md](SECURITY.md) for the boundary and
the passing live-test results. Opening a direct desktop chat
instead of using this launcher does not apply these controls.
