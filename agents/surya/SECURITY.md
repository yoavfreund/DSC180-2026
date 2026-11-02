# Agent access boundary

Launch through `agent.py` or its generated Windows shortcuts. Opening a normal
desktop chat on the run folder does not install these restrictions on that chat.

The launcher uses the official native Linux Codex binary in WSL, not Windows
Codex. This prevents Windows-to-WSL commands from crossing a Windows-only file
boundary. All generated shortcuts route back through the protected launcher.

The trusted launcher supplies a named Codex permission profile as command-line
overrides, uses an isolated client home, ignores user configuration, skips host
skill discovery, and marks project configuration untrusted. It sends the run's
instructions explicitly instead of reading instructions from ancestor folders.
It deliberately does not pass the legacy `--sandbox` flag, which would replace
the named permission profile.

The profile denies filesystem access by default. Agent tools can use the Linux
runtime files they need, read staged training data and run instructions, and
write their own learner directory. The model's working directory is learner/;
the rest of the run and repository are outside its writable workspace. The
immutable Codex executable is readable because its Linux sandbox must re-exec
it during startup; a nested process still inherits the outer restrictions.

Tool network access is disabled. Web search, browser/computer tools, image
generation, apps, plugins, memories, host skill discovery,
and subagents are disabled. Codex's bundled code-mode host remains enabled
because it dispatches tool calls; spawned commands retain the permission profile.
The launcher does not forward API-key environment
variables. Sign-in uses the isolated native client's protected auth store.
The client's own authenticated model connection remains available.

Only the fixed validation MCP server is configured, exposing only
`run_validation`. It runs as trusted host code outside the agent command policy,
retains access to the private key, and uses the existing sandboxed learner
harness. It accepts no tool arguments. Its budget persists across reconnects.
Only that authorized validation tool is configured for automatic approval;
commands still cannot request escalation outside the agent sandbox.

Before each model launch, a no-model probe runs through `codex sandbox` with
the same permission profile. It checks outside-file reads, a symlink escape,
the validation key, training writes, host process access, network access,
credential inheritance, compiler operation, and preservation of host metadata.
Failure or an unsupported runtime stops the launcher; there is no weaker
Windows fallback. Synthetic parent directories inside the sandbox may accept
ephemeral files, so the metadata check also verifies the actual host sentinel
remains unchanged.

Validation status (2026-11-01): both live sandbox tests passed, on native Linux
storage and the Windows-backed repository filesystem. All 13 access checks
passed in each location, including preservation of the real host metadata
sentinel after a sandbox write attempt. All six launcher tests also passed,
including refusal to start a model when an isolation check fails. No model
session or validation call was used. Each model launch still repeats its own
checks; `agent.py <run> --check` can run them separately.

These controls restrict tool access to additional data. They cannot remove
knowledge already present in a pretrained model. OS/kernel compromise and
deliberate changes by the trusted host user are outside this boundary.

Reference: [official Codex permission profiles](https://learn.chatgpt.com/docs/permissions).
