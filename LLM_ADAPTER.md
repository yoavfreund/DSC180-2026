# LLM adapter contract

The coordinator loads a standalone Python file using `--llm-adapter PATH`.
Each team member can supply a different provider adapter; the coordinator,
learner and evaluation code do not need to change. The default is
`coordinator/llm.py`, which calls the separately billed OpenAI API. The adapter
is trusted Python code and runs in a child process with the container's access
and environment. Only use adapters from trusted sources.

The optional `coordinator/llm_chatgpt.py` adapter uses ChatGPT plan sign-in
instead. See the Docker Plus sign-in instructions in BOOSTING.md.

The adapter must expose:

```python
def propose_raw(history, feedback, k, timeout=20):
    """Return a JSON string containing exactly k learner specifications."""
```

`history` is a list of recent dictionaries with `spec` and `err` keys;
`feedback` is an aggregate feedback string including a bounded audit summary;
`k` is the number
of requested candidates; and `timeout` is a request timeout in seconds. The
coordinator passes `timeout=20` and enforces the round deadline by terminating
the child if necessary. Provider timeouts can occur earlier than that deadline.
No image pixels, labels, or per-image weights are sent through this interface.
Example for `k=2`:

```json
[{"family":"stump","features":"pixels"},
 {"family":"tree","features":"lbp","max_depth":2,"min_leaf":5}]
```

Supported families and parameter limits are defined in `coordinator/specs.py`;
feature sets are `pixels`, `lbp`, and `rectangles`. The response is parsed and
validated by the coordinator. If the adapter fails, times out, or returns an
invalid response, the coordinator logs a failure and uses fixed default
specifications. If fallback was already attempted in the same round, a second
fallback stops refinement without training it again. It records the raw response
and feedback in the audit log, so
adapters must never put tokens or other secrets in the response.

In WSL, pass `--llm-adapter /absolute/path/to/adapter.py`. In Docker, mount
the adapter read-only and use its container path, for example add these options
before `face-boosting` in the command in BOOSTING.md:

```powershell
--mount "type=bind,source=C:/path/to/adapter.py,target=/adapter/llm.py,readonly"
```

Then add `--llm-adapter /adapter/llm.py` after `face-boosting`. Additional
modules, credentials and environment variables required by an adapter must be
available inside the container. The stock image does not install arbitrary
provider SDKs. Never build credentials into an image or commit them.

## Information needed to implement an adapter

Provide the provider and exact model, whether to use a paid API key, a free API,
local inference, or the documented ChatGPT plan sign-in, and a link to the
provider's documentation. For a local or custom service, provide its URL as
seen from the container, API format, and model name. State any budget, request
limits, data-sharing restrictions, and desired fallback behavior. If using
ChatGPT plan sign-in, no account details are needed here. We can implement the
sign-in and then you can complete consent in
your browser. Never paste API keys, access tokens, or refresh tokens into chat;
set them locally using the adapter's setup instructions.

This adapter contract chooses among the existing bounded learner
specifications. Supporting agents that generate new learner code requires a
different learner execution interface and isolation strategy.
