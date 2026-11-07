"""ChatGPT plan adapter; authorize first with coordinator.chatgpt_auth."""
import json
import urllib.request

from coordinator.chatgpt_auth import RESOURCE, access_token, reserve_request
from coordinator.llm import COORD_SYSTEM


def propose_raw(history, feedback, k, timeout=20):
    token = access_token()
    reserve_request()  # Count attempts, including timeouts and failed responses.
    prompt = (f"History (most recent last):\n{json.dumps(history)}\n\n"
              f"Latest feedback:\n{feedback or 'None (first attempt).'}\n\n"
              f"Propose {k} specs as a JSON list.")
    body = {"model": "gpt-6-luna", "reasoning": {"effort": "low"},
            "instructions": COORD_SYSTEM,
            "input": [{"role": "user", "content": prompt}],
            "store": False, "stream": True}
    req = urllib.request.Request(RESOURCE + "/responses", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + token,
                                          "Content-Type": "application/json"})
    texts, complete, data = [], False, []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            line = line.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                payload = "\n".join(data)
                data = []
                if payload == "[DONE]":
                    continue
                event = json.loads(payload)
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    texts.append(event.get("delta", ""))
                elif kind == "response.completed":
                    complete = True
                    break
                elif kind in ("response.failed", "response.incomplete", "error"):
                    raise RuntimeError("ChatGPT inference failed or was incomplete")
    if not complete or not texts:
        raise RuntimeError("ChatGPT response did not complete with text")
    return "".join(texts)
