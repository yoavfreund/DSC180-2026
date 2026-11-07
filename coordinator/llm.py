# coordinator/llm.py
import json
import os
import urllib.request

COORD_SYSTEM = """You select weak learners for a boosting ensemble.
Allowed families: "stump" (no params), "tree" (max_depth 1-6, min_leaf 1-500),
"logreg" (iters 10-5000, lr 1e-4 to 1, l2 0 to 10).
For each choose "features": "pixels", "lbp", or "rectangles".
Given the history and the latest evaluator feedback, propose K specs that differ
meaningfully. Return ONLY a JSON list of K objects with "family" and that family's
params. No labels, weights, or alphas."""

def propose_raw(history, feedback, k, timeout=20):
    user = (f"History (most recent last):\n{json.dumps(history)}\n\n"
            f"Latest feedback:\n{feedback or 'None (first attempt).'}\n\n"
            f"Propose {k} specs as a JSON list.")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps({"model": "gpt-6-luna", "reasoning": {"effort": "low"},
                       "instructions": COORD_SYSTEM, "input": user,
                       "max_output_tokens": 1500}).encode()
    request = urllib.request.Request("https://api.openai.com/v1/responses", data=body,
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    texts = [part["text"] for item in result.get("output", [])
             for part in item.get("content", []) if part.get("type") == "output_text"]
    if not texts:
        raise ValueError("LLM response contained no text")
    return "\n".join(texts)
