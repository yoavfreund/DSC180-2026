import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coordinator import chatgpt_auth, llm_chatgpt


class ChatGPTAdapterTests(unittest.TestCase):
    def test_rolling_limit_is_persistent(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(chatgpt_auth, "DIR", Path(tmp)):
            for _ in range(50):
                chatgpt_auth.reserve_request()
            with self.assertRaises(RuntimeError):
                chatgpt_auth.reserve_request()

    def test_stream_requires_completed_event(self):
        events = (b'data: {"type":"response.output_text.delta","delta":"[]"}\n\n'
                  b'data: {"type":"response.completed"}\n\n')
        with mock.patch.object(llm_chatgpt, "access_token", return_value="secret"), \
             mock.patch.object(llm_chatgpt, "reserve_request"), \
             mock.patch.object(llm_chatgpt.urllib.request, "urlopen", return_value=io.BytesIO(events)) as call:
            self.assertEqual(llm_chatgpt.propose_raw([], None, 1), "[]")
            payload = json.loads(call.call_args.args[0].data)
            self.assertEqual(payload["model"], "gpt-6-luna")
            self.assertEqual(payload["reasoning"]["effort"], "low")
            self.assertTrue(payload["stream"])
            self.assertFalse(payload["store"])
            self.assertNotIn("max_output_tokens", payload)
        with mock.patch.object(llm_chatgpt, "access_token", return_value="secret"), \
             mock.patch.object(llm_chatgpt, "reserve_request"), \
             mock.patch.object(llm_chatgpt.urllib.request, "urlopen", return_value=io.BytesIO(events.split(b'data: {"type":"response.completed"}')[0])):
            with self.assertRaises(RuntimeError):
                llm_chatgpt.propose_raw([], None, 1)


if __name__ == "__main__":
    unittest.main()
