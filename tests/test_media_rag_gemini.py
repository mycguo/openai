"""Gemini wire compatibility and safe failures; no real provider calls."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import httpx
from google import genai
from google.genai import errors, types

from media_rag.config import RagError, Settings
from media_rag.gemini import Gemini
from media_rag.models import Evidence


SETTINGS = Settings(gemini_api_key="test-key", poll_interval=0)
EVIDENCE = [Evidence(1, "Episode", "video", 120, 130, "clip.mp4")]
PATHS = [Path("/tmp/test.mp4")]
ANSWER = {"status": "answered", "message": "", "claims": [{"text": "Supported claim", "source_ids": [1]}]}


class GeminiTests(unittest.TestCase):
    def client(self):
        client = Mock()
        client.files.upload.return_value = SimpleNamespace(name="files/clip", state="ACTIVE",
                                                          uri="https://files.example/clip", mime_type="video/mp4")
        client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(ANSWER))
        return client

    def test_wire_request_omits_deprecated_sampling_parameters_and_keeps_media_schema(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
                "role": "model", "parts": [{"text": json.dumps(ANSWER)}],
            }}]})

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        sdk = genai.Client(api_key="test-key", http_options=types.HttpOptions(httpx_client=http_client))
        self.addCleanup(sdk.close)
        client = self.client()
        client.models = sdk.models
        self.assertEqual(Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS), ANSWER)
        self.assertEqual(len(requests), 1)
        self.assertIn(SETTINGS.gemini_model + ":generateContent", str(requests[0].url))
        self.assertEqual(requests[0].headers["x-goog-api-key"], "test-key")
        body = json.loads(requests[0].content)
        config = body["generationConfig"]
        for field in ["temperature", "topP", "topK"]:
            self.assertNotIn(field, config)
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertEqual(config["responseJsonSchema"]["required"], ["status", "message", "claims"])
        part = body["contents"][0]["parts"][-1]
        file_data = part["fileData"]
        self.assertEqual(file_data.get("fileUri", file_data.get("file_uri")), "https://files.example/clip")
        self.assertEqual(part["videoMetadata"]["fps"], 2)
        client.files.delete.assert_called_once_with(name="files/clip")

    def test_provider_errors_are_actionable_and_private_payloads_are_not_logged_or_shown(self):
        expected = {400: "compatibility", 401: "GEMINI_API_KEY", 402: "billing", 403: "permission",
                    404: "MEDIA_RAG_GEMINI_MODEL", 429: "quota", 500: "Retry later", 503: "Retry later",
                    504: "Retry later"}
        for code, guidance in expected.items():
            with self.subTest(code=code):
                client = self.client()
                client.models.generate_content.side_effect = errors.APIError(
                    code, {"error": {"message": "private-key private-question private-file"}})
                with self.assertLogs("media_rag.gemini", level="WARNING") as logs, self.assertRaises(RagError) as raised:
                    Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS)
                self.assertIn(f"HTTP {code}", str(raised.exception))
                self.assertIn(guidance, str(raised.exception))
                self.assertIn(f"http_status={code}", logs.output[0])
                for private in ["private-key", "private-question", "private-file"]:
                    self.assertNotIn(private, str(raised.exception) + " ".join(logs.output))
                client.files.delete.assert_called_once_with(name="files/clip")

    def test_upload_failure_is_identified_and_does_not_attempt_generation(self):
        client = self.client()
        client.files.upload.side_effect = errors.APIError(403, {"error": {"message": "private-key"}})
        with self.assertLogs("media_rag.gemini"), self.assertRaisesRegex(RagError, "upload evidence clips.*HTTP 403"):
            Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS)
        client.models.generate_content.assert_not_called()
        client.files.delete.assert_not_called()

    def test_processing_file_not_found_is_not_reported_as_missing_model(self):
        client = self.client()
        client.files.upload.return_value.state = "PROCESSING"
        client.files.get.side_effect = errors.APIError(404, {"error": {"message": "private-file"}})
        with self.assertLogs("media_rag.gemini"), self.assertRaises(RagError) as raised:
            Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS)
        self.assertIn("process evidence clips (HTTP 404)", str(raised.exception))
        self.assertNotIn("MEDIA_RAG_GEMINI_MODEL", str(raised.exception))
        client.models.generate_content.assert_not_called()
        client.files.delete.assert_called_once_with(name="files/clip")

    def test_empty_and_invalid_json_are_not_reported_as_credentials_or_quota_errors(self):
        for value, guidance in [(None, "no answer text"), ("", "no answer text"), ("private-response", "structured answer")]:
            with self.subTest(value=value):
                client = self.client()
                client.models.generate_content.return_value = SimpleNamespace(text=value)
                with self.assertRaisesRegex(RagError, guidance) as raised:
                    Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS)
                self.assertNotIn("private-response", str(raised.exception))
                client.files.delete.assert_called_once_with(name="files/clip")

    def test_blocked_or_truncated_responses_do_not_return_partial_claims(self):
        for finish, guidance in [("MAX_TOKENS", "cut off"), ("SAFETY", "blocked"), ("RECITATION", "blocked")]:
            with self.subTest(finish=finish):
                client = self.client()
                client.models.generate_content.return_value = SimpleNamespace(
                    text=json.dumps(ANSWER), candidates=[SimpleNamespace(finish_reason=finish)])
                with self.assertRaisesRegex(RagError, guidance):
                    Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS)
                client.files.delete.assert_called_once_with(name="files/clip")

    def test_prompt_block_is_explained_before_attempting_to_parse_empty_text(self):
        client = self.client()
        client.models.generate_content.return_value = SimpleNamespace(
            text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
        with self.assertRaisesRegex(RagError, "blocked this request"):
            Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS)
        client.files.delete.assert_called_once_with(name="files/clip")

    def test_unspecified_block_reason_does_not_reject_a_successful_response(self):
        client = self.client()
        client.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(ANSWER), candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
            prompt_feedback=SimpleNamespace(block_reason=types.BlockedReason.BLOCKED_REASON_UNSPECIFIED))
        self.assertEqual(Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS), ANSWER)

    def test_transport_failures_and_unexpected_errors_remain_safe(self):
        for error, guidance in [(httpx.ReadTimeout("private-key"), "timed out"),
                                (httpx.ConnectError("private-key"), "Could not reach"),
                                (RuntimeError("private-key"), "app logs")]:
            with self.subTest(error=type(error).__name__):
                client = self.client()
                client.models.generate_content.side_effect = error
                with self.assertLogs("media_rag.gemini") as logs, self.assertRaisesRegex(RagError, guidance) as raised:
                    Gemini(SETTINGS, client).answer("Question", EVIDENCE, PATHS)
                self.assertNotIn("private-key", str(raised.exception) + " ".join(logs.output))
                client.files.delete.assert_called_once_with(name="files/clip")


if __name__ == "__main__":
    unittest.main()
