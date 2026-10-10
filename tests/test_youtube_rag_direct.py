"""Direct Gemini media requests and boundary validation, without provider access."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from google import genai
from google.genai import errors, types

from media_rag.config import RagError, Settings
from youtube_rag.direct_gemini import ask_youtube, load_direct_settings, validate_direct_answer


URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
SETTINGS = Settings(gemini_api_key="test-key")
ANSWER = {"status": "answered", "message": "", "claims": [
    {"text": "The speaker explains the benefit.", "start_sec": 5, "end_sec": 20},
]}


class DirectGeminiTests(unittest.TestCase):
    def client(self, response=None):
        client = Mock()
        client.models.generate_content.return_value = response or SimpleNamespace(text=json.dumps(ANSWER))
        return client

    def test_sdk_wire_sends_canonical_video_as_media_and_question_without_file_upload(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
                "role": "model", "parts": [{"text": json.dumps(ANSWER)}],
            }}]})

        sdk = genai.Client(api_key="test-key", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(respond))))
        self.addCleanup(sdk.close)
        client = self.client()
        client.models = sdk.models
        result = ask_youtube(SETTINGS, "https://youtu.be/jNQXAC9IVRw?si=tracking&t=5", " Explain the benefit? ", client)
        self.assertEqual(result, ANSWER)
        self.assertEqual(len(requests), 1)
        self.assertIn(SETTINGS.gemini_model + ":generateContent", str(requests[0].url))
        self.assertEqual(requests[0].headers["x-goog-api-key"], "test-key")
        body = json.loads(requests[0].content)
        parts = body["contents"][0]["parts"]
        file_data = parts[0]["fileData"]
        self.assertEqual(file_data.get("fileUri", file_data.get("file_uri")), URL)
        self.assertEqual(parts[1]["text"], "Question: Explain the benefit?")
        config = body["generationConfig"]
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertEqual(config["responseJsonSchema"]["required"], ["status", "message", "claims"])
        for parameter in ["temperature", "topP", "topK"]:
            self.assertNotIn(parameter, config)
        client.files.upload.assert_not_called()
        client.close.assert_not_called()

    def test_invalid_inputs_and_missing_key_never_construct_a_client(self):
        for url, question, settings in [("http://127.0.0.1/private", "Question", SETTINGS),
                                         (URL, "", SETTINGS), (URL, "q" * 2001, SETTINGS),
                                         (URL, None, SETTINGS), (URL, "Question", Settings())]:
            with self.subTest(url=url, question=question), patch("youtube_rag.direct_gemini.genai.Client") as factory:
                with self.assertRaises(RagError):
                    ask_youtube(settings, url, question)
                factory.assert_not_called()

    def test_owned_client_has_a_deadline_and_no_automatic_retries_and_is_closed_on_failure(self):
        client = self.client()
        client.models.generate_content.side_effect = httpx.ReadTimeout("private-key")
        with patch("youtube_rag.direct_gemini.genai.Client", return_value=client) as factory:
            with self.assertLogs("media_rag.gemini"), self.assertRaisesRegex(RagError, "shorter public video"):
                ask_youtube(SETTINGS, URL, "Question")
        options = factory.call_args.kwargs["http_options"]
        self.assertEqual(options.timeout, 180_000)
        self.assertEqual(options.retry_options.attempts, 1)
        client.close.assert_called_once()

    def test_provider_failures_show_safe_status_guidance(self):
        for code, guidance in [(400, "compatibility"), (403, "permission"),
                                (404, "MEDIA_RAG_GEMINI_MODEL"), (429, "quota"), (503, "Retry later")]:
            with self.subTest(code=code):
                client = self.client()
                client.models.generate_content.side_effect = errors.APIError(
                    code, {"error": {"message": "private-key private-question private-url"}})
                with self.assertLogs("media_rag.gemini") as logs, self.assertRaises(RagError) as raised:
                    ask_youtube(SETTINGS, URL, "Question", client)
                self.assertIn(f"HTTP {code}", str(raised.exception))
                self.assertIn(guidance, str(raised.exception))
                for private in ["private-key", "private-question", "private-url"]:
                    self.assertNotIn(private, str(raised.exception) + " ".join(logs.output))

    def test_blocked_truncated_and_unreadable_responses_do_not_become_answers(self):
        for response, message in [
            (SimpleNamespace(text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY")), "blocked"),
            (SimpleNamespace(text=json.dumps(ANSWER), candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")]), "cut off"),
            (SimpleNamespace(text=json.dumps(ANSWER), candidates=[SimpleNamespace(finish_reason="SAFETY")]), "blocked"),
            (SimpleNamespace(text=None), "no answer text"),
            (SimpleNamespace(text="private-response"), "unreadable structured"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(RagError, message) as raised:
                ask_youtube(SETTINGS, URL, "Question", self.client(response))
            self.assertNotIn("private-response", str(raised.exception))

    def test_invalid_timestamps_and_claims_are_rejected(self):
        changes = [dict(start_sec=True), dict(start_sec=5.0), dict(start_sec=-1),
                   dict(start_sec="5"), dict(start_sec=float("nan")), dict(end_sec=5),
                   dict(end_sec=96), dict(start_sec=86400, end_sec=86401),
                   dict(end_sec=float("inf")), dict(text=""), dict(text="x" * 2001)]
        for change in changes:
            with self.subTest(change=change):
                payload = deepcopy(ANSWER)
                payload["claims"][0].update(change)
                with self.assertRaises(RagError):
                    validate_direct_answer(payload)
        for payload in [None, {}, {**ANSWER, "status": []}, {**ANSWER, "claims": []}, {**ANSWER, "claims": [None]},
                        {**ANSWER, "claims": ANSWER["claims"] * 9}, {**ANSWER, "message": "x" * 1001},
                        {**ANSWER, "status": "insufficient_evidence"}]:
            with self.subTest(payload=payload), self.assertRaises(RagError):
                validate_direct_answer(payload)

    def test_generated_urls_are_discarded_and_insufficient_evidence_is_supported(self):
        payload = deepcopy(ANSWER)
        payload["claims"][0]["url"] = "https://malicious.example/"
        self.assertEqual(validate_direct_answer(payload), ANSWER)
        self.assertEqual(validate_direct_answer({"status": "insufficient_evidence", "message": "Not discussed.",
                                                 "claims": []})["message"], "Not discussed.")


class DirectSettingsTests(unittest.TestCase):
    def test_key_only_configuration_ignores_invalid_infrastructure(self):
        with patch.dict(os.environ, {}, clear=True), patch("youtube_rag.direct_gemini.dotenv_values", return_value={}):
            settings = load_direct_settings({"GEMINI_API_KEY": "test", "MEDIA_RAG_STORAGE_PROVIDER": "invalid",
                                             "MEDIA_RAG_DATABASE_URL_FILE": "/missing/db-secret",
                                             "AWS_SECRET_ACCESS_KEY_FILE": "/missing/storage-secret"})
        self.assertEqual(settings.gemini_api_key, "test")
        self.assertEqual(settings.gemini_model, "gemini-3.8-flash")
        self.assertEqual(settings.database_url, "")

    def test_precedence_matches_existing_app_and_supports_google_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key", "MEDIA_RAG_GEMINI_MODEL": "env-model"}, clear=True), \
                patch("youtube_rag.direct_gemini.dotenv_values", return_value={"GOOGLE_API_KEY": "file-key"}):
            self.assertEqual(load_direct_settings().gemini_api_key, "env-key")
            settings = load_direct_settings({"GEMINI_API_KEY": "table-key", "MEDIA_RAG_GEMINI_MODEL": "table-model"})
            self.assertEqual(settings.gemini_api_key, "table-key")
            self.assertEqual(settings.gemini_model, "table-model")

    def test_explicit_secret_file_fails_closed_without_google_fallback(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch("youtube_rag.direct_gemini.dotenv_values", return_value={}):
            path = Path(directory) / "key"
            values = {"GEMINI_API_KEY_FILE": str(path), "GEMINI_API_KEY": "inline", "GOOGLE_API_KEY": "fallback"}
            self.assertEqual(load_direct_settings(values).gemini_api_key, "")
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(RagError, "secret file"):
                load_direct_settings(values)
            path.write_text("file-key\n")
            self.assertEqual(load_direct_settings(values).gemini_api_key, "file-key")


if __name__ == "__main__":
    unittest.main()
