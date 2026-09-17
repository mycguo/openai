"""Network-free article/image/publish workflow regression tests."""

import base64
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from urllib.parse import urlencode

from streamlit.testing.v1 import AppTest

import ai_podcast as services
from youtube_publishing import DEFAULT_PROMPT, validate_callback, post_fingerprint


APP = str(Path(__file__).resolve().parents[1] / "youtube.py")
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


def element(elements, label):
    return next(item for item in elements if item.label == label)


class PublishingTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(APP)
        app.session_state.yt_transcript_text = "word " * 8000 + "THE END"
        app.session_state.yt_transcript_srt = ""
        app.session_state.yt_transcript_metadata = {"webpage_url": "https://youtube.com/watch?v=jNQXAC9IVRw"}
        return app.run()

    def test_prompt_edit_reset_and_no_automatic_generation(self):
        with patch.object(services, "generate_linkedin_article") as generate:
            app = self.app()
            self.assertFalse(app.exception)
            self.assertEqual(element(app.text_area, "Article generation prompt").value, DEFAULT_PROMPT)
            element(app.text_area, "Article generation prompt").set_value("Explain this for architects. No emojis.").run()
            self.assertEqual(app.session_state.yt_article_prompt, "Explain this for architects. No emojis.")
            element(app.button, "Reset article prompt").click().run()
            self.assertEqual(app.session_state.yt_article_prompt, DEFAULT_PROMPT)
            generate.assert_not_called()

    def test_generate_receives_full_transcript_and_edited_prompt(self):
        with patch.object(services, "ANTHROPIC_API_KEY", "test"), patch.object(services, "generate_linkedin_article", return_value="Draft") as generate:
            app = self.app()
            element(app.text_area, "Article generation prompt").set_value("Custom instructions").run()
            element(app.button, "Generate Article").click().run()
            self.assertFalse(app.exception)
            args, kwargs = generate.call_args
            self.assertTrue(args[0].endswith("THE END"))
            self.assertGreater(len(args[0]), 40000)
            self.assertEqual(kwargs["prompt_override"], "Custom instructions")
            self.assertEqual(app.session_state.yt_article, "Draft")

    def test_shared_prompt_composition_and_refusal(self):
        with patch.object(services, "generate_with_claude", return_value="Draft") as generate:
            services.generate_linkedin_article("FULL TRANSCRIPT END", prompt_override="My custom {instructions}")
            prompt = generate.call_args.args[0]
            self.assertIn("My custom {instructions}", prompt)
            self.assertTrue(prompt.endswith("FULL TRANSCRIPT END"))
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(content=[], stop_reason="refusal")
        with patch.object(services, "_create_anthropic_client", return_value=client), self.assertRaisesRegex(RuntimeError, "declined"):
            services.generate_with_claude("test")

    def test_image_uses_edited_article_and_is_invalidated(self):
        payload = {"bytes": PNG, "mime_type": "image/png"}
        with patch.object(services, "GOOGLE_API_KEY", "test"), patch.object(services, "generate_article_image", return_value=(True, payload, "prompt", None)) as generate:
            app = self.app()
            app.session_state.yt_article = "Original draft"
            app.run()
            element(app.text_area, "Edit Article").set_value("Edited draft").run()
            element(app.button, "Generate Article Image").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generate.call_args.args[0], "Edited draft")
            self.assertEqual(app.session_state.yt_article_image, payload)
            element(app.text_area, "Edit Article").set_value("Different draft").run()
            self.assertNotIn("yt_article_image", app.session_state)
            self.assertFalse(app.session_state.yt_publish_confirm)

    def test_publish_requires_review_and_prevents_duplicates(self):
        with patch.object(services, "post_to_linkedin", return_value=(True, {"id": "urn:li:share:123"})) as publish:
            app = self.app()
            app.session_state.yt_article = "Reviewed draft"
            app.session_state.yt_linkedin = {"token": "test", "author": "urn:li:person:test", "expires": time.time() + 1000}
            app.run()
            self.assertTrue(element(app.button, "Publish to LinkedIn").disabled)
            confirm = next(c for c in app.checkbox if c.key == "yt_publish_confirm")
            confirm.check().run()
            element(app.button, "Publish to LinkedIn").click().run()
            self.assertFalse(app.exception)
            publish.assert_called_once_with("Reviewed draft", "test", "urn:li:person:test", image_payload=None, allow_image_fallback=False)
            self.assertTrue(element(app.button, "Publish to LinkedIn").disabled)
            app.run()
            publish.assert_called_once()

    def test_failed_publish_not_automatically_retried(self):
        with patch.object(services, "post_to_linkedin", side_effect=TimeoutError) as publish:
            app = self.app()
            app.session_state.yt_article = "Draft"
            app.session_state.yt_linkedin = {"token": "test", "author": "urn:li:person:test", "expires": time.time() + 1000}
            app.session_state.yt_publish_confirm = True
            app.run()
            element(app.button, "Publish to LinkedIn").click().run()
            self.assertTrue(element(app.button, "Publish to LinkedIn").disabled)
            app.run()
            publish.assert_called_once()

    def test_oversize_and_expired_token_block_publish(self):
        app = self.app()
        app.session_state.yt_article = "x" * 3001
        app.session_state.yt_linkedin = {"token": "test", "author": "urn:li:person:test", "expires": time.time() + 1000}
        app.session_state.yt_publish_confirm = True
        app.run()
        self.assertTrue(element(app.button, "Publish to LinkedIn").disabled)
        app.session_state.yt_linkedin = {"token": "test", "author": "urn:li:person:test", "expires": 1}
        app.run()
        self.assertNotIn("yt_linkedin", app.session_state)
        self.assertTrue(element(app.button, "Publish to LinkedIn").disabled)

    def test_oauth_state_redirect_and_expiry(self):
        pending = {"state": "random-state", "created": 1000, "redirect_uri": "http://localhost:8502/"}
        url = pending["redirect_uri"] + "?" + urlencode({"state": pending["state"], "code": "test-code"})
        self.assertEqual(validate_callback(url, pending, now=1001), "test-code")
        for bad in [url.replace("random-state", "wrong"), url.replace("8502", "8501"), url + "&code=second"]:
            with self.assertRaises(ValueError):
                validate_callback(bad, pending, now=1001)
        with self.assertRaises(ValueError):
            validate_callback(url, pending, now=1700)
        with self.assertRaises(ValueError):
            validate_callback(url, None)

    def test_connection_exchanges_once_and_clears_callback(self):
        config = {"client_id": "test-client", "client_secret": "test-secret", "redirect_uri": "http://localhost:8502/"}
        with patch("youtube_publishing.linkedin_config", return_value=config), \
             patch.object(services, "exchange_code_for_token", return_value={"access_token": "test-token", "expires_in": 3600}) as exchange, \
             patch.object(services, "fetch_authenticated_member_urn", return_value="urn:li:person:test"):
            app = self.app()
            element(app.button, "Start LinkedIn connection").click().run()
            pending = app.session_state.yt_oauth_pending
            callback = config["redirect_uri"] + "?" + urlencode({"state": pending["state"], "code": "test-code"})
            element(app.text_input, "Returned callback URL").set_value(callback)
            element(app.button, "Complete connection").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.yt_linkedin["author"], "urn:li:person:test")
            self.assertNotIn("yt_oauth_pending", app.session_state)
            self.assertNotIn("yt_oauth_callback", app.session_state)
            app.run()
            exchange.assert_called_once()

    def test_image_upload_failure_never_posts_text_only(self):
        with patch.object(services, "initialize_linkedin_image_upload", return_value=(False, None, None, "test failure")), \
             patch.object(services.requests, "post") as create:
            ok, _ = services.post_to_linkedin("Draft", "test-token", "urn:li:person:test", {"bytes": PNG}, allow_image_fallback=False)
            self.assertFalse(ok)
            create.assert_not_called()

    def test_linkedin_version_uses_active_release_and_rejects_invalid_format(self):
        self.assertEqual(services.DEFAULT_LINKEDIN_API_VERSION, "202609")
        response = Mock(status_code=201, content=b"", headers={})
        response.raise_for_status.return_value = None
        with patch.object(services, "LINKEDIN_API_VERSION", services.DEFAULT_LINKEDIN_API_VERSION), \
             patch.object(services.requests, "post", return_value=response) as create:
            ok, _ = services.post_to_linkedin("Draft", "test-token", "urn:li:person:test")
            self.assertTrue(ok)
            self.assertEqual(create.call_args.kwargs["headers"]["Linkedin-Version"], "202609")

        with patch.object(services, "LINKEDIN_API_VERSION", "20250901"), \
             patch.object(services.requests, "post") as create:
            ok, error = services.post_to_linkedin("Draft", "test-token", "urn:li:person:test")
            self.assertFalse(ok)
            self.assertIn("YYYYMM", error)
            create.assert_not_called()

    def test_fingerprints_include_image_and_account(self):
        self.assertNotEqual(post_fingerprint("a", None, "x"), post_fingerprint("a", {"bytes": PNG}, "x"))
        self.assertNotEqual(post_fingerprint("a", None, "x"), post_fingerprint("a", None, "y"))


if __name__ == "__main__":
    unittest.main()
