"""Network-free checks for the podcast LangSmith trace boundaries."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import langsmith as ls

import ai_podcast as podcast


class PodcastTracingTests(unittest.TestCase):
    def test_article_trace_keeps_the_full_model_prompt(self):
        runs = []
        parent_runs = []

        def article_prompt(source_kind):
            parent_runs.append(ls.get_current_run_tree())
            return "Article instructions"

        def create_message(**kwargs):
            runs.append(ls.get_current_run_tree())
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Generated article")],
                stop_reason="end_turn",
            )

        client = SimpleNamespace(messages=SimpleNamespace(create=create_message))
        with patch.object(podcast, "_create_anthropic_client", return_value=client):
            with patch.object(podcast, "default_article_prompt", side_effect=article_prompt):
                with ls.tracing_context(enabled="local"):
                    article = podcast.generate_linkedin_article("Source transcript", "Episode title")

        self.assertEqual(article, "Generated article")
        self.assertEqual(runs[0].name, "Claude text generation")
        self.assertEqual(parent_runs[0].name, "Generate LinkedIn article")
        self.assertEqual(runs[0].parent_run_id, parent_runs[0].id)
        self.assertIn("Source transcript", runs[0].inputs["prompt"])
        self.assertEqual(parent_runs[0].inputs["transcript_chars"], len("Source transcript"))
        self.assertNotIn("transcript", parent_runs[0].inputs)

    def test_image_trace_excludes_generated_bytes(self):
        runs = []
        parent_runs = []
        image_bytes = b"private image bytes"

        def extract_themes(article_text):
            parent_runs.append(ls.get_current_run_tree())
            return "AI"

        def generate_content(**kwargs):
            runs.append(ls.get_current_run_tree())
            return SimpleNamespace(
                parts=[SimpleNamespace(inline_data=SimpleNamespace(data=image_bytes, mime_type="image/png"))]
            )

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch.object(podcast, "_create_google_client", return_value=client):
            with patch.object(podcast, "_extract_key_themes", side_effect=extract_themes):
                with ls.tracing_context(enabled="local"):
                    success, payload, _, error = podcast.generate_article_image("Article text")

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertEqual(payload["bytes"], image_bytes)
        self.assertEqual(runs[0].name, "Google image generation")
        self.assertEqual(parent_runs[0].name, "Generate LinkedIn image")
        self.assertEqual(runs[0].parent_run_id, parent_runs[0].id)
        self.assertEqual(runs[0].outputs, {"response_received": True})
        self.assertEqual(parent_runs[0].outputs, {"success": True})


if __name__ == "__main__":
    unittest.main()
