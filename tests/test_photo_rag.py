"""Native image requests, safe decoding, and photo service boundaries; no paid calls."""

from copy import deepcopy
import base64
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from google import genai
from google.genai import errors, types
from PIL import Image

from media_rag.config import RagError, Settings
from photo_rag.config import DIMENSIONS, MAX_PHOTO_BYTES, load_settings, validate_vector
from photo_rag.gemini import PhotoGemini
from photo_rag.images import prepare_image
from photo_rag.service import PhotoEvidence, PhotoLibrary


SETTINGS = Settings(gemini_api_key="test-key")
VECTOR = [1.0] + [0.0] * (DIMENSIONS - 1)
ANSWER = {"status": "answered", "message": "", "claims": [{"text": "A red object is visible.", "source_ids": [1]}]}
PHOTO = {"id": "photo-1", "title": "Photo", "status": "queued", "image_key": "photos/1/image.jpg",
         "thumbnail_key": "photos/1/thumbnail.jpg", "object_key": "photos/1/original.png"}
EVIDENCE = [PhotoEvidence(1, "photo-1", "Photo", PHOTO["image_key"], PHOTO["thumbnail_key"], PHOTO["object_key"])]


def upload(format="PNG", name="photo.png"):
    result = io.BytesIO()
    Image.new("RGB", (60, 40), "red").save(result, format=format)
    result.seek(0)
    result.name = name
    return result


class ImageTests(unittest.TestCase):
    def test_normalizes_orientation_and_removes_metadata_from_model_images(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.jpg"
            exif = Image.Exif()
            exif[274] = 6
            exif[270] = "Private metadata"
            Image.new("RGB", (60, 40), "red").save(path, exif=exif)
            prepared = prepare_image(path, directory)
            self.assertEqual((prepared["width"], prepared["height"]), (40, 60))
            with Image.open(prepared["image_path"]) as result:
                self.assertEqual(result.size, (40, 60))
                self.assertFalse(result.getexif())
                self.assertNotIn("Private metadata", str(result.info))
            with Image.open(prepared["thumbnail_path"]) as result:
                self.assertLessEqual(max(result.size), 400)

    def test_heic_png_webp_and_jpeg_are_decoded_as_real_photos(self):
        for format, extension in [("HEIF", ".heic"), ("PNG", ".png"), ("WEBP", ".webp"), ("JPEG", ".jpg")]:
            with self.subTest(format=format), TemporaryDirectory() as directory:
                source = Path(directory) / "source"
                source.write_bytes(upload(format).getvalue())
                result = prepare_image(source, directory)
                self.assertEqual(result["extension"], extension)
                self.assertGreater(result["image_path"].stat().st_size, 0)

    def test_invalid_unsupported_animated_and_large_images_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            for content in [b"", b"<svg><script>private</script></svg>", b"not-an-image"]:
                path.write_bytes(content)
                with self.assertRaises(RagError):
                    prepare_image(path, directory)
            Image.new("RGB", (10, 10)).save(path, format="BMP")
            with self.assertRaisesRegex(RagError, "JPEG"):
                prepare_image(path, directory)
            Image.new("RGB", (10, 10), "red").save(path, format="WEBP", save_all=True,
                                                    append_images=[Image.new("RGB", (10, 10), "blue")])
            with self.assertRaisesRegex(RagError, "still photo"):
                prepare_image(path, directory)
            path.write_bytes(upload().getvalue())
            with patch("photo_rag.images.MAX_PIXELS", 100), self.assertRaisesRegex(RagError, "megapixels"):
                prepare_image(path, directory)


class PhotoProviderTests(unittest.TestCase):
    def test_actual_sdk_recovers_from_503_then_500_with_the_same_photo_request(self):
        requests = []

        def respond(request):
            requests.append(request)
            if len(requests) < 3:
                return httpx.Response([503, 500][len(requests) - 1], json={
                    "error": {"code": [503, 500][len(requests) - 1], "message": "private-provider-body"}})
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
                "role": "model", "parts": [{"text": json.dumps(ANSWER)}]}}]})

        # Leave SDK defaults enabled here to prove per-request retries cannot nest.
        sdk = genai.Client(api_key="test-key", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(respond))))
        self.addCleanup(sdk.close)
        with TemporaryDirectory() as directory, patch("photo_rag.gemini.time.sleep") as sleep:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(upload("JPEG").getvalue())
            self.assertEqual(PhotoGemini(SETTINGS, sdk).answer("What is visible?", EVIDENCE, [path]), ANSWER)
        self.assertEqual(len(requests), 3)
        self.assertEqual([request.content for request in requests], [requests[0].content] * 3)
        self.assertEqual(len({str(request.url) for request in requests}), 1)
        self.assertEqual(sleep.call_count, 2)
        for call, minimum in zip(sleep.call_args_list, [1, 2]):
            self.assertGreaterEqual(call.args[0], minimum)
            self.assertLessEqual(call.args[0], minimum + 0.5)

    def test_retries_are_bounded_and_keep_provider_details_private(self):
        client = Mock()
        client.models.generate_content.side_effect = errors.APIError(503, {
            "error": {"message": "private-key private-photo"}})
        with TemporaryDirectory() as directory, patch("photo_rag.gemini.time.sleep") as sleep:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(upload("JPEG").getvalue())
            with self.assertLogs("media_rag.gemini") as logs, self.assertRaisesRegex(RagError, "HTTP 503.*3 evidence") as raised:
                PhotoGemini(SETTINGS, client).answer("Private question", EVIDENCE, [path])
        self.assertEqual(client.models.generate_content.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        for private in ["private-key", "private-photo", "Private question"]:
            self.assertNotIn(private, str(raised.exception) + str(logs.output))

    def test_bad_requests_quota_and_timeouts_do_not_repeat_paid_calls(self):
        with TemporaryDirectory() as directory, patch("photo_rag.gemini.time.sleep") as sleep:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(upload("JPEG").getvalue())
            for failure in [errors.APIError(code, {"error": {"message": "private"}}) for code in [400, 403, 404, 429]] + [
                    httpx.ReadTimeout("private")]:
                client = Mock()
                client.models.generate_content.side_effect = failure
                with self.subTest(failure=type(failure).__name__), self.assertRaises(RagError):
                    PhotoGemini(SETTINGS, client).answer("Question", EVIDENCE, [path])
                client.models.generate_content.assert_called_once()
            sleep.assert_not_called()

    def test_retries_share_the_original_time_budget(self):
        client = Mock()
        client.models.generate_content.side_effect = errors.APIError(500, {"error": {"message": "private"}})
        with TemporaryDirectory() as directory, patch("photo_rag.gemini.time.sleep") as sleep, \
                patch("photo_rag.gemini.random.uniform", return_value=0), \
                patch("photo_rag.gemini.time.monotonic", side_effect=[0, 0, 100, 101, 179]):
            path = Path(directory) / "photo.jpg"
            path.write_bytes(upload("JPEG").getvalue())
            with self.assertRaisesRegex(RagError, "HTTP 500"):
                PhotoGemini(SETTINGS, client).answer("Question", EVIDENCE, [path])
        self.assertEqual(client.models.generate_content.call_count, 2)
        timeouts = [call.kwargs["config"].http_options.timeout for call in client.models.generate_content.call_args_list]
        self.assertEqual(timeouts, [180_000, 79_000])
        sleep.assert_called_once_with(1)

    def test_actual_sdk_wire_embeds_one_image_and_query_in_same_space_then_sends_image_evidence(self):
        requests = []

        def respond(request):
            requests.append(request)
            if "batchEmbedContents" in str(request.url):
                return httpx.Response(200, json={"embeddings": [{"values": VECTOR}]})
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
                "role": "model", "parts": [{"text": json.dumps(ANSWER)}]}}]})

        sdk = genai.Client(api_key="test-key", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(respond))))
        self.addCleanup(sdk.close)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(upload("JPEG").getvalue())
            provider = PhotoGemini(SETTINGS, sdk)
            self.assertEqual(provider.embed_photo(path), VECTOR)
            self.assertEqual(provider.embed_query("What is visible?"), VECTOR)
            self.assertEqual(provider.answer("What is visible?", EVIDENCE, [path]), ANSWER)
            self.assertEqual(len(requests), 3)
            image, query, answer = [json.loads(request.content) for request in requests]
            self.assertIn("gemini-embedding-2:batchEmbedContents", str(requests[0].url))
            self.assertEqual(image["requests"][0]["outputDimensionality"], DIMENSIONS)
            part = image["requests"][0]["content"]["parts"][0]
            image_part = part.get("inlineData", part.get("inline_data"))
            self.assertEqual(base64.urlsafe_b64decode(image_part["data"]), path.read_bytes())
            self.assertIn("task: search result", query["requests"][0]["content"]["parts"][0]["text"])
            self.assertTrue({"inlineData", "inline_data"} & answer["contents"][0]["parts"][-1].keys())
            self.assertNotIn("temperature", answer["generationConfig"])
            self.assertEqual(answer["generationConfig"]["responseMimeType"], "application/json")

    def test_invalid_vectors_are_rejected(self):
        for value in [[], VECTOR[:-1], [0.0] * DIMENSIONS, [True] + VECTOR[1:],
                      [float("nan")] + VECTOR[1:], [float("inf")] + VECTOR[1:], ["1"] + VECTOR[1:]]:
            with self.subTest(value=value[:1]), self.assertRaises(RagError):
                validate_vector(value)

    def test_provider_error_payload_is_never_shown_or_logged(self):
        for code in [400, 403, 404, 429, 500]:
            client = Mock()
            client.models.embed_content.side_effect = errors.APIError(code, {"error": {"message": "private-key private-photo"}})
            with self.subTest(code=code), self.assertLogs("media_rag.gemini") as logs, self.assertRaises(RagError) as raised:
                PhotoGemini(SETTINGS, client).embed_query("Private question")
            self.assertIn(f"HTTP {code}", str(raised.exception))
            self.assertNotIn("private-key", str(raised.exception) + str(logs.output))
            self.assertNotIn("private-photo", str(raised.exception) + str(logs.output))

    def test_invented_citations_and_inconsistent_answers_are_rejected(self):
        client = Mock()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(upload("JPEG").getvalue())
            for changes in [{"source_ids": [2]}, {"source_ids": [True]}, {"text": ""}, {"text": "x" * 2001}]:
                payload = deepcopy(ANSWER)
                payload["claims"][0].update(changes)
                client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(payload))
                with self.subTest(changes=changes), self.assertRaises(RagError):
                    PhotoGemini(SETTINGS, client).answer("Question", EVIDENCE, [path])
            for status in [[], "insufficient_evidence"]:
                client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({**ANSWER, "status": status}))
                with self.assertRaises(RagError):
                    PhotoGemini(SETTINGS, client).answer("Question", EVIDENCE, [path])

    def test_owned_client_has_a_timeout_no_automatic_retries_and_is_closed(self):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=VECTOR)])
        with patch("photo_rag.gemini.genai.Client", return_value=client) as factory:
            with PhotoGemini(SETTINGS) as provider:
                provider.embed_query("Question")
        self.assertEqual(factory.call_args.kwargs["http_options"].timeout, 180_000)
        self.assertEqual(factory.call_args.kwargs["http_options"].retry_options.attempts, 1)
        client.close.assert_called_once()


class PhotoServiceTests(unittest.TestCase):
    def library(self):
        library = PhotoLibrary(SETTINGS, database=Mock(), storage=Mock())
        library.database.find_hash.return_value = None
        library.database.create_photo.side_effect = lambda asset: {**asset, "status": "queued"}
        library.database.claim.return_value = PHOTO
        return library

    def test_upload_persists_private_original_and_derivatives_and_reuses_duplicates(self):
        library = self.library()
        photo, created = library.add_upload(upload(), "Memory")
        self.assertTrue(created)
        self.assertEqual(library.storage.upload.call_count, 3)
        self.assertTrue(all(call.args[0].startswith("photos/") for call in library.storage.upload.call_args_list))
        library.database.find_hash.return_value = photo
        duplicate, created = library.add_upload(upload(), "Duplicate")
        self.assertFalse(created)
        self.assertEqual(photo, duplicate)
        self.assertEqual(library.storage.upload.call_count, 3)

    def test_bad_or_oversized_upload_never_writes_storage(self):
        library = self.library()
        for content in [b"", b"bad-photo", b"x" * (MAX_PHOTO_BYTES + 1)]:
            value = io.BytesIO(content)
            value.name = "photo.jpg"
            with self.assertRaises(RagError):
                library.add_upload(value)
        library.storage.upload.assert_not_called()

    def test_ambiguous_database_failure_retains_saved_objects(self):
        library = self.library()
        library.database.create_photo.side_effect = RuntimeError("DB write may have committed")
        with self.assertRaises(RuntimeError):
            library.add_upload(upload())
        library.storage.remove.assert_not_called()

    def test_index_failure_is_safe_and_leased_asset_can_be_retried(self):
        library = self.library()
        with patch("photo_rag.service.PhotoGemini") as provider:
            provider.return_value.__enter__.return_value.embed_photo.side_effect = RuntimeError("private-key")
            with self.assertRaisesRegex(RagError, "Photo indexing failed"):
                library.index_photo("photo-1", retry=True)
        library.database.claim.assert_called_once_with("photo-1", retry=True)
        self.assertNotIn("private-key", library.database.fail.call_args.args[1])
        library.database.complete.assert_not_called()

    def test_missing_key_is_checked_before_claiming_and_empty_search_does_not_call_provider(self):
        library = self.library()
        library.settings = Settings()
        with self.assertRaises(RagError):
            library.index_photo("photo-1")
        library.database.claim.assert_not_called()
        with patch("photo_rag.service.PhotoGemini") as provider:
            self.assertEqual(library.answer("Question", [])[0]["status"], "insufficient_evidence")
            provider.assert_not_called()

    def test_retrieval_scopes_selected_photos_and_answer_uses_only_matched_pixels(self):
        library = self.library()
        library.storage.download.side_effect = lambda key, path: path.write_bytes(upload("JPEG").getvalue())
        with patch("photo_rag.service.PhotoGemini") as factory:
            provider = factory.return_value.__enter__.return_value
            provider.embed_query.return_value = VECTOR
            library.retrieve("Question", ["photo-1"], 3, 0.2)
            library.database.search.assert_called_once_with(VECTOR, asset_ids=["photo-1"], limit=3, min_score=0.2)
            provider.answer.return_value = ANSWER
            answer, evidence = library.answer("Question", [PHOTO])
            self.assertEqual(answer, ANSWER)
            self.assertEqual(evidence[0].source_id, 1)
            self.assertEqual(library.storage.download.call_args.args[0], PHOTO["image_key"])

    def test_photo_settings_reuse_shared_loader_without_reading_unused_twelvelabs_secret(self):
        with patch("photo_rag.config.Settings.load", return_value=SETTINGS) as loader:
            load_settings({"GEMINI_API_KEY": "test", "TWELVELABS_API_KEY_FILE": "/missing/unused-secret"})
        self.assertEqual(loader.call_args.args[0]["GEMINI_API_KEY"], "test")
        self.assertEqual(loader.call_args.args[0]["TWELVELABS_API_KEY_FILE"], "")


if __name__ == "__main__":
    unittest.main()
