"""Network-free native media provider, evidence, and worker regression tests."""

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

from media_rag.config import RagError, Settings
from media_rag.gemini import Gemini, validate_answer
from media_rag.marengo import Marengo, MarengoError, parse_segments
from media_rag.models import Evidence, Hit, merge_hits, validate_vector
from media_rag.service import MediaLibrary, index_asset
from media_rag.worker import run_once


VECTOR = [1.0] + [0.0] * 511
SETTINGS = Settings(twelvelabs_api_key="test-twelve-key", gemini_api_key="test-gemini-key", poll_interval=0)


def payload(kind="audio", duration=5):
    options = ["audio"] if kind == "audio" else ["audio", "visual"]
    return {"status": "ready", "metadata": {"embedding_dimension": 512}, "data": [
        {"embedding_scope": "clip", "embedding_option": option, "start_sec": 0,
         "end_sec": duration, "embedding": VECTOR} for option in options
    ]}


class ProviderTests(unittest.TestCase):
    def marengo(self, responses):
        session = Mock()
        session.headers = {}
        session.request.side_effect = [SimpleNamespace(status_code=200, json=lambda value=value: value)
                                       for value in responses]
        return Marengo(SETTINGS, session=session), session

    def test_query_uses_v2_multi_input_same_model_and_dimensions(self):
        client, session = self.marengo([{"data": [{"embedding": VECTOR}]}])
        self.assertEqual(client.embed_query("a speaker explaining retrieval"), VECTOR)
        args, kwargs = session.request.call_args
        self.assertEqual(args, ("POST", "https://api.twelvelabs.io/v1.3/embed-v2"))
        self.assertEqual(kwargs["json"], {"input_type": "multi_input", "model_name": "marengo3.5",
                                         "embedding_dimension": 512,
                                         "multi_input": {"input_text": "a speaker explaining retrieval"}})

    def test_task_embeds_both_native_video_modalities_and_retains_tail(self):
        client, session = self.marengo([{"_id": "task"}])
        self.assertEqual(client.create_task("remote", "video", True), "task")
        request = session.request.call_args.kwargs["json"]
        self.assertEqual(request["model_name"], "marengo3.5")
        self.assertEqual(request["video"]["media_source"], {"asset_id": "remote"})
        self.assertEqual(request["video"]["embedding_option"], ["visual", "audio"])
        self.assertEqual(request["video"]["embedding_scope"], ["clip"])
        self.assertEqual(request["video"]["embedding_type"], ["separate_embedding"])
        self.assertEqual(request["video"]["segmentation"]["temporal"]["strategy"], "dynamic")

    def test_silent_video_does_not_request_an_audio_track(self):
        client, session = self.marengo([{"_id": "task"}])
        client.create_task("remote", "video", False)
        video = session.request.call_args.kwargs["json"]["video"]
        self.assertEqual(video["embedding_option"], ["visual"])
        self.assertNotIn("embedding_type", video)

    def test_audio_does_not_use_a_text_transcription_embedding(self):
        client, session = self.marengo([{"_id": "task"}])
        client.create_task("remote", "audio", True)
        audio = session.request.call_args.kwargs["json"]["audio"]
        self.assertEqual(audio["embedding_option"], ["audio"])
        self.assertNotIn("video", session.request.call_args.kwargs["json"])

    def test_local_media_is_uploaded_directly_instead_of_a_loopback_url(self):
        client, session = self.marengo([{"_id": "remote", "status": "processing"}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            path.write_bytes(b"media")
            client.upload(path, "audio/wav")
        args, kwargs = session.request.call_args
        self.assertTrue(args[1].endswith("/assets"))
        self.assertEqual(kwargs["data"]["method"], "direct")
        self.assertNotIn("url", kwargs["data"])
        self.assertEqual(kwargs["files"]["file"][0], "source.wav")

    def test_polling_renews_lease_until_ready(self):
        client, session = self.marengo([{"status": "processing"}, payload()])
        heartbeat = Mock()
        self.assertEqual(client.wait_task("task", heartbeat)["status"], "ready")
        self.assertEqual(heartbeat.call_count, 2)
        self.assertEqual(session.request.call_count, 2)

    def test_failed_provider_task_is_marked_for_reset(self):
        client, _ = self.marengo([{"status": "failed", "error": {"message": "secret-body"}}])
        with self.assertRaises(MarengoError) as raised:
            client.wait_task("task", Mock())
        self.assertTrue(raised.exception.reset_task)
        self.assertNotIn("secret-body", str(raised.exception))

    def test_http_errors_and_network_errors_do_not_expose_provider_bodies(self):
        client, session = self.marengo([])
        session.request.side_effect = None
        session.request.return_value = SimpleNamespace(status_code=429, text="private-api-key")
        with self.assertRaisesRegex(MarengoError, "429") as raised:
            client.embed_query("question")
        self.assertNotIn("private-api-key", str(raised.exception))
        session.request.side_effect = requests.ConnectionError("private-api-key")
        with self.assertRaises(MarengoError) as raised:
            client.embed_query("question")
        self.assertNotIn("private-api-key", str(raised.exception))

    def test_invalid_vectors_and_timestamps_are_rejected(self):
        for vector in [None, [], [0.0] * 512, [float("nan")] * 512, [float("inf")] * 512, [1.0] * 511]:
            with self.subTest(vector_type=type(vector)), self.assertRaises(RagError):
                validate_vector(vector)
        for start, end in [(-1, 5), (1, 1), (0, 10), (float("nan"), 5)]:
            response = payload()
            response["data"][0].update(start_sec=start, end_sec=end)
            with self.subTest(start=start, end=end), self.assertRaises(RagError):
                parse_segments(response, 5, "audio", True)

    def test_asset_scope_is_not_mistaken_for_searchable_clip_scope(self):
        response = payload()
        response["data"][0]["embedding_scope"] = "asset"
        with self.assertRaises(RagError):
            parse_segments(response, 5, "audio", True)

    def test_missing_video_modality_is_not_published_as_complete(self):
        with self.assertRaises(RagError):
            parse_segments(payload("audio"), 5, "video", True)

    def test_mixed_model_dimensions_are_rejected(self):
        response = payload()
        response["metadata"]["embedding_dimension"] = 256
        with self.assertRaises(RagError):
            parse_segments(response, 5, "audio", True)

    def test_timestamp_tolerance_cannot_create_a_reversed_clip(self):
        response = payload()
        response["data"][0].update(start_sec=5.1, end_sec=5.4)
        with self.assertRaises(RagError):
            parse_segments(response, 5, "audio", True)


class EvidenceTests(unittest.TestCase):
    def hit(self, start=10, end=15, modality="audio", asset_id="asset", score=0.9):
        return Hit(asset_id, "Title", "original/source.mp4", "video", 100, start, end, score, (modality,))

    def test_duplicate_modalities_and_neighbors_merge_with_original_offsets(self):
        merged = merge_hits([self.hit(), self.hit(modality="visual"), self.hit(16, 20, score=0.8)])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (2, 28))
        self.assertEqual(merged[0].modalities, ("audio", "visual"))

    def test_sources_stay_separate_and_context_clamps_to_duration(self):
        merged = merge_hits([self.hit(0, 5), self.hit(95, 100, asset_id="other")])
        self.assertEqual([(hit.start, hit.end) for hit in merged], [(0, 13), (87, 100)])

    def test_merging_does_not_exceed_clip_duration_budget(self):
        hits = [self.hit(start, start + 5, score=1 - start / 1000) for start in range(0, 100, 5)]
        self.assertTrue(all(hit.end - hit.start <= 90 for hit in merge_hits(hits)))

    def test_coarse_provider_segment_is_bounded_and_inside_source(self):
        hit = Hit("asset", "Title", "source.mp4", "video", 200, 0, 200, 0.9, ("visual",))
        merged = merge_hits([hit])[0]
        self.assertEqual((merged.start, merged.end), (55, 145))

    def test_unknown_missing_or_boolean_citations_are_rejected(self):
        evidence = [Evidence(1, "Title", "audio", 10, 20, "clip.mp3")]
        for ids in [[2], [], [True], None]:
            with self.subTest(ids=ids), self.assertRaises(RagError):
                validate_answer({"status": "answered", "claims": [{"text": "Claim", "source_ids": ids}]}, evidence)

    def test_insufficient_evidence_never_renders_uncited_claims(self):
        result = validate_answer({"status": "insufficient_evidence", "message": "Not in the clips",
                                  "claims": [{"text": "Unverified", "source_ids": [99]}]}, [])
        self.assertEqual(result["claims"], [])

    def test_gemini_receives_raw_media_and_cleans_files_even_if_citation_invalid(self):
        client = Mock()
        client.files.upload.return_value = SimpleNamespace(name="files/clip", state="ACTIVE",
                                                          uri="https://files.example/clip", mime_type="audio/mpeg")
        client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({
            "status": "answered", "message": "", "claims": [{"text": "Claim", "source_ids": [99]}],
        }))
        evidence = [Evidence(1, "Title", "audio", 120, 130, "clip.mp3")]
        with self.assertRaises(RagError):
            Gemini(SETTINGS, client).answer("Question", evidence, [Path("/tmp/test.mp3")])
        parts = client.models.generate_content.call_args.kwargs["contents"][0].parts
        self.assertEqual(parts[-1].file_data.file_uri, "https://files.example/clip")
        self.assertIn('"start_sec": 120', parts[-2].text)
        client.files.delete.assert_called_once_with(name="files/clip")

    def test_files_are_cleaned_when_gemini_generation_fails(self):
        client = Mock()
        client.files.upload.return_value = SimpleNamespace(name="files/clip", state="ACTIVE",
                                                          uri="https://files.example/clip", mime_type="video/mp4")
        client.models.generate_content.side_effect = RuntimeError("private-key")
        with self.assertRaises(RagError) as raised:
            Gemini(SETTINGS, client).answer("Question", [Evidence(1, "Title", "video", 0, 10, "clip.mp4")],
                                          [Path("/tmp/test.mp4")])
        self.assertNotIn("private-key", str(raised.exception))
        client.files.delete.assert_called_once_with(name="files/clip")


class RecoveryTests(unittest.TestCase):
    def asset(self, **extra):
        return {"id": "asset", "lease_token": "lease", "remote_asset_id": "remote", "task_id": "saved-task",
                "duration": 5, "kind": "audio", "has_audio": True,
                "object_key": "original/source.wav", "mime_type": "audio/wav", **extra}

    def test_restart_resumes_task_without_uploading_or_creating_another(self):
        library, provider = Mock(), Mock()
        provider.wait_task.return_value = payload()
        asset = self.asset()
        index_asset(library, asset, provider)
        provider.wait_task.assert_called_once()
        self.assertEqual(provider.wait_task.call_args.args[0], "saved-task")
        provider.upload.assert_not_called()
        provider.create_task.assert_not_called()
        library.database.complete.assert_called_once()

    def test_completed_upload_id_is_saved_before_polling(self):
        library, provider = Mock(), Mock()
        events = []
        library.database.progress.side_effect = lambda *args, **kwargs: events.append(kwargs)
        provider.upload.return_value = "new-remote"
        provider.wait_asset.side_effect = lambda *args: self.assertIn({"remote_asset_id": "new-remote"}, events)
        provider.create_task.return_value = "new-task"
        provider.wait_task.side_effect = lambda *args: (self.assertIn({"task_id": "new-task"}, events), payload())[1]
        index_asset(library, self.asset(remote_asset_id=None, task_id=None), provider)
        library.database.complete.assert_called_once()

    def test_timeout_preserves_remote_ids_and_invalid_task_requests_reset(self):
        library = Mock()
        library.database.claim.return_value = self.asset()
        for error, reset in [(MarengoError("timeout"), False), (MarengoError("failed", reset_task=True), True)]:
            with patch("media_rag.worker.index_asset", side_effect=error):
                run_once(library, Mock())
            self.assertEqual(library.database.fail.call_args.kwargs["reset_task"], reset)

    def test_incomplete_vectors_never_publish_recording(self):
        library, provider = Mock(), Mock()
        provider.wait_task.return_value = {"status": "ready", "data": []}
        with self.assertRaises(RagError):
            index_asset(library, self.asset(), provider)
        library.database.complete.assert_not_called()

    def test_duplicates_do_not_upload_or_create_a_second_job(self):
        database, storage = Mock(), Mock()
        database.find_hash.return_value = {"id": "existing", "status": "ready"}
        uploaded = io.BytesIO(b"same content")
        uploaded.name = "source.mp3"
        asset, created = MediaLibrary(SETTINGS, database, storage).add_upload(uploaded, "Title")
        self.assertEqual(asset["id"], "existing")
        self.assertFalse(created)
        storage.upload.assert_not_called()
        database.create_asset.assert_not_called()

    def test_empty_invalid_and_oversized_uploads_are_rejected_before_storage(self):
        database, storage = Mock(), Mock()
        database.find_hash.return_value = None
        library = MediaLibrary(SETTINGS, database, storage)
        for name, content in [("test.txt", b"a"), ("empty.mp3", b""), ("large.mp3", b"12345")]:
            uploaded = io.BytesIO(content)
            uploaded.name = name
            with patch("media_rag.service.MAX_UPLOAD_BYTES", 4), self.assertRaises(RagError):
                library.add_upload(uploaded, "Title")
        storage.upload.assert_not_called()

    def test_settings_repr_does_not_expose_credentials(self):
        self.assertNotIn("test-twelve-key", repr(SETTINGS))
        self.assertNotIn("test-gemini-key", repr(SETTINGS))


if __name__ == "__main__":
    unittest.main()
