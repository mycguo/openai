"""Offline tests: no scraping, billable generation, OAuth traffic, or public posts."""

import base64
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlencode, urlsplit

from streamlit.testing.v1 import AppTest

with patch("streamlit.runtime.secrets.Secrets.get", return_value=None):
    import events
    import events_publishing as publishing


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
ACCOUNT = {"token": "test-token", "author": "urn:li:person:test", "expires": time.time() + 3600}
CONFIG = {"client_id": "test-client", "client_secret": "test-secret", "redirect_uri": "http://localhost:8501/"}
SF = """- **[SF AI meetup](https://lu.ma/sf-test)**
  Date and Time: October 20, 2026 6:00 PM PDT
  Location/Venue: San Francisco
  Host: SF host
  Brief Description: SF demos.
"""
NY = """- **[NY AI meetup](https://lu.ma/ny-test)**
  Date and Time: October 20, 2026 6:00 PM EDT
  Location/Venue: New York
  Host: NY host
  Brief Description: NY demos.
"""


def element(elements, label):
    return next(item for item in elements if item.label == label)


class State(dict):
    """Match Streamlit's mapping and attribute access for offline source tests."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def scraper(state, state_key, content):
    def scrape(region, days):
        state.setdefault(state_key, {})[region] = {"formatted": content, "has_events": True}
        events._refresh_region_combined_state()
        return True
    return Mock(side_effect=scrape)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        self.scrape = Mock(side_effect=lambda region, days, progress: f"{region} fresh source")
        self.organize = Mock(side_effect=lambda source: (True, f"**Agenda**\n{source}\n[RSVP](https://lu.ma/test)"))
        self.image = Mock(return_value=(True, {"bytes": PNG, "mime_type": "image/png"}, None))
        self.publish_patch = patch.object(publishing.services, "post_to_linkedin", return_value=(True, {"id": "urn:li:share:123"}))
        self.publish = self.publish_patch.start()
        self.addCleanup(self.publish_patch.stop)

    def run_region(self, region="Bay Area", account=None):
        return publishing.run_pipeline(region, 8, account or ACCOUNT, self.state,
                                       self.scrape, self.organize, self.image, Mock())

    def test_full_workflow_uses_fresh_region_agenda_and_image(self):
        for region in ("Bay Area", "New York"):
            result = self.run_region(region)
            self.assertEqual(result["status"], "published")
            self.scrape.assert_called_with(region, 8, unittest.mock.ANY)
            self.organize.assert_called_with(f"{region} fresh source")
            self.assertIn(region, self.image.call_args.args[0])
            self.assertIn(result["organized"], self.image.call_args.args[0])
            self.publish.assert_called_with(
                result["text"], ACCOUNT["token"], ACCOUNT["author"],
                image_payload=result["image"], allow_image_fallback=False,
            )
            self.assertNotIn("**", result["text"])
            self.assertIn("RSVP: https://lu.ma/test", result["text"])
        self.assertEqual(len(self.state["events_publish_runs"]), 2)

    def test_success_is_not_repeated(self):
        first = self.run_region()
        self.assertIs(self.run_region(), first)
        self.scrape.assert_called_once()
        self.publish.assert_called_once()

    def test_identical_content_is_not_posted_again_for_another_window(self):
        self.run_region()
        result = publishing.run_pipeline("Bay Area", 9, ACCOUNT, self.state,
                                         self.scrape, self.organize, self.image, Mock())
        self.assertEqual(result["status"], "published")
        self.publish.assert_called_once()

    def test_no_events_does_not_generate_or_publish(self):
        self.scrape.return_value = ""
        self.scrape.side_effect = None
        self.assertEqual(self.run_region()["status"], "failed")
        self.organize.assert_not_called()
        self.image.assert_not_called()
        self.publish.assert_not_called()

    def test_organization_failure_stops_before_image(self):
        self.organize.return_value = (False, None)
        self.organize.side_effect = None
        self.assertEqual(self.run_region()["status"], "failed")
        self.image.assert_not_called()
        self.publish.assert_not_called()

    def test_image_failure_stops_publication_and_allows_retry(self):
        self.image.return_value = (False, None, "Image provider unavailable")
        self.assertEqual(self.run_region()["status"], "failed")
        self.publish.assert_not_called()
        self.image.return_value = (True, {"bytes": PNG}, None)
        self.assertEqual(self.run_region()["status"], "published")
        self.publish.assert_called_once()

    def test_empty_image_is_not_a_text_only_post(self):
        self.image.return_value = (True, {"bytes": b""}, None)
        self.assertEqual(self.run_region()["status"], "failed")
        self.publish.assert_not_called()

    def test_expired_connection_does_not_start(self):
        self.assertEqual(self.run_region(account={**ACCOUNT, "expires": 1})["status"], "failed")
        self.scrape.assert_not_called()
        self.publish.assert_not_called()

    def test_expiry_during_preparation_does_not_publish(self):
        account = dict(ACCOUNT)
        def generate(_):
            account["expires"] = 1
            return True, {"bytes": PNG}, None
        self.image.side_effect = generate
        self.assertEqual(self.run_region(account=account)["status"], "failed")
        self.publish.assert_not_called()

    def test_ambiguous_create_is_not_automatically_retried(self):
        for response in (TimeoutError(), (False, "503: unavailable")):
            with self.subTest(response=type(response).__name__):
                self.state.clear()
                self.publish.reset_mock(side_effect=True)
                if isinstance(response, Exception):
                    self.publish.side_effect = response
                else:
                    self.publish.return_value = response
                self.assertEqual(self.run_region()["status"], "uncertain")
                self.assertEqual(self.run_region()["status"], "uncertain")
                self.publish.assert_called_once()

    def test_image_upload_failure_never_creates_public_post(self):
        self.publish_patch.stop()
        with patch.object(publishing.services, "initialize_linkedin_image_upload", return_value=(False, None, None, "upload failed")), \
             patch.object(publishing.services.requests, "post") as create:
            self.assertEqual(self.run_region()["status"], "uncertain")
            create.assert_not_called()

    def test_long_agenda_is_sent_unchanged_except_markdown(self):
        self.organize.side_effect = None
        self.organize.return_value = (True, "x" * 5000)
        result = self.run_region()
        self.assertEqual(result["status"], "published")
        self.assertEqual(self.publish.call_args.args[0], "x" * 5000)


class RegionalScrapeTests(unittest.TestCase):
    def test_all_sources_are_scraped_and_old_or_other_region_results_are_excluded(self):
        state = State({"luma_region_results": {
            "Bay Area": {"formatted": "stale Bay Area", "has_events": True},
            "New York": {"formatted": NY, "has_events": True},
        }})
        with patch.object(events.st, "session_state", state):
            with patch.object(events, "_scrape_luma_region_to_state", scraper(state, "luma_region_results", SF + NY)) as luma, \
                 patch.object(events, "_scrape_cerebral_valley_region_to_state", scraper(state, "cv_region_results", SF)) as cv, \
                 patch.object(events, "_scrape_evion_region_to_state", scraper(state, "evion_region_results", SF)) as evion, \
                 patch.object(events, "_scrape_meetup_region_to_state", scraper(state, "meetup_region_results", SF)) as meetup:
                source = events._scrape_region_for_publishing("Bay Area", 8, Mock())
                for mock in (luma, cv, evion, meetup):
                    mock.assert_called_once_with("Bay Area", 8)
            self.assertNotIn("stale", source)
            self.assertNotIn("NY AI meetup", source)
            self.assertEqual(source.count("SF AI meetup"), 1)
            self.assertIn("New York", state["luma_region_results"])
            organized_ok, organized = events.generate_organized_events(source)
            self.assertTrue(organized_ok)
            self.assertIn("SF AI meetup", organized)

    def test_failed_sources_do_not_reuse_stale_content(self):
        state = State({"luma_region_results": {"Bay Area": {"formatted": SF, "has_events": True}}})
        with patch.object(events.st, "session_state", state), \
             patch.object(events.st, "warning"), \
             patch.object(events, "_scrape_luma_region_to_state", side_effect=RuntimeError), \
             patch.object(events, "_scrape_cerebral_valley_region_to_state", return_value=False), \
             patch.object(events, "_scrape_evion_region_to_state", return_value=False), \
             patch.object(events, "_scrape_meetup_region_to_state", return_value=False):
            self.assertEqual(events._scrape_region_for_publishing("Bay Area", 8, Mock()), "")

    def test_image_prompt_contains_actual_region_agenda(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(parts=[
            SimpleNamespace(inline_data=SimpleNamespace(data=PNG, mime_type="image/png")),
        ])
        with patch.object(events, "GOOGLE_API_KEY", "test"), patch.object(events.genai, "Client", return_value=client):
            ok, _, _ = events.generate_event_image("New York AI Events\nNY AI meetup")
        self.assertTrue(ok)
        self.assertIn("New York AI Events\nNY AI meetup", client.models.generate_content.call_args.kwargs["contents"])


class PublishingUITests(unittest.TestCase):
    def app(self, connected=False):
        app = AppTest.from_string("import events\nevents.main()")
        if connected:
            app.session_state.events_linkedin = dict(ACCOUNT)
        return app.run()

    def test_disconnected_page_has_connection_and_two_disabled_publish_buttons(self):
        with patch.object(publishing, "linkedin_config", return_value=CONFIG), \
             patch.object(publishing.services, "post_to_linkedin") as post:
            app = self.app()
            self.assertFalse(app.exception)
            self.assertFalse(element(app.button, "Connect to LinkedIn").disabled)
            for region in ("Bay Area", "New York"):
                self.assertTrue(element(app.button, f"Publish {region} events").disabled)
            post.assert_not_called()

    def test_connection_state_validation_exchange_once_and_disconnect(self):
        with patch.object(publishing, "linkedin_config", return_value=CONFIG), \
             patch.object(publishing.services, "exchange_code_for_token", return_value={"access_token": "test-token", "expires_in": 3600}) as exchange, \
             patch.object(publishing.services, "fetch_authenticated_member_urn", return_value=ACCOUNT["author"]):
            app = self.app()
            element(app.button, "Connect to LinkedIn").click().run()
            pending = app.session_state.events_oauth_pending
            params = parse_qs(urlsplit(app.session_state.events_auth_url).query)
            self.assertEqual(params["scope"], ["w_member_social openid profile"])
            bad = CONFIG["redirect_uri"] + "?" + urlencode({"state": "wrong", "code": "code"})
            element(app.text_input, "Returned callback URL").set_value(bad)
            element(app.button, "Complete connection").click().run()
            exchange.assert_not_called()
            callback = CONFIG["redirect_uri"] + "?" + urlencode({"state": pending["state"], "code": "test-code"})
            element(app.text_input, "Returned callback URL").set_value(callback)
            element(app.button, "Complete connection").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.events_linkedin["author"], ACCOUNT["author"])
            self.assertNotIn("events_oauth_pending", app.session_state)
            self.assertNotIn("events_oauth_callback", app.session_state)
            app.run()
            exchange.assert_called_once()
            element(app.button, "Disconnect LinkedIn").click().run()
            self.assertNotIn("events_linkedin", app.session_state)
            self.assertTrue(element(app.button, "Publish Bay Area events").disabled)

    def test_publish_buttons_run_all_steps_and_keep_regions_separate(self):
        def store(key):
            def scrape(region, days):
                events.st.session_state.setdefault(key, {})[region] = {
                    "formatted": SF if region == "Bay Area" else NY, "has_events": True,
                }
                events._refresh_region_combined_state()
                return True
            return scrape
        with patch.object(events, "_scrape_luma_region_to_state", side_effect=store("luma_region_results")) as luma, \
             patch.object(events, "_scrape_cerebral_valley_region_to_state", side_effect=store("cv_region_results")) as cv, \
             patch.object(events, "_scrape_evion_region_to_state", side_effect=store("evion_region_results")) as evion, \
             patch.object(events, "_scrape_meetup_region_to_state", side_effect=store("meetup_region_results")) as meetup, \
             patch.object(events, "generate_event_image", return_value=(True, {"bytes": PNG}, None)) as image, \
             patch.object(publishing.services, "post_to_linkedin", return_value=(True, {"id": "urn:li:share:test"})) as post:
            app = self.app(connected=True)
            for region, own, other in (("Bay Area", "SF AI meetup", "NY AI meetup"), ("New York", "NY AI meetup", "SF AI meetup")):
                element(app.button, f"Publish {region} events").click().run()
                self.assertFalse(app.exception)
                for mock in (luma, cv, evion, meetup):
                    mock.assert_called_with(region, 8)
                self.assertIn(own, post.call_args.args[0])
                self.assertNotIn(other, post.call_args.args[0])
                self.assertIn(own, image.call_args.args[0])
                self.assertNotIn(other, image.call_args.args[0])
                self.assertTrue(element(app.button, f"Publish {region} events").disabled)
            app.run()
            self.assertEqual(post.call_count, 2)

    def test_expired_connection_is_cleared(self):
        with patch.object(publishing, "linkedin_config", return_value=CONFIG):
            app = self.app(connected=True)
            app.session_state.events_linkedin = {**ACCOUNT, "expires": 1}
            app.run()
            self.assertNotIn("events_linkedin", app.session_state)
            self.assertTrue(element(app.button, "Publish New York events").disabled)

    def test_missing_configuration_disables_connection(self):
        with patch.object(publishing, "linkedin_config", return_value={field: "" for field in CONFIG}):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(element(app.button, "Connect to LinkedIn").disabled)

    def test_timeout_requires_explicit_retry_and_reruns_do_not_publish(self):
        with patch.object(events, "_scrape_region_for_publishing", return_value=SF), \
             patch.object(events, "generate_event_image", return_value=(True, {"bytes": PNG}, None)), \
             patch.object(publishing.services, "post_to_linkedin", side_effect=TimeoutError) as post:
            app = self.app(connected=True)
            element(app.button, "Publish Bay Area events").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(element(app.button, "Publish Bay Area events").disabled)
            app.run()
            post.assert_called_once()
            element(app.button, "No Bay Area post exists — allow retry").click().run()
            self.assertFalse(element(app.button, "Publish Bay Area events").disabled)
            post.assert_called_once()
            post.side_effect = None
            post.return_value = True, {"id": "urn:li:share:retry"}
            element(app.button, "Publish Bay Area events").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(post.call_count, 2)
            self.assertTrue(element(app.button, "Publish Bay Area events").disabled)


if __name__ == "__main__":
    unittest.main()
