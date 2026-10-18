"""Offline tests: no scraping, billable generation, OAuth traffic, or public posts."""

import base64
import re
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


def long_agenda_source(count=110, title_suffix=""):
    return "\n".join(
        f"""- **[AI meetup {index:03d} {title_suffix}](https://lu.ma/event-{index:03d})**
  Date and Time: October 20, 2026 11:30 AM EDT
  Location/Venue: New York, US
  Host: NYC AI community and local builders
  Brief Description: {'Meet local AI builders and explore practical applications. ' * 3}
"""
        for index in range(count)
    )


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

    def run_region(self, region="Bay Area", account=None, prepare_post=None):
        return publishing.run_pipeline(region, 8, account or ACCOUNT, self.state,
                                       self.scrape, self.organize, self.image, Mock(), prepare_post=prepare_post)

    def test_full_workflow_uses_fresh_region_agenda_and_image(self):
        for region in ("Bay Area", "New York"):
            result = self.run_region(region)
            self.assertEqual(result["status"], "published")
            self.scrape.assert_called_with(region, 8, unittest.mock.ANY)
            self.organize.assert_called_with(f"{region} fresh source")
            self.assertIn(region, self.image.call_args.args[0])
            self.assertIn(result["text"], self.image.call_args.args[0])
            self.publish.assert_called_with(
                result["text"], ACCOUNT["token"], ACCOUNT["author"],
                image_payload=result["image"], allow_image_fallback=False, plain_text=True,
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

    def test_unstructured_oversized_agenda_stops_before_generation_or_publication(self):
        self.organize.side_effect = None
        self.organize.return_value = (True, "x" * 5000)
        result = self.run_region()
        self.assertEqual(result["status"], "failed")
        self.image.assert_not_called()
        self.publish.assert_not_called()

    def test_long_agenda_creates_one_concise_post_and_preserves_the_full_agenda(self):
        source = long_agenda_source(130)
        self.scrape.side_effect = None
        self.scrape.return_value = source
        self.organize.side_effect = None
        self.organize.return_value = True, events._build_organized_events_fallback(
            events._extract_organize_candidate_events(source),
        )
        result = self.run_region("New York", prepare_post=events._prepare_region_linkedin_post)
        self.assertEqual(result["status"], "published")
        self.assertGreater(len(result["full_text"]), 22051)
        self.assertLessEqual(publishing.linkedin_commentary_length(result["text"]), 4000)
        self.assertIn("AI meetup 129", result["full_text"])
        self.assertNotIn("AI meetup 129", result["text"])
        self.assertIn(result["text"], self.image.call_args.args[0])
        self.assertNotIn("AI meetup 129", self.image.call_args.args[0])
        self.assertIn("<ul>", result["html"])
        self.assertIn('href="https://lu.ma/event-000"', result["html"])
        self.publish.assert_called_once()

    def test_definitive_length_rejection_allows_an_explicit_retry(self):
        self.publish.return_value = False, '400: {"message":"ShareCommentary text length exceeded the maximum allowed"}'
        self.assertEqual(self.run_region()["status"], "failed")
        self.publish.return_value = True, {"id": "urn:li:share:retry"}
        self.assertEqual(self.run_region()["status"], "published")
        self.assertEqual(self.publish.call_count, 2)

    def test_unfittable_url_preserves_a_copyable_agenda_without_generation(self):
        url = "https://lu.ma/" + "a" * 4100
        source = SF.replace("https://lu.ma/sf-test", url)
        self.scrape.side_effect = None
        self.scrape.return_value = source
        self.organize.side_effect = None
        self.organize.return_value = True, events._build_organized_events_fallback(
            events._extract_organize_candidate_events(source),
        )
        result = self.run_region(prepare_post=events._prepare_region_linkedin_post)
        self.assertEqual(result["status"], "failed")
        self.assertIn(url, result["text"])
        self.image.assert_not_called()
        self.publish.assert_not_called()


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


class BrowserBindingTests(unittest.TestCase):
    def test_new_websocket_mask_keeps_the_same_browser_binding(self):
        bindings = []
        token = bytes(range(16))
        for mask in (b"abcd", b"wxyz"):
            masked = bytes(value ^ mask[index % 4] for index, value in enumerate(token))
            cookie = f"2|{mask.hex()}|{masked.hex()}|1000"
            with patch.object(publishing.st, "context", SimpleNamespace(cookies={"_streamlit_xsrf": cookie})):
                bindings.append(publishing._browser_binding())
        self.assertEqual(bindings[0], bindings[1])
        with patch.object(publishing.st, "context", SimpleNamespace(cookies={})):
            with self.assertRaises(ValueError):
                publishing._browser_binding()


class LinkedInFormattingTests(unittest.TestCase):
    def test_published_agenda_matches_existing_organize_copy_format(self):
        candidates = events._extract_organize_candidate_events(SF + NY)
        organized = events._build_organized_events_fallback(candidates)
        expected = events._build_organized_events_text(
            candidates, use_markdown_links=False, include_plain_rsvp_link=True,
        )
        actual = publishing.linkedin_plain_text(organized)
        self.assertEqual(actual, expected)
        self.assertIn("RSVP: https://lu.ma/sf-test", actual)
        self.assertIn("RSVP: https://lu.ma/ny-test", actual)
        self.assertNotIn("](", actual)

    def test_markdown_titles_and_punctuation_become_readable_plain_text(self):
        organized = "## Tuesday, October 20\n\n- **6 PM - [AI [Agents] & RAG](https://lu.ma/ai?ref=feed#details)** - SF host\n"
        self.assertEqual(
            publishing.linkedin_plain_text(organized),
            "Tuesday, October 20\n\n• 6 PM - AI [Agents] & RAG - SF host\nRSVP: https://lu.ma/ai?ref=feed#details",
        )

    def test_event_blocks_and_links_do_not_run_into_the_next_time(self):
        candidates = events._extract_organize_candidate_events(SF + NY)
        text = events._build_organized_events_text(candidates, use_markdown_links=False, include_plain_rsvp_link=True)
        self.assertIn("RSVP: https://lu.ma/ny-test\n\n• ", text)
        self.assertNotIn("https://lu.ma/ny-test\n6", text)
        self.assertTrue(all(line.startswith("https://") for line in re.findall(r"^RSVP: (.+)$", text, re.M)))

    def test_concise_post_keeps_complete_events_and_urls(self):
        source = long_agenda_source()
        prepared = events._prepare_region_linkedin_post(source, "", "New York")
        urls = re.findall(r"^RSVP: (https?://\S+)$", prepared["text"], re.M)
        self.assertTrue(urls)
        self.assertEqual(urls, [f"https://lu.ma/event-{index:03d}" for index in range(len(urls))])
        self.assertIn(f"Showing {len(urls)} of 110 upcoming New York events.", prepared["text"])
        self.assertEqual(prepared["html"].count("<li>"), len(urls))
        self.assertLessEqual(publishing.linkedin_commentary_length(prepared["text"]), 4000)

    def test_compaction_keeps_all_events_when_details_alone_exceed_the_limit(self):
        prepared = events._prepare_region_linkedin_post(long_agenda_source(30), "", "New York")
        self.assertGreater(publishing.linkedin_commentary_length(prepared["full_text"]), 4000)
        self.assertEqual(prepared["text"].count("RSVP:"), 30)
        self.assertNotIn("Showing", prepared["text"])
        self.assertLessEqual(publishing.linkedin_commentary_length(prepared["text"]), 4000)

    def test_concise_post_counts_api_escapes_and_emoji_pairs(self):
        source = long_agenda_source(30, "🚀 [Agents] #AI " * 10)
        prepared = events._prepare_region_linkedin_post(source, "", "New York")
        self.assertLessEqual(publishing.linkedin_commentary_length(prepared["text"]), 4000)
        self.assertIn("🚀", prepared["text"])
        self.assertIn("RSVP: https://lu.ma/event-000", prepared["text"])

    def test_short_post_uses_exact_organized_events_html(self):
        prepared = events._prepare_region_linkedin_post(SF, "", "Bay Area")
        expected = events._build_organized_events_html(events._extract_organize_candidate_events(SF))
        self.assertEqual(prepared["html"], expected)
        self.assertEqual(prepared["text"], prepared["full_text"])

    def test_an_oversized_single_url_is_never_truncated(self):
        source = SF.replace("https://lu.ma/sf-test", "https://lu.ma/" + "a" * 4100)
        with self.assertRaisesRegex(ValueError, "complete RSVP link"):
            events._prepare_region_linkedin_post(source, "", "Bay Area")

    def test_plain_text_api_encoding_preserves_reserved_punctuation(self):
        text = "AI [Agents] (RAG) | @host #AI <demo> {tools} \\ * _ ~\nRSVP: https://lu.ma/ai_event#details"
        response = Mock(status_code=201, content=b"", headers={})
        response.raise_for_status.return_value = None
        with patch.object(publishing.services.requests, "post", return_value=response) as create:
            ok, _ = publishing.services.post_to_linkedin(text, "test-token", ACCOUNT["author"], plain_text=True)
        self.assertTrue(ok)
        sent = create.call_args.kwargs["json"]["commentary"]
        self.assertIn(r"\[Agents\] \(RAG\) \| \@host \#AI", sent)
        self.assertIn(r"ai\_event\#details", sent)
        self.assertIn("\nRSVP:", sent)
        # LinkedIn consumes one escape character, leaving the same text as Copy.
        self.assertEqual(re.sub(r"\\([|{}@\[\]()<>#\\*_~])", r"\1", sent), text)


class PublishingUITests(unittest.TestCase):
    def setUp(self):
        publishing._pending_authorizations.clear()
        self.enterContext(patch.object(publishing, "_browser_binding", return_value="test-browser"))
        self.enterContext(patch.object(publishing.services, "_get_runtime_url", return_value=CONFIG["redirect_uri"]))

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

    def test_automatic_connection_restores_workspace_after_redirect_and_exchanges_once(self):
        with patch.object(publishing, "linkedin_config", return_value=CONFIG), \
             patch.object(publishing.services, "exchange_code_for_token", return_value={"access_token": "test-token", "expires_in": 3600}) as exchange, \
             patch.object(publishing.services, "fetch_authenticated_member_urn", return_value=ACCOUNT["author"]):
            app = self.app()
            app.session_state.combined_events = SF
            app.session_state.generated_image = {"bytes": PNG}
            app.session_state.days_to_scrape = 12
            app.session_state.events_linkedin_attempts = {"previous": {"status": "published"}}
            app.run()
            element(app.button, "Connect to LinkedIn").click().run()
            pending = app.session_state.events_oauth_pending
            params = parse_qs(urlsplit(app.session_state.events_auth_url).query)
            self.assertEqual(params["scope"], ["w_member_social openid profile"])
            exchange.assert_not_called()
            self.assertFalse(any(item.label == "Returned callback URL" for item in app.text_input))
            self.assertFalse(any(item.label == "Complete connection" for item in app.button))
            # A real provider redirect starts a new Streamlit session.
            app = AppTest.from_string("import events\nevents.main()")
            app.query_params.update({"state": pending["state"], "code": "test-code"})
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.events_linkedin["author"], ACCOUNT["author"])
            self.assertEqual(app.session_state.combined_events, SF)
            self.assertEqual(app.session_state.generated_image["bytes"], PNG)
            self.assertEqual(app.session_state.days_to_scrape, 12)
            self.assertIn("previous", app.session_state.events_linkedin_attempts)
            self.assertNotIn("events_oauth_pending", app.session_state)
            self.assertNotIn("code", app.query_params)
            self.assertNotIn("state", app.query_params)
            app.run()
            exchange.assert_called_once()
            element(app.button, "Disconnect LinkedIn").click().run()
            self.assertNotIn("events_linkedin", app.session_state)
            self.assertTrue(element(app.button, "Publish Bay Area events").disabled)

    def test_automatic_connection_rejects_unknown_expired_and_other_browser_requests(self):
        with patch.object(publishing, "linkedin_config", return_value=CONFIG), \
             patch.object(publishing.services, "exchange_code_for_token") as exchange:
            for invalid in ("unknown", "expired", "other browser", "wrong redirect"):
                with self.subTest(invalid=invalid):
                    app = self.app()
                    element(app.button, "Connect to LinkedIn").click().run()
                    state = app.session_state.events_oauth_pending["state"]
                    requests, _ = publishing._pending_authorizations()
                    if invalid == "unknown":
                        state = "unknown"
                    elif invalid == "expired":
                        requests[state]["created"] -= 601
                    elif invalid == "other browser":
                        requests[state]["browser"] = "other-browser"
                    else:
                        requests[state]["redirect_uri"] = "http://localhost:8502/"
                    callback = AppTest.from_string("import events\nevents.main()")
                    callback.query_params.update({"state": state, "code": "test-code"})
                    callback.run()
                    self.assertFalse(callback.exception)
                    self.assertNotIn("events_linkedin", callback.session_state)
                    self.assertTrue(callback.error)
                    self.assertNotIn("code", callback.query_params)
                    exchange.assert_not_called()

    def test_authorization_denied_restores_workspace_without_connecting(self):
        with patch.object(publishing, "linkedin_config", return_value=CONFIG), \
             patch.object(publishing.services, "exchange_code_for_token") as exchange:
            app = self.app()
            app.session_state.combined_events = NY
            element(app.button, "Connect to LinkedIn").click().run()
            state = app.session_state.events_oauth_pending["state"]
            callback = AppTest.from_string("import events\nevents.main()")
            callback.query_params.update({"state": state, "error": "access_denied"})
            callback.run()
            self.assertFalse(callback.exception)
            self.assertEqual(callback.session_state.combined_events, NY)
            self.assertNotIn("events_linkedin", callback.session_state)
            self.assertNotIn("error", callback.query_params)
            exchange.assert_not_called()

    def test_cancel_invalidates_pending_connection(self):
        with patch.object(publishing, "linkedin_config", return_value=CONFIG):
            app = self.app()
            element(app.button, "Connect to LinkedIn").click().run()
            state = app.session_state.events_oauth_pending["state"]
            element(app.button, "Cancel").click().run()
            requests, _ = publishing._pending_authorizations()
            self.assertNotIn(state, requests)
            self.assertNotIn("events_auth_url", app.session_state)

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
             patch.object(events, "render_copy_button", wraps=events.render_copy_button) as copy_button, \
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
                copy_button.assert_any_call(
                    post.call_args.args[0], f"publish-{region.lower().replace(' ', '-')}", label="📋 Copy for LinkedIn",
                    html_content=events._build_organized_events_html(
                        events._extract_organize_candidate_events(SF if region == "Bay Area" else NY),
                    ),
                )
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

    def test_existing_oversized_rejection_can_be_retried_and_full_agenda_can_be_copied(self):
        source = long_agenda_source()
        organized = events._build_organized_events_fallback(events._extract_organize_candidate_events(source))
        with patch.object(events, "_scrape_region_for_publishing") as scrape, \
             patch.object(events, "render_copy_button", wraps=events.render_copy_button) as copy_button, \
             patch.object(publishing.services, "post_to_linkedin") as post:
            app = self.app(connected=True)
            app.session_state.events_publish_runs = {
                publishing.run_key("New York", 8, ACCOUNT): {
                    "region": "New York", "source": source, "organized": organized,
                    "text": publishing.linkedin_plain_text(organized), "status": "uncertain",
                    "fingerprint": "legacy-rejected",
                    "error": 'LinkedIn did not confirm publication: 400: {"message":"ShareCommentary text length (22051 characters) exceeded the maximum allowed (4000 characters)"}',
                },
            }
            app.session_state.events_linkedin_attempts = {"legacy-rejected": {"status": "uncertain"}}
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(element(app.button, "Publish New York events").disabled)
            self.assertNotIn("legacy-rejected", app.session_state.events_linkedin_attempts)
            run = app.session_state.events_publish_runs[publishing.run_key("New York", 8, ACCOUNT)]
            self.assertLessEqual(publishing.linkedin_commentary_length(run["text"]), 4000)
            copy_button.assert_any_call(run["full_text"], "full-agenda-new-york",
                                        label="📋 Copy full agenda", html_content=run["full_html"])
            scrape.assert_not_called()
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
