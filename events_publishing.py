"""LinkedIn connection and one-click regional event publishing."""

import copy
import hashlib
import html
import re
import secrets
import threading
import time
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import streamlit as st

import ai_podcast as services
from youtube_publishing import post_fingerprint, validate_callback


REGION_TIMEZONES = {"Bay Area": "America/Los_Angeles", "New York": "America/New_York"}
LINKEDIN_POST_LIMIT = 4000
WORKSPACE_KEYS = (
    "days_to_scrape", "luma_region_results", "cv_region_results", "evion_region_results",
    "meetup_region_results", "region_combined_events", "combined_events", "selected_event_region",
    "organized_events", "organized_events_mode", "organized_events_cache", "organized_events_text",
    "generated_image", "generated_essay", "loaded_events_source", "events_publish_runs",
    "events_linkedin_attempts",
)


@st.cache_resource(show_spinner=False)
def _pending_authorizations():
    # A redirect opens a new Streamlit session. Keep only short-lived pending requests here.
    return {}, threading.Lock()


def _browser_binding():
    cookie = st.context.cookies.get("_streamlit_xsrf", "")
    if not cookie:
        raise ValueError("Reload this page before connecting to LinkedIn.")
    # Tornado can re-mask the same XSRF token when a new WebSocket opens.
    parts = cookie.split("|")
    if len(parts) == 4 and parts[0] == "2":
        mask, token = bytes.fromhex(parts[1]), bytes.fromhex(parts[2])
        if len(mask) != 4 or len(token) != 16:
            raise ValueError("Reload this page before connecting to LinkedIn.")
        cookie = bytes(value ^ mask[index % 4] for index, value in enumerate(token)).hex()
    return hashlib.sha256(cookie.encode()).hexdigest()


def linkedin_config():
    return {
        "client_id": services._optional_config("LINKEDIN_CLIENT_ID"),
        "client_secret": services._optional_config("LINKEDIN_CLIENT_SECRET"),
        "redirect_uri": services._optional_config("EVENTS_LINKEDIN_REDIRECT_URI")
        or services._strip_redirect_uri_extras(services._get_runtime_url()),
    }


def start_authorization(config):
    state = secrets.token_urlsafe(32)
    pending = {
        "state": state, "created": time.time(), "redirect_uri": config["redirect_uri"],
        "browser": _browser_binding(),
        "workspace": copy.deepcopy({key: st.session_state[key] for key in WORKSPACE_KEYS if key in st.session_state}),
    }
    requests, lock = _pending_authorizations()
    with lock:
        expired = [key for key, value in requests.items() if time.time() - value["created"] > 600]
        previous = st.session_state.get("events_oauth_pending", {}).get("state")
        for key in expired + [previous]:
            requests.pop(key, None)
        requests[state] = pending
    st.session_state.events_oauth_pending = {key: pending[key] for key in ("state", "created", "redirect_uri")}
    return services.LINKEDIN_AUTH_URL + "?" + urlencode({
        "response_type": "code", "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"], "scope": "w_member_social openid profile",
        "state": state,
    })


def handle_authorization_callback():
    """Connect automatically when LinkedIn returns, as in the podcast app."""
    if not st.query_params.get("code") and not st.query_params.get("error"):
        return
    try:
        state = st.query_params.get("state", "")
        browser = _browser_binding()
        requests, lock = _pending_authorizations()
        with lock:
            pending = requests.get(state)
            if not pending or time.time() - pending["created"] > 600:
                requests.pop(state, None)
                raise ValueError("Connection expired. Connect to LinkedIn again.")
            if not secrets.compare_digest(browser, pending["browser"]):
                raise ValueError("Connection must finish in the browser where it started.")
            requests.pop(state)
        callback = services._strip_redirect_uri_extras(services._get_runtime_url()) + "?" + urlencode([
            (key, value) for key in ("state", "code", "error") for value in st.query_params.get_all(key)
        ])
        st.session_state.update(pending["workspace"])
        code = validate_callback(callback, pending)
        config = linkedin_config()
        config["redirect_uri"] = pending["redirect_uri"]
        st.session_state.pop("events_oauth_pending", None)
        st.session_state.pop("events_auth_url", None)
        token_data = services.exchange_code_for_token(code, config)
        if not token_data or not token_data.get("access_token"):
            raise ValueError("Token exchange failed")
        token = token_data["access_token"]
        author = services.fetch_authenticated_member_urn(token)
        if not author:
            raise ValueError("Member identity unavailable")
        expires = time.time() + float(token_data.get("expires_in", 3600))
        if expires <= time.time():
            raise ValueError("Token expired")
        st.session_state.events_linkedin = {"token": token, "author": author, "expires": expires}
        st.session_state.pop("events_auth_error", None)
    except Exception:
        st.session_state.events_auth_error = (
            "Could not connect to LinkedIn. Please start a new connection and authorize "
            "in this browser within ten minutes."
        )
    finally:
        st.session_state.pop("events_oauth_pending", None)
        st.session_state.pop("events_auth_url", None)
        for key in ("code", "state", "error", "error_description"):
            st.query_params.pop(key, None)


def disconnect():
    pending = st.session_state.get("events_oauth_pending", {})
    requests, lock = _pending_authorizations()
    with lock:
        requests.pop(pending.get("state"), None)
    for key in ("events_linkedin", "events_oauth_pending", "events_auth_url",
                "events_oauth_callback", "events_auth_error"):
        st.session_state.pop(key, None)


def render_connection():
    account = st.session_state.get("events_linkedin")
    if account and account["expires"] <= time.time():
        disconnect()
        account = None
    if account:
        st.success(f"LinkedIn connected • {account['author']}")
        st.button("Disconnect LinkedIn", key="events_disconnect", on_click=disconnect)
        return account

    config = linkedin_config()
    configured = all(config.values())
    if st.button("Connect to LinkedIn", key="events_connect", disabled=not configured):
        try:
            st.session_state.events_auth_url = start_authorization(config)
            st.session_state.pop("events_auth_error", None)
        except ValueError as exc:
            st.session_state.events_auth_error = str(exc)
    if not configured:
        st.info("Set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in secrets or environment. "
                "Set EVENTS_LINKEDIN_REDIRECT_URI to this Events page URL if needed.")
    if st.session_state.get("events_auth_url"):
        st.caption(f"Register this redirect URL in your LinkedIn app: {config['redirect_uri']}")
        st.html(
            f'<a href="{html.escape(st.session_state.events_auth_url, quote=True)}" '
            'target="_self" style="display:inline-block;padding:0.5rem 0.9rem;'
            'border:1px solid #7fbef2;border-radius:0.5rem;text-decoration:none;">'
            'Authorize on LinkedIn</a>'
        )
        if st.button("Cancel", key="events_cancel_oauth"):
            disconnect()
            st.rerun()
    if st.session_state.get("events_auth_error"):
        st.error(st.session_state.events_auth_error)
    return None


def linkedin_plain_text(organized):
    """Keep the organized agenda and RSVP URLs in LinkedIn's plain-text commentary."""
    lines = []
    for line in organized.splitlines():
        urls = []

        def unwrap_link(match):
            urls.append(match.group(2))
            return match.group(1)

        line = re.sub(r"\[(.+?)\]\((https?://[^\s]+)\)", unwrap_link, line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*]\s+", "• ", line)
        if urls:
            if not line.startswith("• "):
                line = "• " + line
            line += "\nRSVP: " + ", ".join(dict.fromkeys(urls)) + "\n"
        lines.append(line.rstrip())
        if urls:
            lines.append("")
    return "\n".join(lines).strip()


def linkedin_commentary_length(text):
    # Count the encoded API value conservatively, including UTF-16 emoji pairs.
    return len(services.encode_linkedin_plain_text(text).encode("utf-16-le")) // 2


def _prepare_post(run, source, region, prepare_post):
    full_text = linkedin_plain_text(run["organized"])
    run.update(text=full_text, full_text=full_text)
    if prepare_post:
        run.update(prepare_post(source, run["organized"], region))
    if linkedin_commentary_length(run["text"]) > LINKEDIN_POST_LIMIT:
        raise ValueError("The event agenda is too long for one LinkedIn post. Nothing was published.")


def _is_length_rejection(error):
    return bool(re.match(r"^(?:LinkedIn did not confirm publication: )?400:", str(error))) and (
        "text length" in str(error) and "maximum allowed" in str(error)
    )


def run_key(region, days, account):
    day = datetime.now(ZoneInfo(REGION_TIMEZONES[region])).date().isoformat()
    return post_fingerprint(f"{region}|{day}|{days}", None, account["author"])


def run_pipeline(region, days, account, state, scrape, organize, generate_image, progress, prepare_post=None):
    """Use only this run's region content; mark the post attempt before sending it."""
    key = run_key(region, days, account)
    runs = state.setdefault("events_publish_runs", {})
    prior = runs.get(key)
    if prior and prior["status"] in {"published", "uncertain", "running"}:
        return prior
    run = {"region": region, "status": "running", "stage": "scraping"}
    runs[key] = run
    try:
        if account["expires"] <= time.time():
            raise ValueError("LinkedIn connection expired. Reconnect before publishing.")
        source = scrape(region, days, progress)
        if not source or not source.strip():
            raise ValueError(f"No upcoming {region} events found. Nothing was published.")
        run["source"] = source

        run["stage"] = "organizing"
        progress(f"Organizing {region} events…")
        ok, organized = organize(source)
        if not ok or not organized or not organized.strip():
            raise ValueError("Could not organize the scraped events. Nothing was published.")
        run["organized"] = organized
        _prepare_post(run, source, region, prepare_post)

        run["stage"] = "image generation"
        progress(f"Generating the {region} event image…")
        ok, image, error = generate_image(f"{region} AI Events\n\n{run['text']}")
        if not ok or not image or not image.get("bytes"):
            raise ValueError(error or "No image was generated. Nothing was published.")
        image = {**image, "alt_text": f"Upcoming {region} AI events"}
        run["image"] = image
        if account["expires"] <= time.time():
            raise ValueError("LinkedIn connection expired. Reconnect before publishing.")

        fingerprint = post_fingerprint(run["text"], image, account["author"])
        attempts = state.setdefault("events_linkedin_attempts", {})
        run["fingerprint"] = fingerprint
        if attempts.get(fingerprint, {}).get("status") in {"published", "uncertain"}:
            run.update(attempts[fingerprint])
            return run

        run["stage"] = "publishing"
        run["status"] = "uncertain"
        attempts[fingerprint] = {"status": "uncertain"}
        progress(f"Publishing {region} events to LinkedIn…")
        ok, result = services.post_to_linkedin(
            run["text"], account["token"], account["author"],
            image_payload=image, allow_image_fallback=False, plain_text=True,
        )
        if ok:
            run.update(status="published", url=services.build_linkedin_post_url(result.get("id", "")))
        elif _is_length_rejection(result):
            run.update(status="failed", error="LinkedIn rejected the post length. The full agenda is available below.")
        else:
            run["error"] = f"LinkedIn did not confirm publication: {result}"
        attempts[fingerprint] = {field: run[field] for field in ("status", "url", "error") if field in run}
    except Exception as exc:
        if run["status"] == "uncertain":
            run["error"] = "Publishing was interrupted. Check LinkedIn before retrying; the post may already exist."
        else:
            run["status"] = "failed"
            run["error"] = str(exc) if isinstance(exc, ValueError) else (
                f"The {run['stage']} step failed. Nothing was published. Please try again."
            )
    return run


def render_publishing(days, scrape, organize, generate_image, copy_button, prepare_post=None, render_preview=None):
    st.subheader("Publish events to LinkedIn")
    account = render_connection()
    st.caption("Each button scrapes Lu.ma, Cerebral Valley, Evion, and Meetup, organizes upcoming "
               "events starting tomorrow, generates an image, and publishes both publicly to "
               "your connected LinkedIn account. Image generation uses your Google API key.")
    runs = st.session_state.setdefault("events_publish_runs", {})
    for region, column in zip(REGION_TIMEZONES, st.columns(2)):
        with column:
            key = run_key(region, days, account) if account else ""
            run = runs.get(key, {})
            if run.get("status") == "uncertain" and _is_length_rejection(run.get("error", "")):
                st.session_state.get("events_linkedin_attempts", {}).pop(run.get("fingerprint"), None)
                run.update(status="failed", error="LinkedIn rejected the full agenda. Publish again to create one concise post.")
                source = run.get("source") or (st.session_state.get("region_combined_events") or {}).get(region)
                if source and run.get("organized"):
                    try:
                        _prepare_post(run, source, region, prepare_post)
                    except ValueError:
                        pass
            if run.get("status") == "running":
                run.update(status="failed", error="Preparation was interrupted. You can try again.")
            blocked = run.get("status") in {"published", "uncertain"}
            if st.button(f"Publish {region} events", key=f"events_publish_{region}",
                         type="primary", width="stretch", disabled=not account or blocked):
                with st.status(f"Preparing {region} events…", expanded=True) as status:
                    result = run_pipeline(
                        region, days, account, st.session_state, scrape, organize, generate_image,
                        lambda label: status.update(label=label),
                        prepare_post=prepare_post,
                    )
                    status.update(state="complete" if result["status"] == "published" else "error")
                st.rerun()
            if run.get("error"):
                st.error(run["error"])
            if run.get("status") == "published":
                st.success(f"{region} events published!")
                if run.get("url"):
                    st.link_button(f"View {region} LinkedIn post", run["url"])
            if run.get("status") == "uncertain":
                st.warning("Check your LinkedIn profile before retrying to avoid a duplicate.")
                if st.button(f"No {region} post exists — allow retry", key=f"events_retry_{region}"):
                    st.session_state.get("events_linkedin_attempts", {}).pop(run.get("fingerprint"), None)
                    runs.pop(key, None)
                    st.rerun()
            if run.get("organized"):
                copy_button(run["text"], f"publish-{region.lower().replace(' ', '-')}",
                            label="📋 Copy for LinkedIn", html_content=run.get("html"))
                with st.expander(f"{region} organized events and image"):
                    if render_preview and run.get("html"):
                        render_preview(run["html"])
                    else:
                        st.text(run["text"])
                    if run.get("image"):
                        st.image(run["image"]["bytes"], caption=f"{region} event image", width="stretch")
                if run.get("full_text") and run["full_text"] != run["text"]:
                    st.caption("Condensed to one LinkedIn post. The full agenda is preserved below.")
                    with st.expander(f"Full {region} agenda"):
                        copy_button(run["full_text"], f"full-agenda-{region.lower().replace(' ', '-')}",
                                    label="📋 Copy full agenda", html_content=run.get("full_html"))
                        if render_preview and run.get("full_html"):
                            render_preview(run["full_html"])
                        else:
                            st.text(run["full_text"])
