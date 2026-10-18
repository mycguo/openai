"""LinkedIn connection and one-click regional event publishing."""

import re
import secrets
import time
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import streamlit as st

import ai_podcast as services
from youtube_publishing import post_fingerprint, validate_callback


REGION_TIMEZONES = {"Bay Area": "America/Los_Angeles", "New York": "America/New_York"}


def linkedin_config():
    return {
        "client_id": services._optional_config("LINKEDIN_CLIENT_ID"),
        "client_secret": services._optional_config("LINKEDIN_CLIENT_SECRET"),
        "redirect_uri": services._optional_config("EVENTS_LINKEDIN_REDIRECT_URI")
        or services._strip_redirect_uri_extras(services._get_runtime_url()),
    }


def start_authorization(config):
    state = secrets.token_urlsafe(32)
    st.session_state.events_oauth_pending = {
        "state": state, "created": time.time(), "redirect_uri": config["redirect_uri"],
    }
    return services.LINKEDIN_AUTH_URL + "?" + urlencode({
        "response_type": "code", "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"], "scope": "w_member_social openid profile",
        "state": state,
    })


def finish_authorization():
    callback = st.session_state.pop("events_oauth_callback", "")
    pending = st.session_state.get("events_oauth_pending")
    try:
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
            "Could not connect. Check the returned URL and the ten-minute connection expiry, "
            "or start a new connection."
        )


def disconnect():
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
        st.session_state.events_auth_url = start_authorization(config)
    if not configured:
        st.info("Set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET in secrets or environment. "
                "Set EVENTS_LINKEDIN_REDIRECT_URI to this Events page URL if needed.")
    if st.session_state.get("events_auth_url"):
        st.caption(f"Register this redirect URL in your LinkedIn app: {config['redirect_uri']}")
        st.link_button("Authorize on LinkedIn (new tab)", st.session_state.events_auth_url)
        st.write("After authorizing, copy the returned tab's full address and paste it here "
                 "in this original tab to preserve your workspace.")
        with st.form("events_connect_form"):
            st.text_input("Returned callback URL", type="password", key="events_oauth_callback")
            st.form_submit_button("Complete connection", on_click=finish_authorization)
    if st.session_state.get("events_auth_error"):
        st.error(st.session_state.events_auth_error)
    return None


def linkedin_plain_text(organized):
    """Keep the organized agenda and RSVP URLs in LinkedIn's plain-text commentary."""
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s]+)\)", r"\1 - RSVP: \2", organized)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return re.sub(r"(?m)^#{1,6}\s+", "", text).strip()


def run_key(region, days, account):
    day = datetime.now(ZoneInfo(REGION_TIMEZONES[region])).date().isoformat()
    return post_fingerprint(f"{region}|{day}|{days}", None, account["author"])


def run_pipeline(region, days, account, state, scrape, organize, generate_image, progress):
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

        run["stage"] = "organizing"
        progress(f"Organizing {region} events…")
        ok, organized = organize(source)
        if not ok or not organized or not organized.strip():
            raise ValueError("Could not organize the scraped events. Nothing was published.")
        run["organized"] = organized
        run["text"] = linkedin_plain_text(organized)

        run["stage"] = "image generation"
        progress(f"Generating the {region} event image…")
        ok, image, error = generate_image(f"{region} AI Events\n\n{organized}")
        if not ok or not image or not image.get("bytes"):
            raise ValueError(error or "No image was generated. Nothing was published.")
        image = {**image, "alt_text": f"Upcoming {region} AI events"}
        run["image"] = image
        if account["expires"] <= time.time():
            raise ValueError("LinkedIn connection expired. Reconnect before publishing.")

        fingerprint = post_fingerprint(run["text"], image, account["author"])
        attempts = state.setdefault("events_linkedin_attempts", {})
        run["fingerprint"] = fingerprint
        if fingerprint in attempts:
            run.update(attempts[fingerprint])
            return run

        run["stage"] = "publishing"
        run["status"] = "uncertain"
        attempts[fingerprint] = {"status": "uncertain"}
        progress(f"Publishing {region} events to LinkedIn…")
        ok, result = services.post_to_linkedin(
            run["text"], account["token"], account["author"],
            image_payload=image, allow_image_fallback=False,
        )
        if ok:
            run.update(status="published", url=services.build_linkedin_post_url(result.get("id", "")))
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


def render_publishing(days, scrape, organize, generate_image):
    st.subheader("Publish events to LinkedIn")
    if st.query_params.get("code") or st.query_params.get("error"):
        st.info("Return to your original Events tab and paste this tab's full address into "
                "the connection form. Keep that address private.")
    account = render_connection()
    st.caption("Each button scrapes Lu.ma, Cerebral Valley, Evion, and Meetup, organizes upcoming "
               "events starting tomorrow, generates an image, and publishes both publicly to "
               "your connected LinkedIn account. Image generation uses your Google API key.")
    runs = st.session_state.setdefault("events_publish_runs", {})
    for region, column in zip(REGION_TIMEZONES, st.columns(2)):
        with column:
            key = run_key(region, days, account) if account else ""
            run = runs.get(key, {})
            if run.get("status") == "running":
                run.update(status="failed", error="Preparation was interrupted. You can try again.")
            blocked = run.get("status") in {"published", "uncertain"}
            if st.button(f"Publish {region} events", key=f"events_publish_{region}",
                         type="primary", width="stretch", disabled=not account or blocked):
                with st.status(f"Preparing {region} events…", expanded=True) as status:
                    result = run_pipeline(
                        region, days, account, st.session_state, scrape, organize, generate_image,
                        lambda label: status.update(label=label),
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
                with st.expander(f"{region} organized events and image"):
                    st.text(run["text"])
                    if run.get("image"):
                        st.image(run["image"]["bytes"], caption=f"{region} event image", width="stretch")
