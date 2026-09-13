"""YouTube workflow UI backed by the podcast app's existing service functions."""

import hashlib
import secrets
import time
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import streamlit as st

import ai_podcast as services


DEFAULT_PROMPT = services.default_article_prompt("YouTube video")
CONTENT_KEYS = ("yt_article", "yt_article_image", "yt_image_prompt", "yt_include_image", "yt_publish_confirm")


def clear_generated_content():
    """Keep user-written article instructions and account connection across videos."""
    for key in CONTENT_KEYS:
        st.session_state.pop(key, None)


def article_changed():
    for key in ("yt_article_image", "yt_include_image", "yt_publish_confirm"):
        st.session_state.pop(key, None)


def reset_article_prompt():
    st.session_state.yt_article_prompt = DEFAULT_PROMPT


def linkedin_config():
    # Do not reuse the podcast redirect: it may point at port 8501 or another app.
    runtime = services._get_runtime_url()
    redirect = services._optional_config("YOUTUBE_LINKEDIN_REDIRECT_URI") or services._strip_redirect_uri_extras(runtime)
    return {
        "client_id": services._optional_config("LINKEDIN_CLIENT_ID"),
        "client_secret": services._optional_config("LINKEDIN_CLIENT_SECRET"),
        "redirect_uri": redirect,
    }


def start_authorization(config):
    state = secrets.token_urlsafe(32)
    st.session_state.yt_oauth_pending = {"state": state, "created": time.time(), "redirect_uri": config["redirect_uri"]}
    return services.LINKEDIN_AUTH_URL + "?" + urlencode({
        "response_type": "code", "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"], "scope": "w_member_social openid profile",
        "state": state,
    })


def validate_callback(callback_url, pending, now=None):
    if not pending or (time.time() if now is None else now) - pending["created"] > 600:
        raise ValueError("Connection request expired. Start a new connection.")
    parsed = urlsplit(callback_url.strip())
    base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    if base != pending["redirect_uri"]:
        raise ValueError("The callback URL does not match this app's redirect URL.")
    params = parse_qs(parsed.query)
    states = params.get("state", [])
    if len(states) != 1 or not secrets.compare_digest(states[0], pending["state"]):
        raise ValueError("Connection state mismatch. Use the callback from this tab's connection request.")
    if "error" in params:
        raise ValueError("LinkedIn authorization was declined. Start a new connection if needed.")
    codes = params.get("code", [])
    if len(codes) != 1 or not codes[0]:
        raise ValueError("The callback URL does not contain an authorization code.")
    return codes[0]


def finish_authorization():
    """Form callback: permits clearing sensitive widget contents before rendering."""
    callback = st.session_state.pop("yt_oauth_callback", "")
    pending = st.session_state.get("yt_oauth_pending")
    try:
        code = validate_callback(callback, pending)
        config = linkedin_config()
        config["redirect_uri"] = pending["redirect_uri"]
        st.session_state.pop("yt_oauth_pending", None)  # one-time exchange
        st.session_state.pop("yt_auth_url", None)
        token_data = services.exchange_code_for_token(code, config)
        if not token_data or not token_data.get("access_token"):
            raise ValueError("LinkedIn token exchange failed. Start a new connection.")
        token = token_data["access_token"]
        author = services.fetch_authenticated_member_urn(token)
        if not author:
            raise ValueError("Could not identify the LinkedIn member. Check the profile permission and reconnect.")
        st.session_state.yt_linkedin = {"token": token, "author": author,
                                      "expires": time.time() + float(token_data.get("expires_in", 3600))}
        st.session_state.pop("yt_auth_error", None)
    except (ValueError, TypeError):
        st.session_state.yt_auth_error = "Could not complete the connection. Verify the callback URL, matching state, and ten-minute expiry, or start again."


def disconnect():
    for key in ("yt_linkedin", "yt_oauth_pending", "yt_auth_url", "yt_oauth_callback", "yt_auth_error", "yt_publish_confirm"):
        st.session_state.pop(key, None)


def render_connection():
    account = st.session_state.get("yt_linkedin")
    if account and account["expires"] <= time.time():
        disconnect()
        account = None
    if account:
        st.success(f"LinkedIn connected • {account['author']}")
        st.button("Disconnect LinkedIn", on_click=disconnect)
        return account
    config = linkedin_config()
    if not all(config.values()):
        st.info("Configure LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET, and YOUTUBE_LINKEDIN_REDIRECT_URI to connect.")
        return None
    with st.expander("Connect LinkedIn"):
        st.caption(f"Register this exact redirect URL in your LinkedIn developer app: {config['redirect_uri']}")
        st.write("Authorize in a new tab. When LinkedIn redirects back, copy that tab's full address and paste it below in this original tab. This keeps your draft and connection request together.")
        if st.button("Start LinkedIn connection"):
            st.session_state.yt_auth_url = start_authorization(config)
        if st.session_state.get("yt_auth_url"):
            st.link_button("Authorize on LinkedIn (new tab)", st.session_state.yt_auth_url)
            with st.form("yt_connect_form"):
                st.text_input("Returned callback URL", type="password", key="yt_oauth_callback")
                st.form_submit_button("Complete connection", on_click=finish_authorization)
        if st.session_state.get("yt_auth_error"):
            st.error(st.session_state.yt_auth_error)
    return None


def post_fingerprint(article, image, author):
    digest = hashlib.sha256()
    for value in (author.encode(), article.encode(), image.get("bytes", b"") if image else b""):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def render_workflow():
    if st.query_params.get("code") or st.query_params.get("error"):
        st.info("LinkedIn has returned here. Copy this page's full address into the connection form in your original YouTube app tab. Do not share that address.")

    st.subheader("Generate LinkedIn Article")
    st.caption("Creates a LinkedIn feed post, not a long-form LinkedIn newsletter/article. Generation uses Claude; image generation uses Google and may also use Claude for themes. Provider charges apply.")
    st.session_state.setdefault("yt_article_prompt", DEFAULT_PROMPT)
    st.text_area("Article generation prompt", key="yt_article_prompt", height=280,
                 help="Edit tone, audience, structure, and focus. The complete transcript is appended automatically.")
    st.button("Reset article prompt", on_click=reset_article_prompt)
    transcript = st.session_state.get("yt_transcript_text", "")
    if not services.ANTHROPIC_API_KEY:
        st.info("Set ANTHROPIC_API_KEY to enable article generation.")
    if st.button("Generate Article", disabled=not transcript or not services.ANTHROPIC_API_KEY or not st.session_state.yt_article_prompt.strip()):
        try:
            with st.spinner("Generating article from the full transcript..."):
                metadata = st.session_state.get("yt_transcript_metadata", {})
                article = services.generate_linkedin_article(
                    transcript, episode_title=metadata.get("title") or metadata.get("webpage_url", ""),
                    source_kind="YouTube video", prompt_override=st.session_state.yt_article_prompt,
                )
            if not article.strip():
                raise ValueError("No article text was returned.")
            article_changed()
            st.session_state.yt_article = article
        except Exception as exc:
            st.error(f"Article generation failed: {exc}")

    if "yt_article" in st.session_state:
        st.text_area("Edit Article", key="yt_article", height=350, on_change=article_changed)
        article = st.session_state.yt_article.strip()
        st.caption(f"{len(article):,}/3,000 characters")
        st.download_button("Download article", article, file_name="youtube-linkedin-post.txt", mime="text/plain")
    else:
        article = ""

    st.subheader("Generate Article Image")
    st.session_state.setdefault("yt_image_prompt", services._build_article_image_prompt())
    st.text_area("Image generation prompt", key="yt_image_prompt", height=130)
    if not services.GOOGLE_API_KEY:
        st.info("Set GOOGLE_API_KEY to enable image generation.")
    if st.button("Generate Article Image", disabled=not article or not services.GOOGLE_API_KEY):
        with st.spinner("Generating image from the edited article..."):
            ok, payload, _, error = services.generate_article_image(article, prompt_override=st.session_state.yt_image_prompt)
        if ok and payload:
            st.session_state.yt_article_image = payload
            st.session_state.pop("yt_publish_confirm", None)
        else:
            st.error(error or "Image generation failed.")
    image = st.session_state.get("yt_article_image")
    if image:
        st.image(image["bytes"], caption="Image preview", width="stretch")
        mime = image.get("mime_type", "image/png")
        extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "bin")
        st.download_button("Download image", image["bytes"], file_name=f"youtube-linkedin-image.{extension}", mime=mime)

    st.subheader("Publish to LinkedIn")
    account = render_connection()
    include_image = st.checkbox("Include generated image", value=True, key="yt_include_image", on_change=lambda: st.session_state.pop("yt_publish_confirm", None)) if image else False
    selected_image = image if include_image else None
    confirmed = st.checkbox("I reviewed this draft and want to publish it publicly to the connected LinkedIn account.", key="yt_publish_confirm")
    if len(article) > 3000:
        st.warning("Shorten the article to 3,000 characters or fewer before publishing.")
    attempts = st.session_state.setdefault("yt_publish_attempts", {})
    fingerprint = post_fingerprint(article, selected_image, account["author"]) if account else ""
    prior = attempts.get(fingerprint)
    if st.button("Publish to LinkedIn", type="primary", disabled=not account or not confirmed or not article or len(article) > 3000 or bool(prior)):
        # Mark before sending: reruns must never automatically repeat a create request.
        attempts[fingerprint] = {"status": "uncertain"}
        try:
            with st.spinner("Publishing reviewed content..."):
                ok, result = services.post_to_linkedin(article, account["token"], account["author"],
                                                       image_payload=selected_image, allow_image_fallback=False)
            if ok:
                attempts[fingerprint] = {"status": "published", "url": services.build_linkedin_post_url(result.get("id", ""))}
            else:
                attempts[fingerprint]["error"] = f"Publishing did not report success: {result}"
        except Exception:
            attempts[fingerprint]["error"] = "Publishing was interrupted. Check LinkedIn before retrying; the post may already exist."
        st.rerun()
    if prior:
        if prior["status"] == "published":
            st.success("Published to LinkedIn. This exact content will not be posted twice in this session.")
            if prior.get("url"):
                st.link_button("View LinkedIn post", prior["url"])
        else:
            if prior.get("error"):
                st.error(prior["error"])
            st.warning("A publish attempt was made. Check LinkedIn before retrying to avoid a duplicate.")
            if st.button("I checked LinkedIn; no post exists — allow retry"):
                attempts.pop(fingerprint, None)
                st.rerun()
