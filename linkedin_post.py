import streamlit as st
import requests
import urllib.parse
import time

# LinkedIn API Configuration
LINKEDIN_API_URL = "https://api.linkedin.com/v2/ugcPosts"
LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

# Page configuration
st.set_page_config(
    page_title="LinkedIn Post Publisher",
    page_icon="📱",
    layout="centered"
)

def get_linkedin_config():
    """Get LinkedIn app configuration from secrets"""
    try:
        return {
            "client_id": st.secrets["LINKEDIN_CLIENT_ID"],
            "client_secret": st.secrets["LINKEDIN_CLIENT_SECRET"],
            "redirect_uri": st.secrets["LINKEDIN_REDIRECT_URI"]
        }
    except KeyError as e:
        st.error(f"Missing LinkedIn configuration: {e}")
        st.info("""
        Please add the following to your .streamlit/secrets.toml file:

        ```toml
        LINKEDIN_CLIENT_ID = "your-linkedin-app-client-id"
        LINKEDIN_CLIENT_SECRET = "your-linkedin-app-client-secret"
        LINKEDIN_REDIRECT_URI = "http://localhost:8501"
        ```

        To get these credentials:
        1. Go to https://www.linkedin.com/developers/apps
        2. Create a new app or select existing one
        3. Add 'w_member_social' permission
        4. Set redirect URI to your Streamlit app URL + /callback
        """)
        return None

def generate_auth_url(config):
    """Generate LinkedIn OAuth authorization URL"""
    params = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "scope": "w_member_social openid email profile",
        "state": "linkedin_post_app"  # CSRF protection
    }
    return f"{LINKEDIN_AUTH_URL}?{urllib.parse.urlencode(params)}"


def get_query_param_value(params, key):
    """Return the first value for a query parameter irrespective of Streamlit API version."""
    if params is None:
        return None

    value = None

    if isinstance(params, dict):
        value = params.get(key)
    else:
        getter = getattr(params, "get", None)
        if callable(getter):
            try:
                value = getter(key)
            except TypeError:
                value = getter(key, None)

    if isinstance(value, (list, tuple)):
        return value[0] if value else None

    return value

def exchange_code_for_token(code, config):
    """Exchange authorization code for access token"""
    params = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': config["redirect_uri"],
        'client_id': config["client_id"],
        'client_secret': config["client_secret"]
    }

    try:
        response = requests.post(LINKEDIN_TOKEN_URL, data=params)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Token exchange failed: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        st.error(f"Failed to get access token: {e}")
        return None


def post_to_linkedin(content, access_token, author_id):
    """Post content to LinkedIn using UGC Post API"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    # Construct the post payload according to LinkedIn UGC Post API
    # Use "~" to represent the authenticated user
    payload = {
        "author": "urn:li:person:~" if author_id == "~" else f"urn:li:person:{author_id}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": content
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    try:
        response = requests.post(LINKEDIN_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as e:
        return False, str(e)

def main():
    st.title("📱 LinkedIn Post Publisher")
    st.markdown("Create and publish posts directly to your LinkedIn profile")

    # Get LinkedIn configuration
    config = get_linkedin_config()
    if not config:
        return

    # Handle OAuth callback
    try:
        # Try the new method first
        query_params = st.query_params
    except AttributeError:
        # Fallback to experimental method for older Streamlit versions
        query_params = st.experimental_get_query_params()

    code_param = get_query_param_value(query_params, "code")
    state_param = get_query_param_value(query_params, "state")
    token_active = (
        "linkedin_token" in st.session_state
        and time.time() < st.session_state.get("token_expires", 0)
    )

    if code_param and state_param == "linkedin_post_app" and not token_active:
        code = code_param

        with st.spinner("Exchanging authorization code for access token..."):
            token_data = exchange_code_for_token(code, config)

            if token_data and "access_token" in token_data:
                st.session_state.linkedin_token = token_data["access_token"]
                st.session_state.token_expires = time.time() + token_data.get("expires_in", 5184000)

                # For posting, we'll use "~" as the author which represents the authenticated user
                st.session_state.author_id = "~"

                try:
                    st.query_params.clear()
                except AttributeError:
                    st.experimental_set_query_params()

                st.success("✅ Successfully connected to LinkedIn!")
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            else:
                st.error("Failed to get access token from LinkedIn")

    # Check if user is authenticated
    if "linkedin_token" not in st.session_state or time.time() > st.session_state.get("token_expires", 0):
        st.info("🔐 Please connect your LinkedIn account to start posting")

        col1, col2, _ = st.columns([1, 2, 1])
        with col2:
            if st.button("🔗 Connect LinkedIn Account", type="primary", use_container_width=True):
                auth_url = generate_auth_url(config)
                st.markdown(f"[Click here to authorize the app]({auth_url})")
                st.info("You'll be redirected to LinkedIn to authorize the app, then redirected back here.")

        st.divider()
        st.markdown("### 📋 Setup Instructions")
        st.markdown("""
        1. **Create LinkedIn App**: Go to [LinkedIn Developers](https://www.linkedin.com/developers/apps) and create a new app
        2. **Add Permissions**: Enable the following permissions: `w_member_social`, `openid`, `email`, `profile`
        3. **Set Redirect URI**: Add your Streamlit app URL (e.g., `http://localhost:8501`)
        4. **Configure Secrets**: Add your app credentials to `.streamlit/secrets.toml`

        **Required Permissions**:
        - `w_member_social`: For posting content to LinkedIn
        - `openid`, `email`, `profile`: For user authentication and identification
        """)
        return

    # Show connected user
    if "linkedin_token" in st.session_state:
        st.success("✅ Connected to LinkedIn account!")

    st.divider()

    # Content input section
    st.subheader("✍️ Create Your Post")

    # Text area for post content
    post_content = st.text_area(
        "Post Content",
        height=200,
        max_chars=3000,
        placeholder="What would you like to share on LinkedIn? You can write up to 3,000 characters.",
        help="LinkedIn posts can be up to 3,000 characters long."
    )

    # Character counter
    char_count = len(post_content)
    char_remaining = 3000 - char_count

    col1, col2 = st.columns([1, 1])
    with col1:
        if char_remaining < 0:
            st.error(f"❌ {abs(char_remaining)} characters over limit")
        elif char_remaining < 100:
            st.warning(f"⚠️ {char_remaining} characters remaining")
        else:
            st.info(f"ℹ️ {char_remaining} characters remaining")

    with col2:
        st.metric("Character Count", f"{char_count:,}/3,000")

    # Preview section
    if post_content:
        st.subheader("👀 Preview")
        with st.container():
            st.markdown("**LinkedIn Post Preview:**")
            st.text_area("Post Preview", value=post_content, height=100, disabled=True, label_visibility="collapsed")

    st.divider()

    # Post button
    col1, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "📤 Post to LinkedIn",
            type="primary",
            use_container_width=True,
            disabled=not post_content or char_remaining < 0
        ):
            if not post_content.strip():
                st.error("Please enter some content to post")
                return

            with st.spinner("Publishing to LinkedIn..."):
                success, result = post_to_linkedin(
                    post_content.strip(),
                    st.session_state.linkedin_token,
                    st.session_state.author_id
                )

            if success:
                st.success("🎉 Successfully posted to LinkedIn!")
                st.balloons()

                # Show post details
                if isinstance(result, dict) and "id" in result:
                    post_id = result["id"]
                    st.info(f"Post ID: {post_id}")

                # Clear the content after successful post
                if st.button("Create Another Post"):
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
            else:
                st.error(f"❌ Failed to post to LinkedIn: {result}")
                st.info("Please check your connection and try again.")

    # Disconnect option
    st.divider()
    if st.button("🔌 Disconnect LinkedIn Account", type="secondary"):
        # Clear all LinkedIn-related session state
        keys_to_remove = ["linkedin_token", "token_expires", "author_id"]
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Disconnected from LinkedIn")
        st.rerun()

if __name__ == "__main__":
    main()
