# Prepare regional AI events for LinkedIn

Run the Events app with `python -m streamlit run events.py`. Configure
`GOOGLE_API_KEY` in Streamlit secrets or environment variables for image generation.

Click **Prepare Bay Area events** or **Prepare New York events**. Each button:

1. Scrapes Lu.ma, Cerebral Valley, Evion, and Meetup for fresh events in that region.
2. Organizes upcoming events into a daily agenda, starting tomorrow.
3. Prepares the full organized agenda and generates an image from it.
4. Displays the organized preview, image, and **Copy full agenda** button.

No earlier scrape, organize, or image-generation step is required. **Days to
scrape** controls the date window. Image generation incurs provider usage.
Preparation works without a LinkedIn connection and never uploads images or
creates LinkedIn posts. Each region's results are kept separately. Clicking
Prepare again refreshes that region; rerendering the page does not repeat work.

An unavailable source is skipped with a warning. No events or failed organization
stops preparation before image generation. If image generation fails, the prepared
text remains available to copy and the button allows another attempt.

Regional previews use the same day headings, bulleted event list, and linked
titles as **Organize Events → LinkedIn Preview**. **Copy full agenda** copies both
rich HTML and plain text, as in that flow. Plain text places each complete RSVP
URL on its own line and separates event blocks with blank lines.

The page shows one full agenda and image per region. Every event, host,
description, and complete RSVP link from the organized agenda is retained; there
is no shorter version or publication-length cutoff. The image is generated from
the full agenda. Results remain available after disconnecting LinkedIn.

## Optional LinkedIn connection

The existing connection control remains available, but is not required for
preparation. Configure `LINKEDIN_CLIENT_ID` and `LINKEDIN_CLIENT_SECRET` in
Streamlit secrets or environment variables. Register this Events page's exact URL
as an authorized redirect in the LinkedIn developer console, including any trailing
slash. Optionally set `EVENTS_LINKEDIN_REDIRECT_URI` to that URL; otherwise the app
uses its current URL. The podcast and YouTube redirect settings are not reused.
Enable Share on LinkedIn and Sign In with LinkedIn using OpenID Connect for
`w_member_social`, `openid`, and `profile`.

Click **Connect to LinkedIn**, then **Authorize on LinkedIn**. After authorization,
you return to Events in the same tab and connect automatically; no URL copying or
extra completion step is needed. Your Events workspace, including prepared
regional results, is restored after the redirect. Pending requests are kept in
server memory for ten minutes, bound to the initiating browser, and consumed once.
An app restart requires starting a new connection. Tokens remain in the returned
session and are cleared on disconnect.

Automated tests mock scraping, generation, and OAuth. They verify that preparation
never invokes LinkedIn publication or image-upload initialization. They do not
make billable image requests or create public LinkedIn posts.
