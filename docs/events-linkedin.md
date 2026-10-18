# Publish regional AI events to LinkedIn

Run the Events app with `python -m streamlit run events.py`.

Configure `LINKEDIN_CLIENT_ID`, `LINKEDIN_CLIENT_SECRET`, and `GOOGLE_API_KEY` in
Streamlit secrets or environment variables. Register this Events page's exact URL
as an authorized redirect in the LinkedIn developer console, including any trailing
slash. Optionally set `EVENTS_LINKEDIN_REDIRECT_URI` to that URL; otherwise the app
uses its current URL. The podcast and YouTube redirect settings are not reused.
Enable Share on LinkedIn and Sign In with LinkedIn using OpenID Connect for
`w_member_social`, `openid`, and `profile`. The existing `LINKEDIN_API_VERSION`
setting is shared with the podcast app.

1. Click **Connect to LinkedIn**, then **Authorize on LinkedIn**.
2. Authorize your account on LinkedIn. You return to Events in the same tab and
   connect automatically; no URL copying or extra completion step is needed.
   Your Events workspace is restored after the redirect. Pending requests are
   kept in server memory for ten minutes, bound to the initiating browser, and
   consumed once. An app restart requires starting a new connection. Tokens
   remain in the returned session and are cleared on disconnect.
3. Click **Publish Bay Area events** or **Publish New York events**. That click
   scrapes all four sources, keeps only upcoming events in the chosen region,
   organizes them into a daily agenda, generates a Google image from that agenda,
   and publishes the agenda and image publicly to the connected LinkedIn profile.
   No earlier scrape, organize, or image-generation step is required. Image
   generation incurs provider usage. **Days to scrape** controls the date window,
   starting tomorrow.

The workflow uses fresh source results and keeps each region's publication and
image separately. An unavailable source is skipped with a warning. No events,
failed organization, or failed image generation stops publication. An image
upload failure never falls back to a text-only post. The completed post link and
its organized text and image remain visible on the page.

Regional post previews use the same day headings, bulleted event list, and linked
titles as **Organize Events → LinkedIn Preview**. **Copy for LinkedIn** copies both
rich HTML and plain text, as in that flow. The plain-text/API version places each
complete RSVP URL on its own line and separates event blocks with blank lines.

Publishing creates one post within the observed 4,000-character API limit,
including reserved-character escaping and emoji accounting. If the full agenda
is too long, remove descriptions and hosts, then include the earliest complete
event entries that fit, with a count when some events are omitted. **Full agenda**
and **Copy full agenda** retain every event. The image is generated from the
concise post that will actually be published. Existing oversized 400 rejections
are unlocked for an explicit retry; rerendering the page never posts by itself.

Published runs are blocked from repeating for the same region, account, day, and
date window within the session. A timeout or unconfirmed create request requires
checking LinkedIn before explicitly allowing a retry. This is session-local
duplicate protection; restarting the app or using another session resets it.

Automated tests mock scraping, generation, OAuth, and publishing. They do not make
billable image requests or create public LinkedIn posts. A live OAuth round trip
and public post still require validation with the configured LinkedIn app.

References: [LinkedIn authorization code flow](https://learn.microsoft.com/en-us/linkedin/shared/authentication/authorization-code-flow),
[Posts API](https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api),
[Images API](https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/images-api),
and [text format](https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/little-text-format).
