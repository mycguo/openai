

## 1. Getting Access to Google Drive and Google Docs

### Step 1: Create a Google Cloud Project
1. Log in to the [Google Cloud Console](https://console.cloud.google.com/).
2. Sign in with the Google account where your Google Docs are stored.
3. Click on **Create Project** and provide a suitable name for your project.
4. Note the **Project ID** for future reference.

### Step 2: Enable Required APIs
1. Navigate to **APIs & Services > Library**.
2. Search for **Google Drive API** and **Google Docs API**, then enable them.
3. Ensure the following APIs are enabled:
   - **Google Drive API**  
   - **Google Docs API**  

### Step 3: Create OAuth 2.0 Credentials
1. Go to **APIs & Services > Credentials**.
2. Click **Create Credentials** and select **OAuth 2.0 Client IDs**.
3. **Configure the OAuth Consent Screen** (if prompted):
   - Choose **External** for the user type.
   - Fill in the required fields (e.g., **App Name**, **User Support Email**).
   - Save and continue without adding scopes (the default email scope is sufficient for personal use).
   - Add your email under the **Test Users** section and finish.
4. **Choose Application Type**:
   - Select **Desktop App**.
   - Name it (e.g., `"RAG App"`) and click **Create**.
5. **Download `credentials.json`**:
   - After creating the credentials, click **Download** and save the file as `credentials.json`.

### Step 4: Add Yourself as a Tester
1. Navigate to **OAuth Consent Screen**.
2. Scroll to the **Test Users** section and add your Google account.
3. Publish the app for testing purposes.