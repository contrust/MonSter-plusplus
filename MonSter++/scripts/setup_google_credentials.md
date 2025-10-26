# Google Cloud Credentials Setup Guide

This guide will help you set up Google Cloud credentials to download the WHU-Stereo dataset.

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" at the top
3. Click "New Project"
4. Enter a project name (e.g., "WHU-Dataset-Download")
5. Click "Create"

## Step 2: Enable Google Drive API

1. In your new project, go to "APIs & Services" > "Library"
2. Search for "Google Drive API"
3. Click on "Google Drive API"
4. Click "Enable"

## Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client ID"
3. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: "WHU Dataset Download"
   - User support email: your email
   - Developer contact information: your email
   - **IMPORTANT**: In "Test users" section, add your own email address
   - Save and continue through the steps
4. Back to credentials, choose "Desktop application"
5. Name: "WHU Dataset Download Client"
6. Click "Create"
7. Download the JSON file (it will be named something like `client_secret_xxx.json`)

## Step 3a: Configure OAuth Consent Screen (if needed)

If you get verification errors, you need to configure the consent screen properly:

1. Go to "APIs & Services" > "OAuth consent screen"
2. Set "Publishing status" to "Testing"
3. Add your email address to "Test users"
4. Save changes

## Step 4: Set Up Credentials File

1. Rename the downloaded file to `credentials.json`
2. Place it in the MonSter project root directory (same level as the scripts folder)
3. The file structure should look like:
   ```
   MonSter/
   ├── scripts/
   │   └── download_whu_python.py
   ├── credentials.json  ← Place here
   └── ...
   ```

## Step 5: Run the Download Script

Once you have `credentials.json` in place, run:

```bash
python scripts/download_whu_python.py
```

The first time you run it, it will:
1. Open a browser window for authentication
2. Ask you to sign in to your Google account
3. Request permission to access your Google Drive
4. Download the WHU-Stereo dataset

## Troubleshooting

- **"credentials.json not found"**: Make sure the file is in the correct location
- **"Google Drive API not enabled"**: Go back to Step 2 and enable the API
- **Authentication errors**: Delete `token.pickle` if it exists and try again
- **Permission denied**: Make sure the Google Drive file is shared with your account

### Error 403: access_denied / App not verified

If you get this error, follow these steps:

1. **Go to Google Cloud Console** > "APIs & Services" > "OAuth consent screen"
2. **Set Publishing status to "Testing"** (not "In production")
3. **Add your email address to "Test users"**:
   - Click "Add Users"
   - Enter your Google account email
   - Click "Add"
4. **Save changes**
5. **Try the download script again**

This error occurs because Google requires apps to be verified for public use, but for personal/testing use, you can add yourself as a test user.

## Security Notes

- Keep `credentials.json` secure and don't share it
- The script creates `token.pickle` to save authentication tokens
- You can delete `token.pickle` to force re-authentication 