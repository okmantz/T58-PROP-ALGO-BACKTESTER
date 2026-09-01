# Deploying to Firebase (installable phone/web app)

## How this actually works

Firebase Hosting only serves static files -- it cannot run this app's Python
backend (real pandas/numpy backtests, Monte Carlo, GA search). So:

- **Cloud Run** runs the actual Flask app, in the Docker container defined
  by `Dockerfile`, at the repo root.
- **Firebase Hosting** sits in front of it and proxies every request to that
  Cloud Run service (see `firebase.json`'s `rewrites`). Your phone/browser
  only ever sees one URL -- the Firebase Hosting domain.
- The app is already a PWA (`app/web/static/manifest.json` + `sw.js`) --
  "Add to Home Screen" on that URL installs it with its own icon, standalone
  window, no App Store/Play Store submission.
- Firebase and Cloud Run are the same underlying Google Cloud project --
  there's nothing extra to "connect."

This is **not** a native App Store/Play Store app. If you want that later,
it's a separate step (e.g. wrapping this same web app with Capacitor) --
not something Firebase alone provides.

## One-time setup

1. **Create a Firebase project**: https://console.firebase.google.com ->
   Add project. Note the **Project ID** (not the display name) -- it's
   shown under Project Settings.

2. **Put that Project ID in two places:**
   - `.firebaserc` in this repo, replacing `REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID`.
   - A GitHub repo secret named `GCP_PROJECT_ID` (Settings -> Secrets and
     variables -> Actions -> New repository secret).

3. **Enable billing** on the GCP project. Cloud Run has a generous free
   tier, but it requires a billing account to be attached even to stay
   within it.

4. **Create a service account** for GitHub Actions to deploy with:
   - Google Cloud Console -> IAM & Admin -> Service Accounts -> Create.
   - Grant it these roles: `Cloud Run Admin`, `Service Account User`,
     `Artifact Registry Writer`, `Firebase Hosting Admin`.
   - Create a JSON key for it, download the file.
   - Paste the **entire contents** of that JSON file into a GitHub repo
     secret named `GCP_SA_KEY`.

5. **Enable the required APIs** on the project (one-time, either click
   through the console prompts on first deploy, or run):
   ```
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com firebasehosting.googleapis.com
   ```

6. **Push to `main`.** `.github/workflows/deploy-firebase.yml` runs
   automatically: builds the Docker image, pushes it to Artifact Registry,
   deploys it to Cloud Run, then deploys Firebase Hosting on top of it.
   Your app is live at `https://<your-project-id>.web.app`.

## Deploying by hand (no GitHub Actions)

```bash
# One-time local setup
npm install -g firebase-tools
gcloud auth login
firebase login

# Every deploy
gcloud run deploy t58-backtester --source . --region us-central1 --allow-unauthenticated
firebase deploy --only hosting
```

## Important limitation: storage is not durable

`app/data/storage.py` writes datasets/strategies/reports next to the app's
own code (`get_app_base_dir()`). On the desktop .exe that's a real folder
on your machine that persists forever. **On Cloud Run, that's the
container's own ephemeral filesystem** -- it resets on every redeploy and
isn't shared if the service ever scales past one instance.

The workflow above pins `--min-instances=1 --max-instances=1` specifically
so you always land on the same running instance and don't lose data
mid-session from scaling. But a redeploy (pushing new code) still wipes
whatever was saved to the Strategy Library or Reports through the web app
in between. For a single-user personal tool this is usually fine -- just
know that "download the strategy/report you care about" beats "assume it's
still there next week." Migrating this to real persistent storage (Cloud
Storage for files, Firestore for the strategy library / experiment
history) is a bigger change and a good candidate for a later round if it
starts to bite.

## Costs

Cloud Run's free tier covers light personal use easily. With
`min-instances=1` the container never fully idles down to zero, so it will
show a small (typically low-single-digit-dollars/month) charge instead of
staying at $0 -- worth knowing going in, not a surprise on the bill.
