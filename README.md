# Auto Music Channel

Makes and uploads 2 AI music videos per day (9:00 AM and 7:00 PM IST) using GitHub Actions.

## Required GitHub Secrets (Settings > Secrets and variables > Actions)
YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN, GEMINI_API_KEY
(HF_TOKEN is optional, it only speeds up model downloads.)

## First test (does NOT upload)
Actions tab > "Make and upload music video" > Run workflow > keep dry_run ticked.
When it finishes, download the "test-output" artifact and check video.mp4 and thumbnail.jpg.

## Go live
Nothing else to do. The schedule runs by itself. To post publicly the YouTube API
project must pass the API compliance audit, until then uploads are forced to private.

## Settings you can change (top of main.py or env vars)
MIN_MINUTES / MAX_MINUTES (video length), MAX_CLIPS (more = slower but less repetition),
NICHES list (add or remove genres), UPLOAD_PRIVACY (public / unlisted / private).
Posting times: edit the two cron lines in .github/workflows/upload.yml (times are UTC).
