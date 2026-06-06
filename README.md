# Farm Detection

DetectField is a Raspberry Pi wildlife and human detection dashboard. It runs YOLO on a Pi camera or USB webcam and serves a live Flask dashboard for surveillance control, power telemetry, detection events, and MJPEG video.

## Project Layout

- `raspberry-pi-animal-detection/web_app.py` - Flask dashboard, status API, live feed, and threaded YOLO surveillance.
- `raspberry-pi-animal-detection/detect.py` - original camera detection script.
- `raspberry-pi-animal-detection/templates/index.html` - live local dashboard UI.
- `raspberry-pi-animal-detection/systemd/aegisfield.service` - systemd service used on the Pi.
- `docs/index.html` - static GitHub Pages site for the project.

## Raspberry Pi Runtime

Expected Pi path:

```bash
/home/project/detection
```

Run manually:

```bash
cd /home/project/detection
source venv/bin/activate
python web_app.py
```

Service commands:

```bash
sudo systemctl restart aegisfield.service
systemctl status aegisfield.service
journalctl -u aegisfield.service -n 80 --no-pager
```

Health checks:

```bash
curl -I http://127.0.0.1:8080/
curl http://127.0.0.1:8080/api/status
```

## Login, Roles, And Events

The dashboard uses local Flask sessions and SQLite. No Firebase is required for the current single-Pi setup.

Default first-run login:

- Username: `admin`
- Password: `detector`

Override before first launch:

```bash
DETECTFIELD_ADMIN_USER=admin DETECTFIELD_ADMIN_PASSWORD='change-me' python web_app.py
```

Roles:

- `admin` - surveillance controls and user management.
- `operator` - surveillance controls.
- `viewer` - dashboard and live feed access.

Database file:

```bash
/home/project/detection/detectfield.db
```

Stored records:

- `detection_events` - initial detection, continuing presence heartbeat, and exit.
- `detection_sessions` - active detection interval with start time, end time, labels, peak confidence, and event count.

The app keeps a short no-detection grace window before closing an event interval, so split-second YOLO misses do not create false exits.

## Clip Archive And Settings

Detection intervals are recorded as MP4 clips in:

```bash
/home/project/detection/clips
```

The dashboard has separate tabs:

- Dashboard - live feed, controls, power, detections, event history.
- Archive - all recorded human/animal detection clips with playback and delete controls.
- Settings - security, notifications, storage retention, camera source, feed permissions, and user management.

Default clip retention is 30 days. Admins can set retention to 1 week, 2 weeks, 30 days, 60 days, or 90 days. Old clips are deleted automatically; deleting a clip from the Archive removes the video file and marks the database record as deleted.

Clips are recorded as fast temporary MJPEG files, then converted in the background to browser-playable H.264 MP4 with `ffmpeg`. This avoids blocking the camera stream while keeping playback reliable in the web dashboard.

Firebase Storage can be used for online clip storage. Configure these on the Pi service, then set `Clip Storage` to `Firebase Storage` in Settings:

```bash
FIREBASE_STORAGE_BUCKET=your-project.appspot.com
FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

When Firebase is enabled, the Pi uses temporary files only during recording/conversion/upload, then deletes the local video after upload.

## WhatsApp Alerts

WhatsApp alerts are optional and currently wired for Twilio WhatsApp. Set these environment variables in the systemd service or shell:

```bash
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
ALERT_WHATSAPP_TO=whatsapp:+15551234567
```

Alerts fire on initial detection. If these variables are missing, the app still runs without phone alerts.

## Camera Notes

The app scans USB camera indexes `0-5` by default. Override the source with:

```bash
AEGISFIELD_CAMERA=0 python web_app.py
```

Useful camera diagnostics:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
rpicam-hello --list-cameras
```

## GitHub Pages

GitHub Pages can host only static files. The static website in `docs/` is a public project page and can link to the Pi dashboard, but it cannot run YOLO, access the Pi camera, or replace the Flask backend.

Enable Pages in GitHub repo settings with:

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`
