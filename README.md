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
