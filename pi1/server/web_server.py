"""
server/web_server.py
====================
Flask web server for Pi 1.

Endpoints:
    GET /               — MJPEG viewer page
    GET /video_feed     — MJPEG multipart stream
    GET /person_status  — JSON { last_seen: <epoch>, seconds_ago: <float> }
                          Polled by Pi 2's ChildWatchdogThread.

The MJPEG frame and person-seen timestamp are shared state written by the
InferenceThread and read here.  Both are protected by threading locks.
"""

import time
import json
import threading

from flask import Flask, Response, render_template_string


# ─── Shared state (written by InferenceThread) ───────────────────────────────

mjpeg_lock          = threading.Lock()
mjpeg_frame         = None          # latest JPEG bytes for the MJPEG stream

person_status_lock  = threading.Lock()
person_last_seen    = 0.0           # epoch of last frame with a YOLO detection


# ─── Helper writers (called by InferenceThread) ──────────────────────────────

def update_mjpeg_frame(jpeg_bytes):
    global mjpeg_frame
    with mjpeg_lock:
        mjpeg_frame = jpeg_bytes


def update_person_seen():
    global person_last_seen
    with person_status_lock:
        person_last_seen = time.time()


# ─── Flask app ────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)

_PAGE_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pi 1 Monitor</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#e0e0e0;font-family:'Segoe UI',sans-serif;
display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:24px 16px}
h1{font-size:1.4rem;font-weight:600;margin-bottom:20px;color:#fff}
.s{width:100%;max-width:800px;border-radius:12px;overflow:hidden;
border:2px solid #2a2a2a;box-shadow:0 8px 32px rgba(0,0,0,0.6)}
img{width:100%;display:block}
.f{margin-top:14px;font-size:.75rem;color:#555}
</style></head><body>
<h1>Pi 1 — Fight + Fall Monitor (USB Camera)</h1>
<div class="s"><img src="/video_feed" alt="Loading..."></div>
<p class="f">YOLOv8n-pose &bull; Fight + Fall + Motionless Detection</p>
</body></html>"""


def _gen_mjpeg():
    """Generator that yields MJPEG frames for the /video_feed endpoint."""
    while True:
        with mjpeg_lock:
            f = mjpeg_frame
        if f is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
        time.sleep(0.033)   # ~30 fps ceiling for the browser stream


@flask_app.route('/')
def index():
    return render_template_string(_PAGE_HTML)


@flask_app.route('/video_feed')
def video_feed():
    return Response(_gen_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/person_status')
def person_status():
    """
    Returns JSON:  { "last_seen": <epoch float>, "seconds_ago": <float> }
    last_seen == 0.0 means no person detected since startup.
    Polled by Pi 2's ChildWatchdogThread to implement the cross-camera watchdog.
    """
    with person_status_lock:
        t = person_last_seen
    age = (time.time() - t) if t > 0 else float('inf')
    return flask_app.response_class(
        response=json.dumps({'last_seen': t, 'seconds_ago': round(age, 1)}),
        mimetype='application/json',
    )


def start_server(port):
    """Launch Flask in a daemon thread.  Returns immediately."""
    t = threading.Thread(
        target=lambda: flask_app.run(
            host='0.0.0.0', port=port,
            use_reloader=False, threaded=True),
        daemon=True,
    )
    t.start()
    print(f'[Server] Stream:        http://0.0.0.0:{port}')
    print(f'[Server] Person status: http://0.0.0.0:{port}/person_status')


def get_latest_frame():
    """Return the latest MJPEG JPEG bytes for /snapshot command. Returns None if no frame yet."""
    with mjpeg_lock:
        return mjpeg_frame
