"""
server/web_server.py
====================
Flask web server for Pi 2.

Endpoints:
    GET /           — MJPEG viewer page
    GET /video_feed — MJPEG multipart stream

Note: Pi 2 does NOT expose /person_status.
Pi 1 exposes /person_status and Pi 2 polls it.
"""

import time
import threading

from flask import Flask, Response, render_template_string
import cv2
import numpy as np


# ─── Shared MJPEG state ───────────────────────────────────────────────────────

mjpeg_lock  = threading.Lock()
mjpeg_frame = None


def update_mjpeg_frame(jpeg_bytes):
    """Called by InferenceThread to push the latest annotated frame."""
    global mjpeg_frame
    with mjpeg_lock:
        mjpeg_frame = jpeg_bytes


# ─── Idle frame ───────────────────────────────────────────────────────────────

def make_idle_frame(w=640, h=480, msg='Waiting for PIR...'):
    """
    Dark placeholder frame shown on the MJPEG stream when the camera
    is paused (PIR sleep mode).
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    ts = time.strftime('%H:%M:%S')
    for i, line in enumerate(['Camera Idle', msg, ts]):
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, line, ((w-tw)//2, h//2 - 20 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


# ─── Flask app ────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)

_PAGE_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pi 2 Monitor</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#e0e0e0;font-family:'Segoe UI',sans-serif;
display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:24px 16px}
h1{font-size:1.4rem;font-weight:600;margin-bottom:20px;color:#fff}
.s{width:100%;max-width:800px;border-radius:12px;overflow:hidden;
border:2px solid #2a2a2a;box-shadow:0 8px 32px rgba(0,0,0,0.6)}
img{width:100%;display:block}
.f{margin-top:14px;font-size:.75rem;color:#555}
</style></head><body>
<h1>Pi 2 — Fight + Fall Monitor (PIR + Audio)</h1>
<div class="s"><img src="/video_feed" alt="Loading..."></div>
<p class="f">YOLOv8n-pose &bull; Fight + Fall + Motionless + Audio</p>
</body></html>"""


def _gen_mjpeg():
    while True:
        with mjpeg_lock:
            f = mjpeg_frame
        if f is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
        time.sleep(0.033)


@flask_app.route('/')
def index():
    return render_template_string(_PAGE_HTML)


@flask_app.route('/video_feed')
def video_feed():
    return Response(_gen_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_server(port):
    """Launch Flask in a daemon thread. Returns immediately."""
    t = threading.Thread(
        target=lambda: flask_app.run(
            host='0.0.0.0', port=port,
            use_reloader=False, threaded=True),
        daemon=True,
    )
    t.start()
    print(f'[Server] Stream: http://0.0.0.0:{port}/')
