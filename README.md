# INF2009-Group21 — Edge-Based Home Safety Monitor

> **Course:** INF2009 — Edge Computing and Analytics | **Group:** 21

A real-time home safety monitoring system running entirely at the edge — no cloud inference. Two Raspberry Pi 5 nodes collaborate to detect falls, fights, and audio events, and alert the user via Telegram.

---

## Team

| Name                     | Student ID |
| ------------------------ | ---------- |
| Sidharth Vinod           | 2400635    |
| Lim Bing Xian            | 2401649    |
| Tan Yu Xuan              | 2400653    |
| Boo Wai Yee Terry        | 2402445    |
| Premanand Aishwarya Shri | 2403053    |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  LOCAL NETWORK                             │
│                                                            │
│  ┌─────────────────────┐  HTTP   ┌─────────────────────┐  │
│  │  PI 1 (Always-On)   │◄────────│  PI 2 (PIR-Trigger) │  │
│  │                     │         │                     │  │
│  │  USB Camera → YOLO  │         │  USB Camera → YOLO  │  │
│  │  Mic → YAMNet       │         │  Mic → YAMNet       │  │
│  │  DHT22 (Temp/Hum)   │         │  PIR Sensor         │  │
│  │  USB Speaker (TTS)  │         │  Child Watchdog     │  │
│  │  Flask :5001        │         │  Flask :5000        │  │
│  └──────────┬──────────┘         └──────────┬──────────┘  │
└─────────────┼──────────────────────────────┼──────────────┘
              └──────────────┬───────────────┘
                      Telegram API (HTTPS)
                             │
                       User's Phone
```

**Pi 1** runs continuously and serves as the primary monitor. **Pi 2** wakes on PIR motion and polls Pi 1 every 10 s to implement a cross-camera child-missing watchdog — an alert fires only when *both* cameras see no person for 60 s, nearly eliminating false alarms.

---

## Data Sources

The system integrates four sensor modalities:

| Source                | Type          | Node        | Data                    |
| --------------------- | ------------- | ----------- | ----------------------- |
| USB Camera            | Vision        | Pi 1 & Pi 2 | 640×480 @ 30 fps video  |
| Logitech Built-in Mic | Audio         | Pi 1 & Pi 2 | 16 kHz PCM stream       |
| DHT22                 | Environmental | Pi 1 only   | Temperature & humidity  |
| PIR Sensor            | Motion        | Pi 2 only   | Binary presence trigger |

**Multi-modal fusion:** Audio context (laughing, crying, screaming) is fused with vision detections before alerts are sent. A laugh within 8 s of a fight score peak suppresses the alert; a cry within 6 s of a fall enriches the alert message.

---

## ML / AI Techniques

| Model               | Task                                        | Framework                  | Runs On     |
| ------------------- | ------------------------------------------- | -------------------------- | ----------- |
| YOLO26n-pose (ONNX) | Person detection + 17-keypoint skeleton     | ONNX Runtime / Ultralytics | Pi 1 & Pi 2 |
| YAMNet (TF-Hub)     | Audio classification — 521 AudioSet classes | TensorFlow 2.21            | Pi 1 & Pi 2 |

All inference runs **locally on the Raspberry Pi**. No video or audio is sent to the cloud — ensuring low latency (< 100 ms), privacy, and resilience to internet outages.

**Fall Detection** uses a 3-stage state machine: (1) candidate — spine horizontal ratio dx/dy > 0.9, torso angle < 60°, hips > 50% down the frame; (2) confirmation — candidate pose held for ≥ 3 s; (3) classification — slow angle-delta transitions → "lying down", fast → "fall alert". A motionless tracker fires an additional alert if the person does not move for 15–20 s after falling.

**Fight Detection** computes an 11-signal composite score per person pair covering wrist velocity, torso invasion, mutual facing, head proximity, kick signals, strike bursts, grapple IoU, and asymmetric aggression. A score above threshold (6 on Pi 1, 4 on Pi 2) sustained for ≥ 35% of frames in a 2 s window triggers an alert.

**Audio Detection** runs two layers simultaneously: YAMNet (512 ms inference interval) for broad AudioSet classification, and heuristic detectors (64 ms chunks) for low-latency impact, cry, scream, and laugh detection using RMS and spectral flux thresholds.

---

## Features

- **Fall detection** with lying-down vs fall classification and motionless follow-up alert
- **Fight detection** with 11-signal scoring and laugh-based false-positive suppression
- **Audio event detection** — impact, crying, screaming, laughing (neural + heuristic)
- **Cross-camera child-missing watchdog** (both cameras must agree before alert fires)
- **PIR-triggered camera sleep** on Pi 2 — saves CPU when no motion detected
- **Telegram alerts** with 6-second annotated GIF clips
- **Telegram bot** with 8 commands: `/help`, `/temp`, `/status`, `/snapshot`, `/clip`, `/arm`, `/disarm`, `/uptime`
- **DHT22 temperature & humidity** monitoring (Pi 1)
- **TTS speaker announcements** for immediate on-device feedback (Pi 1)
- **Live web dashboard** — MJPEG stream accessible from any browser on the LAN

---

## Dashboard

Both nodes serve a live web dashboard on the local network:

| URL                                  | Description                                               |
| ------------------------------------ | --------------------------------------------------------- |
| `http://<pi1-ip>:5001/`              | Pi 1 live dashboard — annotated video                     |
| `http://<pi1-ip>:5001/video_feed`    | Raw MJPEG stream                                          |
| `http://<pi1-ip>:5001/person_status` | JSON — `{ "last_seen": <epoch>, "seconds_ago": <float> }` |
| `http://<pi2-ip>:5000/`              | Pi 2 live dashboard — annotated video                     |
| `http://<pi2-ip>:5000/video_feed`    | Raw MJPEG stream                                          |

The annotated stream shows skeleton overlays, bounding boxes, track IDs, detection state labels (FALLING / FIGHTING / MOTIONLESS), and an FPS counter. Pi 2 shows an idle placeholder frame when the camera is in PIR-sleep mode.

The **Telegram bot** (Pi 1) acts as a remote dashboard — send `/snapshot` for the current frame, `/temp` for sensor readings, or `/status` for system state from anywhere with internet access.

---

## Hardware & Justification

| Component          | Choice                        | Key Reason                                                                   |
| ------------------ | ----------------------------- | ---------------------------------------------------------------------------- |
| Compute            | Raspberry Pi 5 (8 GB RAM)     | Enough RAM for concurrent ONNX + TF inference; quad-core for multi-threading |
| Camera             | Logitech USB webcam (640×480) | Plug-and-play V4L2; hot-swappable; no CSI ribbon fragility                   |
| Microphone         | Logitech webcam built-in mic  | Integrated with camera — one USB device for both video and audio             |
| Temperature sensor | DHT22 on GPIO 26              | ±0.5°C accuracy; single-wire; 3.3 V compatible                               |
| Motion sensor      | PIR on GPIO 26                | Zero standby power; digital GPIO trigger                                     |
| Speaker            | USB speaker                   | pyttsx3 compatible; network-independent TTS                                  |
| Vision model       | YOLOv8n-pose ONNX             | Smallest YOLO with 17 keypoints; fastest CPU inference                       |
| Audio model        | YAMNet (TF-Hub)               | 521 classes; pre-trained; matches 16 kHz input directly                      |
| Alerting           | Telegram Bot API              | Free; push delivery; rich media (GIF); two-way bot commands                  |
| Web server         | Flask                         | Lightweight; native MJPEG `multipart/x-mixed-replace` support                |

**Why not cloud inference?** Edge processing keeps video on-device (privacy), eliminates round-trip latency (< 100 ms vs 400–800 ms), works offline, and has no per-request API cost.

---

## Work Packages

| #    | Package                                                                                          | Responsible                             |
| ---- | ------------------------------------------------------------------------------------------------ | --------------------------------------- |
| WP1  | **Camera & Vision Pipeline** — USB capture, YOLO inference, HUD overlay, clip buffer             | Sidharth Vinod                          |
| WP2  | **Fall Detection** — Spine-angle state machine, motionless tracking, lying-down classification   | Sidharth Vinod, Tan Yu Xuan             |
| WP3  | **Fight Detection** — 11-signal scorer, per-person state machines, alert cooldowns               | Lim Bing Xian                           |
| WP4  | **Audio Detection** — YAMNet integration, heuristic detectors, cross-modal fusion                | Tan Yu Xuan                             |
| WP5  | **Alerting & Telegram Bot** — GIF encoding, alert queue, 8-command bot                           | Boo Wai Yee Terry                       |
| WP6  | **Sensor Integration** — DHT22 (Pi 1), PIR thread (Pi 2), USB speaker TTS                        | Premanand Aishwarya Shri                |
| WP7  | **Cross-Camera Child Watchdog** — HTTP polling, fault tolerance, missing-person alert            | Lim Bing Xian, Premanand Aishwarya Shri |
| WP8  | **Web Dashboard** — Flask MJPEG server, `/person_status` API, dark-theme UI                      | Boo Wai Yee Terry                       |
| WP9  | **Configuration & Deployment** — Centralised `settings.py`, `.env`, threading, graceful shutdown | Sidharth Vinod, Lim Bing Xian           |
| WP10 | **Documentation & Testing** — README, setup guide, hardware justification, test scenarios        | All members                             |

---

## Setup & Installation

### Prerequisites
- 2 X Raspberry Pi 5 (8 GB), Raspberry Pi OS 64-bit (Bookworm)
- Python 3.13, Logitech USB webcam
- Pi 1 additionally: DHT22 on GPIO 26, USB speaker
- Pi 2 additionally: PIR sensor on GPIO 26
- Telegram bot token from [@BotFather](https://t.me/BotFather)

### Pi 1

```bash
cd pi1
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, MODEL_PATH
python main.py
```
### Pi 2

```bash
cd pi2
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, MODEL_PATH, PI1_STATUS_URL
python main.py
```
### General
> The YOLO model files are not stored in this repository. Run the helper script once from the repo root to download and convert them automatically:
> ```bash
> python -m venv env
> source env/bin/activate
> pip install ultralytics
> python download_model.py
> ```
> This downloads `yolo26n-pose.pt` via Ultralytics, exports it to ONNX (imgsz=320), and saves both files to `pi1/models/` and `pi2/models/`.

### DHT22 Wiring (Pi 1)
```
VCC  → Pi 3.3V (Pin 1)
DATA → GPIO 26  (Pin 37, with 10 kΩ pull-up to 3.3V)
GND  → GND      (Pin 39)
```

### PIR Wiring (Pi 2)
```
VCC → Pi 5V   (Pin 2)
GND → GND     (Pin 6)
OUT → GPIO 26 (Pin 37)
```


## Key Configuration (`config/settings.py`)
The configuration values can be seen from each PI's specific settings, the default settings are the ones that are recommended after completing our testing. 

## Security & Privacy

- **Local inference** — no video or audio leaves the device for processing
- **Encrypted alerts** — Telegram Bot API uses TLS; only the configured `CHAT_ID` receives alerts
- **Credential management** — tokens stored in `.env` (excluded from version control via `.gitignore`)
- **LAN-only dashboard** — Flask servers are not exposed to the internet; use a VPN for remote access
- **Model files** — not stored in this repository; distributed separately

---

## Testing & Validation

| Area                | Method                                      | Result                                                    |
| ------------------- | ------------------------------------------- | --------------------------------------------------------- |
| Fall detection      | Staged falls at 1.5–3 m; 30-min walkthrough | No false alerts; alert fires within 3.5–4.5 s             |
| Fight detection     | Simulated aggression; threshold sweep       | Laugh-suppression rule blocks all play-fight false alerts |
| Audio detection     | Curated clips played at 70 dB, 1 m          | ≥ 80% correct labels per event class                      |
| Child watchdog      | Both cameras occluded for 65 s              | Alert fires at 60–70 s; fault tolerance confirmed         |
| PIR wake/sleep      | Walk-in / walk-out tests                    | Camera pauses within 25 s idle; wakes < 600 ms            |
| Network resilience  | Pi 1 disconnected during watchdog poll      | Graceful degradation after 3 failures; no crash           |
| Sustained operation | 24-hour continuous run                      | No memory leaks, no thread crashes                        |

---

## License

Licensed under the [Apache License 2.0](LICENSE).
