"""
config/settings.py
==================
Central configuration for Pi 1 Monitor.
All tunable parameters live here — no magic numbers scattered in code.

Credentials are read from environment variables first; hardcoded fallbacks
are provided for development only. In production, set env vars:
  export TELEGRAM_TOKEN=<your_token>
  export TELEGRAM_CHAT_ID=<your_chat_id>
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root (or any parent directory)

# ─── Telegram ────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN        = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID      = os.getenv('TELEGRAM_CHAT_ID', '')
TELEGRAM_COOLDOWN     = 30    # per-person/pair cooldown for same alert type (s)
TELEGRAM_GLOBAL_COOLDOWN = 15 # minimum seconds between ANY two alerts
TELEGRAM_FIGHT_COOLDOWN  = 45 # fight-specific cooldown (fights are noisy)

# ─── Alert GIF clip settings ─────────────────────────────────────────────────

ALERT_CLIP_SECONDS = 6    # rolling buffer length to include in GIF
ALERT_CLIP_FPS     = 4    # GIF playback FPS (lower = smaller file)
ALERT_CLIP_SCALE   = 0.4  # downscale factor before encoding GIF
FIGHT_GIF_DELAY    = 3    # seconds after fight confirmed before grabbing GIF

# ─── Camera ──────────────────────────────────────────────────────────────────

# USB camera index — 0 is the default/first connected USB camera.
# Change to 1, 2, etc. if you have multiple cameras connected.
CAMERA_SOURCE      = int(os.getenv('CAMERA_INDEX', 0))
CAMERA_WIDTH       = 640
CAMERA_HEIGHT      = 480
CAMERA_FPS         = 30

# ─── Flask / Web server ──────────────────────────────────────────────────────

WEB_PORT = int(os.getenv('WEB_PORT', 5001))

# ─── Inference ───────────────────────────────────────────────────────────────

CONF_THRESHOLD      = 0.4
INFERENCE_SKIP_FRAMES = 1   # 0=every frame, 1=every 2nd, 2=every 3rd
MODEL_PATH          = os.getenv('MODEL_PATH', '')

# ─── Person tracking ─────────────────────────────────────────────────────────

PERSON_TIMEOUT      = 3.0   # seconds before a person track is pruned
CENTROID_MATCH_DIST = 80    # max pixel distance for centroid re-ID

# ─── Fall detection ──────────────────────────────────────────────────────────

FALL_CONFIRM_SECONDS       = 3
LYING_DOWN_SECONDS         = 4
SPINE_HORIZONTAL_RATIO     = 0.9
TORSO_ANGLE_THRESHOLD      = 60
HIP_GROUND_RATIO           = 0.5
ANGLE_BUFFER_SIZE          = 20
FALL_ENTRY_DELTA           = 35
FALL_ANGLE_DELTA_THRESHOLD = 25

MOTIONLESS_SAMPLE_FRAMES   = 8
MOTIONLESS_THRESHOLD_LYING = 4.0
MOTIONLESS_THRESHOLD_FALL  = 6.0
MOTIONLESS_SECONDS_LYING   = 20
MOTIONLESS_SECONDS_FALL    = 15
MOTIONLESS_ALERT_REPEAT    = 60

# ─── Fight detection ─────────────────────────────────────────────────────────

DEPTH_CONFIDENCE_THRESHOLD = 0.50
MAX_HEIGHT_RATIO           = 1.8
MIN_BBOX_IOU               = 0.05
MAX_BBOX_IOU               = 0.45
MIN_LATERAL_SEPARATION     = 0.20
MAX_LATERAL_SEPARATION     = 2.50
MIN_SHOULDER_WIDTH_RATIO   = 0.35
PROXIMITY_THRESHOLD        = 1.4

WRIST_VELOCITY_THRESHOLD   = 18
ELBOW_VELOCITY_THRESHOLD   = 14
KNEE_VELOCITY_THRESHOLD    = 16
MOTION_BUFFER_SIZE         = 20
POSE_INSTABILITY_THRESH    = 180

POSSIBLE_FIGHT_SCORE       = 3
FIGHT_SCORE_THRESHOLD      = 6
SCORE_SMOOTHING_ALPHA      = 0.4

FIGHT_CONFIRM_SECONDS      = 4.0
FIGHT_WINDOW_SECONDS       = 4.0
FIGHT_WINDOW_RATIO         = 0.55
HEAD_PROXIMITY_RATIO       = 0.5

ANKLE_INVASION_ENABLED     = True
ACCELERATION_MULTIPLIER    = 2.0
TORSO_VELOCITY_THRESHOLD   = 12
TORSO_ACCEL_THRESHOLD      = 8
MUTUAL_AGGRESSION_BONUS    = 1
GRAPPLE_IOU_THRESHOLD      = 0.35
GRAPPLE_IOU_DELTA          = 0.15
VICTIM_STATIONARY_THRESHOLD = 6


# ═════════════════════════════════════════════════════════════════════════════
# AUDIO DETECTION
# ═════════════════════════════════════════════════════════════════════════════

# System microphone device index passed to pyaudio.
# None = use the system default microphone.
# Change to 0, 1, 2 etc. if you have multiple audio input devices.
# Run: python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
# to list available devices.
AUDIO_DEVICE_INDEX = None

# Microphone sample rate in Hz. YAMNet requires 16000 Hz — do not change.
AUDIO_SAMPLE_RATE  = 16000

# Number of audio frames per read chunk. Smaller = lower latency but more CPU.
# At 16000 Hz: 1024 frames ≈ 64ms per chunk.
# Higher → less CPU, more latency per detection.
# Lower  → faster detection, higher CPU and risk of buffer overflow.
AUDIO_CHUNK_SIZE   = 1024

# Number of audio channels. 1 = mono (required for YAMNet).
AUDIO_CHANNELS     = 1

# Minimum seconds between recording the SAME label in AudioState.
# Prevents the same event from being recorded dozens of times per second.
# Higher → less sensitive to repeated detections of same sound.
# Lower  → records every occurrence (may flood state with duplicates).
AUDIO_COOLDOWN_SEC = 8.0

# ── Standalone Telegram alert flags ──────────────────────────────────────────
# Controls which audio labels fire their OWN Telegram alert (independent of video).
# Scream and cry are intentionally False — they only ENRICH fall/fight alerts.
# Setting them True would cause duplicate messages (one from audio, one from video).

# True  → scream fires its own Telegram alert.
# False → scream only enriches fall/fight video alerts (recommended).
AUDIO_ALERT_SCREAM = False

# True  → cry fires its own Telegram alert.
# False → cry only enriches fall/fight video alerts (recommended).
AUDIO_ALERT_CRY    = False

# Laugh never alerts — it is used only to SUPPRESS false fight alerts.
AUDIO_ALERT_LAUGH  = False

# True  → impact (bang, thud, crash) fires its own Telegram alert.
# False → disable standalone impact alerts.
AUDIO_ALERT_IMPACT = True

# Minimum seconds between standalone audio Telegram alerts for the same label.
# Higher → less alert spam for repeated impacts.
# Lower  → more frequent alerts per impact event.
AUDIO_ALERT_COOLDOWN = 20

# ── Cross-modal suppression / enrichment windows ─────────────────────────────

# If laugh was detected within this many seconds of a FIGHT alert,
# the fight alert is SUPPRESSED (children playing, not fighting).
# Higher → laugh suppresses fight alerts for longer after the laugh ends.
# Lower  → fight alert fires sooner after laugh detection stops.
LAUGH_SUPPRESS_WINDOW = 8.0

# If cry was detected within this many seconds of a FALL or FIGHT alert,
# the alert message is enriched with "+ CRY".
# Higher → wider window, more likely to catch a cry that started before detection.
# Lower  → stricter — cry must be very recent to be included.
CRY_ENRICH_WINDOW = 6.0

# Same as above but for scream detection enrichment.
SCREAM_ENRICH_WINDOW = 6.0

# Seconds to WAIT after a fall is confirmed before sending the Telegram alert.
# This gives the audio detector time to pick up a cry/scream so both arrive
# in the SAME message rather than as separate messages.
# 3s is usually enough — a child typically cries within 1-2s of a fall.
# Higher → more time to capture audio context, but alert arrives later.
# Lower  → faster alert but may miss the cry/scream that follows.
FALL_AUDIO_WAIT = 3.0

# ── Heuristic detector thresholds ────────────────────────────────────────────

# Minimum RMS amplitude to count as an impact frame.
# RMS range is 0.0 (silence) to 1.0 (maximum loudness).
# Higher → only very loud impacts trigger (ignores minor thuds).
# Lower  → more sensitive, may false-positive on loud speech or music.
IMPACT_RMS_THRESH  = 0.45

# Minimum duration (ms) of sustained high RMS to count as an impact.
# Higher → requires a longer loud burst (ignores very brief spikes).
# Lower  → more sensitive to short sharp sounds.
IMPACT_DURATION_MS = 80.0

# Maximum time (ms) after the RMS peak before it must drop back down.
# Impacts are characterised by a fast rise AND a fast drop.
# Higher → tolerates impacts with longer decay (e.g. reverb in a room).
# Lower  → only very sharp, clean impacts trigger.
IMPACT_DROP_MS     = 200.0

# Minimum RMS amplitude required for scream detection.
# Higher → only very loud screams detected.
# Lower  → quieter vocalisations also detected (more false positives).
SCREAM_RMS_THRESH  = 0.35

# Frequency band (Hz) used for scream spectral flux analysis.
# Children's screams peak in 900-4000 Hz range.
SCREAM_LOW_HZ      = 900.0
SCREAM_HIGH_HZ     = 4000.0

# Minimum spectral flux in the scream band to count as a scream frame.
# Higher → only sharp scream onsets score (stricter).
# Lower  → gentler rising energy in this band also scores (noisier).
SCREAM_FLUX_THRESH = 0.25

# Minimum duration (ms) of sustained scream flux before confirming.
# Higher → requires longer sustained scream (fewer false positives).
# Lower  → shorter screams also trigger.
SCREAM_DURATION_MS = 400.0

# Frequency band (Hz) for cry detection. Children's cries peak in 350-900 Hz.
CRY_LOW_HZ         = 350.0
CRY_HIGH_HZ        = 900.0

# Minimum spectral flux in the cry band to count as a cry frame.
CRY_FLUX_THRESH    = 0.28

# Minimum duration (ms) of sustained cry flux before confirming.
CRY_DURATION_MS    = 400.0

# Frequency band (Hz) for laugh detection. Laughter peaks in 500-2000 Hz.
LAUGH_LOW_HZ       = 500.0
LAUGH_HIGH_HZ      = 2000.0

# Minimum RMS level required before laugh flux is even considered.
# Prevents ambient noise at the right frequency from triggering laugh.
LAUGH_RMS_THRESH   = 0.05

# Minimum spectral flux in the laugh band to count as a laugh frame.
# Lower than cry/scream because laughter has more variable energy.
LAUGH_FLUX_THRESH  = 0.20

# Minimum duration (ms) of sustained laugh flux before confirming.
LAUGH_DURATION_MS  = 400.0

# ── YAMNet settings ───────────────────────────────────────────────────────────

# How many audio chunks to accumulate before sending to YAMNet for inference.
# At 1024 frames/chunk and 16000 Hz: 8 chunks ≈ 512ms of audio per YAMNet call.
# Higher → more audio context per call (more accurate), but longer delay.
# Lower  → faster YAMNet calls but less context (less accurate).
YAMNET_EVERY_N_CHUNKS = 8

# Minimum YAMNet class score to accept a detection.
# Score range is 0.0 to 1.0.
# Higher → only very confident YAMNet detections count.
# Lower  → more detections but higher false positive rate.
YAMNET_CONF_THRESH    = 0.20


# ═════════════════════════════════════════════════════════════════════════════
# TEMPERATURE SENSOR (DHT22 on GPIO 26)
# ═════════════════════════════════════════════════════════════════════════════

# GPIO BCM pin number the DHT22 data line is connected to.
# Default is GPIO 26. Change if wired differently.
DHT_GPIO_PIN = 26

# How often (seconds) to poll the DHT22 sensor in the background.
# DHT22 minimum interval is ~2s. We cache longer to reduce wear.
# Higher → less frequent reads, sensor lasts longer, data slightly stale.
# Lower  → fresher readings but more sensor stress (do not go below 3s).
TEMP_READ_INTERVAL = 10.0


# ═════════════════════════════════════════════════════════════════════════════
# SPEAKER (USB speaker via pyttsx3 / espeak)
# ═════════════════════════════════════════════════════════════════════════════

# Master switch for TTS announcements.
# True  → speaker announces fall/fight/missing-child events locally.
# False → silent operation (Telegram alerts still work).
SPEAKER_ENABLED = True

# TTS speech rate in words per minute.
# Higher (e.g. 200) → faster speech, harder to understand.
# Lower  (e.g. 100) → slower, clearer speech.
SPEAKER_RATE = 150

# TTS volume. Range 0.0 (silent) to 1.0 (maximum).
# Higher → louder. Lower → quieter.
SPEAKER_VOLUME = 0.6

# Maximum number of messages queued for the speaker.
# If the queue is full, oldest message is dropped.
# Higher → more messages buffered (may speak outdated alerts).
# Lower  → only the most recent alerts are spoken.
SPEAKER_QUEUE_SIZE = 3
