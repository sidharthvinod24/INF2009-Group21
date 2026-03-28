"""
config/settings.py
==================
Central configuration for Pi 2 Monitor.
All tunable parameters live here — no magic numbers scattered in code.

Pi 2 differences from Pi 1:
  - Has a PIR sensor on GPIO 26 — camera starts paused, PIR wakes it
  - Has a ChildWatchdogThread that polls Pi 1's /person_status
  - Runs on port 5000 (Pi 1 uses 5001)
  - Has audio detection (same as Pi 1)

Credentials are read from environment variables first:
  export TELEGRAM_TOKEN=<your_token>
  export TELEGRAM_CHAT_ID=<your_chat_id>
  export PI1_STATUS_URL=http://<pi1-ip>:5001/person_status
"""

import os
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root (or any parent directory)
# ═════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════════════════════════════════════

# Your Telegram bot token from @BotFather.
TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN',   '')

# The chat/group ID to send alerts to. Get it from @userinfobot.
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Minimum seconds before the SAME alert type fires again for the SAME person/pair.
# e.g. a second "FALL DETECTED" for Person 1 won't fire until 30s after the first.
# Higher → fewer repeated alerts for the same event.
# Lower  → more frequent repeat alerts (risk of spam).
TELEGRAM_COOLDOWN = 30

# Minimum seconds between ANY two alerts regardless of type or person.
# Acts as a global rate limiter — even different events are throttled.
# Higher → quieter overall, may miss rapid back-to-back events.
# Lower  → more responsive but risks Telegram flood-banning the bot.
TELEGRAM_GLOBAL_COOLDOWN = 15

# Minimum seconds before a FIGHT alert fires again for the same pair.
# Separate from TELEGRAM_COOLDOWN because fights produce noisy repeated frames.
# Higher → fewer fight alerts per incident.
# Lower  → more granular fight alerting (higher spam risk).
TELEGRAM_FIGHT_COOLDOWN = 45


# ═════════════════════════════════════════════════════════════════════════════
# ALERT GIF CLIP SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

# How many seconds of footage to stitch into the alert GIF.
# Higher → more context in the GIF, but larger file size and slower to send.
# Lower  → faster to send but may miss the build-up to the event.
ALERT_CLIP_SECONDS = 6

# Frame rate of the output GIF (not the capture rate).
# Higher → smoother GIF, larger file size.
# Lower  → choppier but smaller and faster to upload to Telegram.
ALERT_CLIP_FPS = 4

# Scale factor applied to each frame before building the GIF.
# 0.4 = 40% of original resolution.
# Higher (e.g. 0.8) → clearer GIF, much larger file.
# Lower  (e.g. 0.2) → tiny file, may be too blurry to be useful.
ALERT_CLIP_SCALE = 0.4

# Seconds to wait after a fight is confirmed before capturing the GIF clip.
# Gives time to capture the aftermath (actual blows) not just the build-up.
# Higher → captures more of the fight aftermath.
# Lower  → captures the moment of detection but less action may be visible.
FIGHT_GIF_DELAY = 3


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA
# ═════════════════════════════════════════════════════════════════════════════

# USB camera device index passed to cv2.VideoCapture().
# 0 = first USB camera, 1 = second, etc.
# Override via environment: CAMERA_INDEX=1 python main.py
CAMERA_SOURCE = int(os.getenv('CAMERA_INDEX', 0))

# Requested capture resolution. Driver will use the closest supported mode.
# Higher → more detail for detection, higher CPU load for YOLO.
# Lower  → faster inference, less spatial detail.
CAMERA_WIDTH  = 480
CAMERA_HEIGHT = 360

# Requested capture frame rate. Actual FPS depends on camera hardware.
# Higher → smoother video and more inference frames available.
# Lower  → reduces USB bandwidth and CPU load.
CAMERA_FPS = 30


# ═════════════════════════════════════════════════════════════════════════════
# FLASK / WEB SERVER
# ═════════════════════════════════════════════════════════════════════════════

# Port for the Flask MJPEG stream.
# Pi 2 uses 5000. Pi 1 uses 5001. Keep different to avoid LAN conflicts.
# Override via environment: WEB_PORT=8080 python main.py
WEB_PORT = int(os.getenv('WEB_PORT', 5000))


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

# YOLO detection confidence threshold. Detections below this are ignored.
# Higher (e.g. 0.6) → fewer false-positive skeletons, may miss real people.
# Lower  (e.g. 0.2) → detects more people, more ghost/noisy keypoints.
CONF_THRESHOLD = 0.4

# How many frames to SKIP between YOLO inference calls.
# 0 = run YOLO on every frame (highest CPU usage).
# 1 = run YOLO every 2nd frame (~2x CPU saving, slight detection lag).
# 2 = run YOLO every 3rd frame.
# Skipped frames still stream — they reuse the last annotated overlay.
# Recommended for Pi 5: 1. For weaker hardware: 2.
INFERENCE_SKIP_FRAMES = 1

# Absolute path to the YOLOv8 pose model file inside the models/ folder.
# Supports .pt (PyTorch) or .onnx (faster CPU inference on Pi).
# Override via environment: MODEL_PATH=/path/to/model.onnx python main.py
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'models', 'yolo26n-pose.onnx'))


# ═════════════════════════════════════════════════════════════════════════════
# PERSON TRACKING
# ═════════════════════════════════════════════════════════════════════════════

# Seconds of no detection before a person's track is deleted.
# Higher → track survives brief occlusions (person walks behind furniture).
# Lower  → stale tracks cleaned up faster, less memory/CPU overhead.
PERSON_TIMEOUT = 3.0

# Maximum pixel distance between a detected centroid and an existing track
# centroid for them to be considered the same person.
# Higher → more forgiving re-ID across fast movement or large frame jumps.
# Lower  → stricter matching, may create duplicate IDs for fast-moving people.
CENTROID_MATCH_DIST = 80


# ═════════════════════════════════════════════════════════════════════════════
# PIR SENSOR
# ═════════════════════════════════════════════════════════════════════════════

# Master switch for PIR-driven camera wake/sleep.
# True  → camera starts paused, PIR wakes it on motion (saves CPU).
# False → camera runs continuously (same behaviour as Pi 1).
ENABLE_PIR = True

# GPIO BCM pin number the PIR sensor data line is connected to.
# Default is GPIO 26. Change if you have wired it to a different pin.
PIR_GPIO_PIN = 26

# Seconds of no person detected by YOLO before the camera is paused.
# The idle timer starts after the last YOLO-confirmed person disappears.
# Higher → camera stays on longer after person leaves (more coverage).
# Lower  → camera sleeps faster (saves more CPU but may miss quick re-entry).
NO_PERSON_IDLE_SEC = 25.0

# Minimum seconds between camera turning OFF and PIR being allowed to
# wake it again. Prevents rapid on/off cycling if the PIR is noisy.
# Higher → more stable, longer pause before PIR can re-trigger.
# Lower  → more responsive, PIR can re-trigger sooner after sleep.
PIR_REARM_COOLDOWN_SEC = 8.0

# Seconds after PIR triggers during which a person is assumed to be present.
# Acts as a grace window before the idle timer starts counting.
# Higher → more time assumed before idle timer kicks in.
# Lower  → idle timer starts sooner after PIR trigger.
PIR_TRIGGERED_WINDOW_SEC = 6.0


# ═════════════════════════════════════════════════════════════════════════════
# CHILD WATCHDOG
# ═════════════════════════════════════════════════════════════════════════════

# Full URL to Pi 1's /person_status JSON endpoint.
# Pi 2 polls this every CHILD_WATCHDOG_POLL_SEC seconds to get Pi 1's
# last person-seen timestamp.
# Set via environment: PI1_STATUS_URL=http://192.168.1.x:5001/person_status
PI1_STATUS_URL = os.getenv('PI1_STATUS_URL', 'http://10.93.156.188:5001/person_status')

# Seconds BOTH cameras must see nobody before the child-missing alert fires.
# Both Pi 1 and Pi 2 must individually exceed this threshold.
# Higher → less sensitive, requires a longer absence before alerting.
# Lower  → alerts sooner (risk of false alarms during brief absences).
CHILD_MISSING_SECONDS = 60

# How often (seconds) Pi 2 polls Pi 1 for its person-seen status.
# Higher → less network traffic but slower to detect Pi 1 going offline.
# Lower  → more responsive but generates more LAN traffic.
CHILD_WATCHDOG_POLL_SEC = 10

# Minimum seconds between repeated child-missing Telegram alerts.
# Prevents spam if the child remains unseen for a long time.
# Higher → less repeat spam. Lower → more frequent reminders.
CHILD_ALERT_COOLDOWN = 120

# Number of consecutive failed polls to Pi 1 before the watchdog alert
# is suppressed. Prevents false alarms when Pi 1 is rebooting or offline.
# Higher → more tolerant of Pi 1 being offline before suppressing.
# Lower  → suppresses alert sooner when Pi 1 is unreachable.
CHILD_PI1_OFFLINE_TOLERANCE = 3


# ═════════════════════════════════════════════════════════════════════════════
# FALL DETECTION
# ═════════════════════════════════════════════════════════════════════════════

# Seconds a person must remain in a fallen pose before "FALL DETECTED" fires.
# Higher → fewer false positives (e.g. someone bending over).
# Lower  → faster detection but more false positives.
FALL_CONFIRM_SECONDS = 3

# Seconds a person must remain horizontal before being classified "Lying Down"
# (gradual, intentional pose — not a fall).
# Higher → slower to classify intentional lying, reduces false positives.
# Lower  → quicker lying-down classification.
LYING_DOWN_SECONDS = 4

# dx/dy ratio threshold for the spine vector to be considered horizontal.
# A ratio > 1.0 means the spine is wider than it is tall (lying flat).
# Higher (e.g. 1.2) → only flags very flat poses as horizontal.
# Lower  (e.g. 0.7) → flags shallower angles, more sensitive but noisier.
SPINE_HORIZONTAL_RATIO = 0.9

# Shoulder-hip-knee angle (degrees) below which the torso is considered fallen.
# At 180 degrees the person is fully upright.
# Higher (e.g. 80) → detects shallower falls, more sensitive.
# Lower  (e.g. 40) → only detects very acute collapses.
TORSO_ANGLE_THRESHOLD = 60

# Hip keypoint Y position as a fraction of frame height.
# If the hip is below this line it is considered "near ground".
# Higher (e.g. 0.7) → only flags hips very close to the bottom of frame.
# Lower  (e.g. 0.3) → flags hips as "near ground" even when mid-frame.
HIP_GROUND_RATIO = 0.5

# Number of spine angle samples kept in the rolling buffer per person.
# Used to compute angle delta (rate of change) for fall path classification.
# Higher → smoother delta estimate, slower to react to sudden changes.
# Lower  → more reactive to sudden angle changes, noisier.
ANGLE_BUFFER_SIZE = 20

# Minimum angle delta (degrees) to classify the transition as a FALL
# (rapid drop) rather than someone lying down slowly.
# Higher → only very sudden drops classified as falls (less sensitive).
# Lower  → slower transitions also classified as falls (more false positives).
FALL_ENTRY_DELTA = 35

# Secondary angle delta threshold for ambiguous cases.
# Works the same way as FALL_ENTRY_DELTA.
# Higher → stricter, fewer ambiguous cases classified as falls.
# Lower  → more generous, ambiguous poses lean toward fall classification.
FALL_ANGLE_DELTA_THRESHOLD = 25

# Number of keypoint snapshots to compare when checking if a person is motionless.
# Higher → less sensitive to micro-movements (breathing, trembling).
# Lower  → detects even small movements as "still moving".
MOTIONLESS_SAMPLE_FRAMES = 8

# Maximum mean keypoint displacement (pixels) between snapshots to consider
# a LYING person motionless.
# Higher → more tolerant of movement (e.g. breathing shifts keypoints).
# Lower  → stricter — any small movement resets the motionless timer.
MOTIONLESS_THRESHOLD_LYING = 4.0

# Same as above but for a FALLEN person (not just lying).
# Slightly higher than lying threshold because fall victims may still shudder.
# Higher → more tolerant of post-fall movement.
# Lower  → strict — person must be very still to trigger motionless alert.
MOTIONLESS_THRESHOLD_FALL = 6.0

# Seconds a LYING person must be motionless to trigger a "Motionless" alert.
# Higher → only alerts after prolonged stillness (fewer false alarms).
# Lower  → alerts quickly (more sensitive but may false-alarm on napping).
MOTIONLESS_SECONDS_LYING = 20

# Seconds a FALLEN person must be motionless to trigger "MOTIONLESS AFTER FALL".
# Shorter than lying because post-fall immobility is more urgent.
# Higher → waits longer before escalating (more conservative).
# Lower  → escalates faster (more aggressive monitoring).
MOTIONLESS_SECONDS_FALL = 15

# Seconds between repeated motionless alerts for the same person.
# 0 = alert only once per motionless episode.
# Higher → less repeat spam. Lower → more frequent reminders.
MOTIONLESS_ALERT_REPEAT = 60


# ═════════════════════════════════════════════════════════════════════════════
# FIGHT DETECTION
# ═════════════════════════════════════════════════════════════════════════════

# Minimum confidence that two people are at the same depth before scoring.
# Prevents scoring people at very different distances from the camera.
# Higher → stricter depth check, may skip valid nearby pairs.
# Lower  → more permissive, may score depth-separated pairs (false positives).
DEPTH_CONFIDENCE_THRESHOLD = 0.50

# Maximum bbox height ratio between two people for depth confidence.
# If one person's bbox is 1.8x taller than the other, they are likely at
# different depths and should not be scored.
# Higher → allows bigger size differences (more permissive).
# Lower  → rejects pairs with smaller size differences (stricter).
MAX_HEIGHT_RATIO = 1.8

# Minimum IoU (overlap) between two person bboxes to consider them close enough.
# Below this they are too far apart for a confrontation.
# Higher → requires more overlap before scoring begins.
# Lower  → starts scoring even with minimal overlap.
MIN_BBOX_IOU = 0.05

# Maximum IoU allowed. Above this the bboxes overlap so much they are likely
# front-to-back rather than side-by-side (depth ambiguity).
# Higher → tolerates very high overlap.
# Lower  → rejects heavily overlapping bboxes (stricter).
MAX_BBOX_IOU = 0.45

# Minimum lateral separation between bbox centres normalised by avg bbox width.
# Prevents scoring when two people are stacked vertically.
# Higher → requires people to be more side-by-side.
# Lower  → also scores vertically stacked people.
MIN_LATERAL_SEPARATION = 0.20

# Maximum lateral separation. Beyond this the pair is too far apart laterally.
# Higher → scores pairs further apart. Lower → only scores close pairs.
MAX_LATERAL_SEPARATION = 2.50

# Minimum normalised shoulder width ratio between the two people.
# Ensures both people are facing the camera at a similar angle.
# Higher → stricter facing requirement. Lower → allows more profile angles.
MIN_SHOULDER_WIDTH_RATIO = 0.35

# Maximum normalised centre-to-centre distance for fight scoring to proceed.
# Higher → scores pairs further apart (more sensitive, more false positives).
# Lower  → only scores pairs very close together (stricter).
PROXIMITY_THRESHOLD = 1.4

# Minimum pixel displacement per frame for a wrist keypoint to count as
# a fast/aggressive wrist movement (punching signal S1).
# Higher → only very fast wrist movements score.
# Lower  → slower arm movements also score (more sensitive).
WRIST_VELOCITY_THRESHOLD = 18

# Same as above for elbow keypoints (secondary arm velocity signal S1).
# Elbows move less than wrists during a punch so threshold is lower.
ELBOW_VELOCITY_THRESHOLD = 14

# Minimum pixel displacement per frame for knee/ankle keypoints to count
# as a kick movement (kick signal S6).
# Higher → only fast kicks score. Lower → slower leg movements also score.
KNEE_VELOCITY_THRESHOLD = 16

# Number of arm angle samples kept in the rolling buffer per person per pair.
# Used to compute pose instability (arm flailing variance) for signal S4.
# Higher → more stable variance estimate, slower to respond.
# Lower  → more reactive, noisier.
MOTION_BUFFER_SIZE = 20

# Minimum variance of arm joint angles to flag pose instability (S4).
# A high variance means the arms are moving erratically — a fight indicator.
# Higher → only very erratic arm movement scores.
# Lower  → moderate arm movement also scores (more false positives).
POSE_INSTABILITY_THRESH = 180

# Score threshold for "Interaction" state (people near each other, some movement).
# Scores between POSSIBLE_FIGHT_SCORE and FIGHT_SCORE_THRESHOLD = "Possible fight".
# Higher → harder to enter Interaction state.
# Lower  → almost any proximity triggers Interaction.
POSSIBLE_FIGHT_SCORE = 3

# Score threshold (per frame, after smoothing) to start the fight confirmation timer.
# Score range is 0-14 across all 11 signals.
# Higher → harder to confirm a fight (fewer false positives, may miss real fights).
# Lower  → easier to confirm (more sensitive, higher false positive rate).
FIGHT_SCORE_THRESHOLD = 4

# EMA smoothing factor applied to the raw fight score each frame.
# score = alpha * prev_score + (1 - alpha) * raw_score
# Higher (e.g. 0.7) → very smooth, slow to react to sudden changes.
# Lower  (e.g. 0.2) → reacts quickly to each frame, more jittery.
SCORE_SMOOTHING_ALPHA = 0.4

# Seconds the score must stay above FIGHT_SCORE_THRESHOLD (with window ratio met)
# before "FIGHT DETECTED" is declared. Prevents reacting to a single bad frame.
# Higher → more confident detections, slower response.
# Lower  → faster detection, more false positives.
FIGHT_CONFIRM_SECONDS = 1.0

# Length of the rolling time window (seconds) used to check if scores are
# consistently high. Only frames within this window are considered.
# Higher → looks at a longer history, more stable but slower to reset.
# Lower  → shorter memory, reacts to recent frames only.
FIGHT_WINDOW_SECONDS = 2.0

# Fraction of frames within FIGHT_WINDOW_SECONDS that must have score >=
# FIGHT_SCORE_THRESHOLD to confirm a fight.
# Higher (e.g. 0.7) → most frames must be high-score (strict).
# Lower  (e.g. 0.3) → a minority of high-score frames is enough (loose).
FIGHT_WINDOW_RATIO = 0.35

# Maximum head-to-head distance (normalised by avg bbox height) for the
# head proximity signal S5. Below this ratio = heads are very close.
# Higher → flags head proximity at greater distances.
# Lower  → only flags when heads are very close (grappling/headlock range).
HEAD_PROXIMITY_RATIO = 0.5

# Whether to check for ankle/foot invasion into opponent's lower-body bbox
# as part of the kick signal S6. Adds specificity to kick detection.
# True  → more accurate kick detection.
# False → disable if causing false positives (e.g. people walking past each other).
ANKLE_INVASION_ENABLED = True

# Multiplier for detecting a strike burst S7. A wrist velocity that exceeds
# ACCELERATION_MULTIPLIER x previous velocity counts as a burst.
# Higher → only very sudden acceleration scores. Lower → gradual speed-ups also score.
ACCELERATION_MULTIPLIER = 2.0

# Minimum torso (hip midpoint) displacement per frame to count as a torso push S8.
# Higher → only hard pushes score. Lower → small torso shifts also score.
TORSO_VELOCITY_THRESHOLD = 12

# Maximum torso velocity for the OTHER person to still count as a push victim S8.
# If both torsos are moving fast it is mutual movement, not a push.
# Higher → more permissive. Lower → stricter one-sided push detection.
TORSO_ACCEL_THRESHOLD = 8

# Bonus score added when BOTH people have fast wrists simultaneously S9.
# Higher → mutual exchanges score more heavily. 0 = disable.
MUTUAL_AGGRESSION_BONUS = 1

# Minimum bbox IoU between two people to start counting as a grapple S10.
# Higher → only very overlapping bboxes count. Lower → looser contact also counts.
GRAPPLE_IOU_THRESHOLD = 0.35

# Minimum increase in IoU per frame to score a grapple event S10.
# Detects two people moving closer / locking together.
# Higher → only rapid closure scores. Lower → slow advances also score.
GRAPPLE_IOU_DELTA = 0.15

# Maximum wrist velocity for the VICTIM in the asymmetric aggression signal S11.
# If one person's wrist is invading AND the other is relatively still, it scores.
# Higher → victim can be moving more and still score.
# Lower  → victim must be almost stationary to score.
VICTIM_STATIONARY_THRESHOLD = 6


# ═════════════════════════════════════════════════════════════════════════════
# AUDIO DETECTION
# ═════════════════════════════════════════════════════════════════════════════

# System microphone device index passed to pyaudio.
# None = use the system default microphone.
# Run python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
# to list available devices.
AUDIO_DEVICE_INDEX = None

# Microphone sample rate in Hz. YAMNet requires 16000 Hz — do not change.
AUDIO_SAMPLE_RATE = 16000

# Number of audio frames per read chunk. At 16000 Hz: 1024 frames ≈ 64ms per chunk.
# Higher → less CPU, more latency. Lower → faster detection, higher CPU.
AUDIO_CHUNK_SIZE = 1024

# Number of audio channels. 1 = mono (required for YAMNet).
AUDIO_CHANNELS = 1

# Minimum seconds between recording the same label in AudioState.
# Prevents the same event being recorded dozens of times per second.
# Higher → less sensitive to repeated detections. Lower → records every occurrence.
AUDIO_COOLDOWN_SEC = 8.0

# ── Standalone Telegram alert flags ──────────────────────────────────────────
# Controls which audio labels fire their OWN Telegram alert.
# Scream and cry are False — they only enrich fall/fight alerts, never send alone.

# True  → scream fires its own Telegram alert.
# False → scream only enriches fall/fight video alerts (recommended).
AUDIO_ALERT_SCREAM = False

# True  → cry fires its own Telegram alert.
# False → cry only enriches fall/fight video alerts (recommended).
AUDIO_ALERT_CRY = False

# Laugh never alerts — used only to SUPPRESS false fight alerts.
AUDIO_ALERT_LAUGH = False

# True  → impact (bang, thud, crash) fires its own Telegram alert.
# False → disable standalone impact alerts.
AUDIO_ALERT_IMPACT = True

# Minimum seconds between standalone audio Telegram alerts for the same label.
# Higher → less alert spam for repeated impacts. Lower → more frequent alerts.
AUDIO_ALERT_COOLDOWN = 20

# ── Cross-modal suppression / enrichment windows ─────────────────────────────

# If laugh was detected within this many seconds of a FIGHT alert,
# the fight alert is SUPPRESSED (children playing, not fighting).
# Higher → laugh suppresses fight alerts for longer.
# Lower  → fight alert fires sooner after laugh detection stops.
LAUGH_SUPPRESS_WINDOW = 8.0

# If cry was detected within this many seconds of a FALL or FIGHT alert,
# the alert message is enriched with "+ CRY".
# Higher → wider window, more likely to catch a cry before detection.
# Lower  → stricter — cry must be very recent to be included.
CRY_ENRICH_WINDOW = 6.0

# Same as above but for scream detection enrichment.
SCREAM_ENRICH_WINDOW = 6.0

# Seconds to wait after a fall is confirmed before sending the Telegram alert.
# Gives the audio detector time to pick up a cry/scream so both arrive in
# the SAME message. 3s is usually enough — child cries within 1-2s of a fall.
# Higher → more time to capture audio context, alert arrives later.
# Lower  → faster alert but may miss the cry/scream that follows.
FALL_AUDIO_WAIT = 3.0

# ── Heuristic detector thresholds ────────────────────────────────────────────

# Minimum RMS amplitude (0.0 to 1.0) to count as an impact frame.
# Higher → only very loud impacts trigger. Lower → more sensitive.
IMPACT_RMS_THRESH = 0.45

# Minimum duration (ms) of sustained high RMS to count as an impact.
# Higher → requires longer loud burst. Lower → more sensitive to short sounds.
IMPACT_DURATION_MS = 80.0

# Maximum time (ms) after the RMS peak before it must drop back down.
# Impacts have a fast rise AND fast drop.
# Higher → tolerates longer decay (e.g. reverb). Lower → only sharp impacts.
IMPACT_DROP_MS = 200.0

# Minimum RMS amplitude required for scream detection.
# Higher → only very loud screams. Lower → quieter vocalisations also detected.
SCREAM_RMS_THRESH = 0.35

# Frequency band (Hz) used for scream spectral flux analysis.
# Children's screams peak in 900-4000 Hz range.
SCREAM_LOW_HZ  = 900.0
SCREAM_HIGH_HZ = 4000.0

# Minimum spectral flux in the scream band to count as a scream frame.
# Higher → only sharp scream onsets score. Lower → gentler rising energy also scores.
SCREAM_FLUX_THRESH = 0.25

# Minimum duration (ms) of sustained scream flux before confirming.
# Higher → requires longer sustained scream. Lower → shorter screams also trigger.
SCREAM_DURATION_MS = 400.0

# Frequency band (Hz) for cry detection. Children's cries peak in 350-900 Hz.
CRY_LOW_HZ  = 350.0
CRY_HIGH_HZ = 900.0

# Minimum spectral flux in the cry band to count as a cry frame.
CRY_FLUX_THRESH = 0.28

# Minimum duration (ms) of sustained cry flux before confirming.
CRY_DURATION_MS = 400.0

# Frequency band (Hz) for laugh detection. Laughter peaks in 500-2000 Hz.
LAUGH_LOW_HZ  = 500.0
LAUGH_HIGH_HZ = 2000.0

# Minimum spectral flux in the laugh band to count as a laugh frame.
# Lower than cry/scream because laughter has more variable energy.
LAUGH_FLUX_THRESH = 0.20

# Minimum duration (ms) of sustained laugh flux before confirming.
LAUGH_DURATION_MS = 400.0

# ── YAMNet settings ───────────────────────────────────────────────────────────

# How many audio chunks to accumulate before sending to YAMNet.
# At 1024 frames/chunk and 16000 Hz: 8 chunks ≈ 512ms of audio per call.
# Higher → more context per call (more accurate), longer delay.
# Lower  → faster calls but less context (less accurate).
YAMNET_EVERY_N_CHUNKS = 8

# Minimum YAMNet class score (0.0 to 1.0) to accept a detection.
# Higher → only very confident YAMNet detections count.
# Lower  → more detections but higher false positive rate.
YAMNET_CONF_THRESH = 0.20
