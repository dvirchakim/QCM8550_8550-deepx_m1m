"""
Edge-Art Interactive Silhouette Mural - Configuration
PRD v1.0.0 - Heterogeneous AI Showcase (QCS8550 + DX-M1)
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")
VIDEO_DIR   = os.path.join(ASSETS_DIR, "videos")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Hardware mode
# ---------------------------------------------------------------------------
# When True, simulated handlers are used. Auto-disabled on the board if the
# real DX-RT / QNN runtimes are detected by the handlers themselves.
SIMULATE_DEEPX   = True
SIMULATE_QUALCOMM = True

# ---------------------------------------------------------------------------
# UI Layout  (PRD section 5)
# ---------------------------------------------------------------------------
WINDOW_TITLE  = "IMDT QCS8550 + DEEPX HETEROGENEOUS AI PIPELINE"
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 800
HEADER_HEIGHT = 60
FOOTER_HEIGHT = 90   # touch style buttons
TARGET_DISPLAY_FPS = 60

# Color theme (high-contrast for booth visibility from 5 m)
COLOR_HEADER_BG = "#0B5FFF"
COLOR_HEADER_FG = "#FFFFFF"
COLOR_PANEL_BG  = "#0A0E1A"
COLOR_ACCENT    = "#00E5FF"
COLOR_OK        = "#00FF88"
COLOR_WARN      = "#FFB300"
COLOR_ERR       = "#FF3355"

# ---------------------------------------------------------------------------
# Camera pipeline  (PRD section 4.1)
# ---------------------------------------------------------------------------
CAMERA_1_INDEX = 0   # qtiqmmfsrc camera=0 on board / cv2 cam 0 on dev
CAMERA_2_INDEX = 1   # qtiqmmfsrc camera=1
CAPTURE_WIDTH  = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS    = 30

# ---------------------------------------------------------------------------
# DX-M1 pose inference  (PRD section 4.2)
# Reality: board ships with YOLOv5-Pose (17-KP COCO), not v8. We keep the
# PRD's 17-KP contract intact and reference the actual on-board artefacts.
# ---------------------------------------------------------------------------
BOARD_DXSTREAM_ROOT  = "/usr/share/dx-stream"
POSE_MODEL_DXNN      = f"{BOARD_DXSTREAM_ROOT}/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
POSE_CFG_DIR         = f"{BOARD_DXSTREAM_ROOT}/configs/Pose_Estimation/YOLOV5Pose640_1"
POSE_PREPROCESS_CFG  = f"{POSE_CFG_DIR}/preprocess_config.json"
POSE_INFERENCE_CFG   = f"{POSE_CFG_DIR}/inference_config.json"
POSE_POSTPROCESS_CFG = f"{POSE_CFG_DIR}/postprocess_config.json"
POSE_NUM_KEYPOINTS   = 17                    # COCO format
POSE_INPUT_SIZE      = 640                   # from preprocess_config.json
POSE_CONF_THRESHOLD  = 0.35
POSE_TARGET_LATENCY_MS = 5.0                 # PRD KPI

# ---------------------------------------------------------------------------
# MQTT pose extraction  (dxmsgconv -> dxmsgbroker -> mosquitto -> Python)
# ---------------------------------------------------------------------------
MQTT_HOST       = "127.0.0.1"
MQTT_PORT       = 1883
MQTT_TOPIC_POSE = "dxstream/pose"
MQTT_USERNAME   = "user"
MQTT_PASSWORD   = "1234"
MQTT_USE_TLS    = False    # broker_mqtt.cfg has TLS; disable for prototype
MSGCONV_CFG     = f"{BOARD_DXSTREAM_ROOT}/configs/msgconv_config.json"
MSGBROKER_CFG   = f"{BOARD_DXSTREAM_ROOT}/configs/broker_mqtt.cfg"

# COCO skeleton edges (17 keypoints)
SKELETON_EDGES = [
    (0, 1),  (0, 2),  (1, 3),  (2, 4),         # head
    (5, 6),  (5, 7),  (7, 9),  (6, 8),  (8, 10), # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
]

# ---------------------------------------------------------------------------
# Snapdragon generative pass  (PRD section 4.3)
# ---------------------------------------------------------------------------
SD_MODEL_DLC          = os.path.join(MODELS_DIR, "sd15_controlnet.dlc")
SD_INFERENCE_STEPS    = 8        # LCM / distilled, async pane
SD_OUTPUT_SIZE        = 512
SD_REFRESH_INTERVAL_S = 1.2      # background regeneration cadence

# Style presets - shown as touch buttons (PRD section 5.1)
STYLES = [
    {
        "id": "neon",
        "label": "NEON CYBERPUNK",
        "prompt": "neon cyberpunk hologram silhouette, glowing wireframe, dark city background, magenta and cyan lights, ultra detailed",
        "negative": "blurry, low quality, deformed",
        "color": (255, 0, 200),
    },
    {
        "id": "vangogh",
        "label": "VAN GOGH SKETCH",
        "prompt": "van gogh oil painting of a person, swirling brush strokes, vivid impasto, expressive yellows and blues",
        "negative": "photo realistic, smooth",
        "color": (255, 200, 60),
    },
    {
        "id": "comic",
        "label": "COMIC BOOK INK",
        "prompt": "comic book ink illustration, bold black outlines, halftone shading, dynamic action pose, vibrant primary colors",
        "negative": "photo, gradient",
        "color": (50, 180, 255),
    },
    {
        "id": "noir",
        "label": "COMIC NOIR",
        "prompt": "film noir black and white silhouette, dramatic chiaroscuro lighting, rain, smoke, 1940s detective scene",
        "negative": "color, bright",
        "color": (220, 220, 220),
    },
]
DEFAULT_STYLE_INDEX = 0

# ---------------------------------------------------------------------------
# Monitoring  (PRD section 6)
# ---------------------------------------------------------------------------
THERMAL_ZONE_PATH      = "/sys/class/thermal/thermal_zone1/temp"
POWER_SUPPLY_PATH      = "/sys/class/power_supply"   # fallback if no INA sensor
THERMAL_THROTTLE_TEMP  = 80.0
POWER_BUDGET_W         = 10.0    # PRD KPI - hard cap
MEMORY_FLUSH_INTERVAL  = 300
