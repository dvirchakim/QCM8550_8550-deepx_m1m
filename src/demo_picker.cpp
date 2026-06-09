/**
 * demo_picker.cpp  –  Touchscreen demo selector for QCS8550 + DeepX DX-M1
 *
 * Displays a 2×2 tile grid on the Wayland display; a tap launches the
 * corresponding systemd service (previous service is stopped first).
 *
 * No OpenCV dependency – uses raw pixel ops + GStreamer textoverlay for text.
 *
 * Cross-compile:
 *   aarch64-linux-gnu-g++ -O2 -std=c++17 -o demo_picker demo_picker.cpp \
 *     $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0) \
 *     -lpthread
 *
 * Deploy:
 *   adb push demo_picker /data/local/tmp/demo_picker
 *   adb shell chmod +x /data/local/tmp/demo_picker
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <fcntl.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

// ── Display resolution ────────────────────────────────────────────────────────
static constexpr int W = 1920;
static constexpr int H = 1080;

// ── Tile definitions ──────────────────────────────────────────────────────────
struct Tile {
    const char* label;      // large text
    const char* subtitle;   // small text (NPU info)
    const char* service;    // systemd unit name to start
    uint8_t     br, bg, bb; // BGR accent colour
};

static const std::array<Tile, 4> TILES = {{
    { "Edge Art",        "DeepX Pose  +  Qualcomm GenAI",      "edge-art.service",        210,  60, 180 },
    { "OEM Reference",   "Qualcomm  +  DeepX  Pipeline",       "imdt-deepx-demo.service",   0, 120, 180 },
    { "YOLO26 Parallel", "DeepX Det  ||  DeepX Seg",           "yolo26-parallel.service",  30, 160,  30 },
    { "CLIP  Demo",      "Qualcomm HTP  -  Semantic CLIP",     "clip-demo.service",         0, 140, 220 },
}};

// ── Service control ───────────────────────────────────────────────────────────
static std::string g_running_service;

static void stop_service(const std::string& svc) {
    if (svc.empty()) return;
    std::string cmd = "systemctl stop " + svc + " 2>/dev/null";
    system(cmd.c_str());
}

static void start_service(const std::string& svc) {
    std::string cmd = "systemctl start " + svc + " 2>/dev/null";
    system(cmd.c_str());
}

static void launch(int tile_idx) {
    stop_service(g_running_service);
    g_running_service = TILES[tile_idx].service;
    start_service(g_running_service);
}

// ── Raw pixel drawing (no OpenCV) ─────────────────────────────────────────────
static uint8_t g_frame[H * W * 3];  // BGR

static inline void fill_rect(int x, int y, int w, int h,
                              uint8_t b, uint8_t g, uint8_t r) {
    for (int row = y; row < y + h; ++row) {
        uint8_t* p = g_frame + (row * W + x) * 3;
        for (int col = 0; col < w; ++col) {
            p[0] = b; p[1] = g; p[2] = r;
            p += 3;
        }
    }
}

static inline void draw_border(int x, int y, int w, int h, int t,
                                uint8_t b, uint8_t g, uint8_t r) {
    fill_rect(x,         y,         w, t, b, g, r);   // top
    fill_rect(x,         y+h-t,     w, t, b, g, r);   // bottom
    fill_rect(x,         y,         t, h, b, g, r);   // left
    fill_rect(x+w-t,     y,         t, h, b, g, r);   // right
}

// ── UI rendering ──────────────────────────────────────────────────────────────
static void render(int highlighted) {
    // Background
    memset(g_frame, 20, sizeof(g_frame));

    const int TW = W / 2, TH = H / 2;
    const int PAD = 18;

    for (int i = 0; i < (int)TILES.size(); i++) {
        int col = i % 2, row = i / 2;
        int rx = col * TW + PAD,   ry = row * TH + PAD;
        int rw = TW - 2 * PAD,     rh = TH - 2 * PAD;

        bool active = (g_running_service == TILES[i].service);
        bool hi     = (highlighted == i);

        uint8_t b = TILES[i].br, g = TILES[i].bg, r = TILES[i].bb;

        // Tile background
        if (hi) {
            fill_rect(rx, ry, rw, rh,
                      (uint8_t)std::min(b*14/10,255),
                      (uint8_t)std::min(g*14/10,255),
                      (uint8_t)std::min(r*14/10,255));
        } else if (active) {
            fill_rect(rx, ry, rw, rh,
                      (uint8_t)(b*9/10), (uint8_t)(g*9/10), (uint8_t)(r*9/10));
        } else {
            fill_rect(rx, ry, rw, rh, 38, 38, 38);
        }

        // Accent border
        draw_border(rx, ry, rw, rh, 3, b, g, r);
        if (active)
            draw_border(rx+3, ry+3, rw-6, rh-6, 4, 255, 255, 255);
    }

    // Header bar
    fill_rect(0, 0, W, 42, 12, 12, 12);
}

// ── GStreamer display ─────────────────────────────────────────────────────────
static GstElement* g_pipeline = nullptr;
static GstElement* g_appsrc   = nullptr;

static bool gst_init_display() {
    GError* err = nullptr;
    // textoverlay elements render tile labels and the header on top of the
    // coloured rectangles drawn in g_frame.
    // Layout: tile 0=TL 1=TR 2=BL 3=BR  (TW=960,TH=540,PAD=18)
    const char* pipe_str =
        "appsrc name=src is-live=true format=time "
        "  caps=video/x-raw,format=BGR,width=1920,height=1080,framerate=30/1 "
        "! videoconvert "
        /* header */
        "! textoverlay text=\"Touch a panel to launch the chosen demo.\""
        "  valignment=top halignment=left deltax=20 deltay=12"
        "  font-desc=\"Sans Bold 18\" color=0xffa0a0a0 "
        /* tile 0 TL label */
        "! textoverlay text=\"Edge Art\""
        "  valignment=top halignment=left deltax=218 deltay=222"
        "  font-desc=\"Sans Bold 38\" color=0xffffffff "
        "! textoverlay text=\"DeepX Pose + Qualcomm GenAI\""
        "  valignment=top halignment=left deltax=118 deltay=278"
        "  font-desc=\"Sans 20\" color=0xffd23cb4 "
        /* tile 1 TR label */
        "! textoverlay text=\"OEM Reference\""
        "  valignment=top halignment=left deltax=1138 deltay=222"
        "  font-desc=\"Sans Bold 38\" color=0xffffffff "
        "! textoverlay text=\"Qualcomm + DeepX Pipeline\""
        "  valignment=top halignment=left deltax=1088 deltay=278"
        "  font-desc=\"Sans 20\" color=0xff7890b4 "
        /* tile 2 BL label */
        "! textoverlay text=\"YOLO26 Parallel\""
        "  valignment=top halignment=left deltax=178 deltay=762"
        "  font-desc=\"Sans Bold 38\" color=0xffffffff "
        "! textoverlay text=\"DeepX Det || DeepX Seg\""
        "  valignment=top halignment=left deltax=158 deltay=818"
        "  font-desc=\"Sans 20\" color=0xff1ea01e "
        /* tile 3 BR label */
        "! textoverlay text=\"CLIP Demo\""
        "  valignment=top halignment=left deltax=1238 deltay=762"
        "  font-desc=\"Sans Bold 38\" color=0xffffffff "
        "! textoverlay text=\"Qualcomm HTP - Semantic CLIP\""
        "  valignment=top halignment=left deltax=1068 deltay=818"
        "  font-desc=\"Sans 20\" color=0xff008cdc "
        "! waylandsink fullscreen=true sync=false";

    g_pipeline = gst_parse_launch(pipe_str, &err);
    if (!g_pipeline || err) {
        fprintf(stderr, "GST error: %s\n", err ? err->message : "unknown");
        return false;
    }
    g_appsrc = GST_ELEMENT(
        gst_bin_get_by_name(GST_BIN(g_pipeline), "src"));
    gst_element_set_state(g_pipeline, GST_STATE_PLAYING);
    return true;
}

static void push_frame() {
    if (!g_appsrc) return;
    constexpr size_t sz = (size_t)H * W * 3;
    GstBuffer* buf = gst_buffer_new_allocate(nullptr, sz, nullptr);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_WRITE);
    memcpy(map.data, g_frame, sz);
    gst_buffer_unmap(buf, &map);
    gst_app_src_push_buffer(GST_APP_SRC(g_appsrc), buf);
}

// ── Touch input ───────────────────────────────────────────────────────────────
static std::atomic<int> g_touch_x{-1}, g_touch_y{-1};
static std::atomic<bool> g_tap{false};
static int g_touch_max_x = 4096;
static int g_touch_max_y = 4096;

static void touch_thread() {
    char path[64];
    int fd = -1;
    for (int i = 0; i < 8; i++) {
        snprintf(path, sizeof(path), "/dev/input/event%d", i);
        int f = open(path, O_RDONLY | O_NONBLOCK);
        if (f < 0) continue;
        unsigned long bits[4] = {};
        if (ioctl(f, EVIOCGBIT(0, sizeof(bits)), bits) >= 0 &&
            (bits[0] & (1 << EV_ABS))) {
            // Read actual touch range
            struct input_absinfo abs_x{}, abs_y{};
            if (ioctl(f, EVIOCGABS(ABS_X), &abs_x) >= 0 && abs_x.maximum > 0)
                g_touch_max_x = abs_x.maximum;
            if (ioctl(f, EVIOCGABS(ABS_Y), &abs_y) >= 0 && abs_y.maximum > 0)
                g_touch_max_y = abs_y.maximum;
            fprintf(stderr, "[picker] touch range x[0..%d] y[0..%d]\n",
                    g_touch_max_x, g_touch_max_y);
            fd = f;
            break;
        }
        close(f);
    }
    if (fd < 0) {
        fprintf(stderr, "No touch device found\n");
        return;
    }

    int tx = -1, ty = -1;
    struct input_event ev;
    while (true) {
        ssize_t n = read(fd, &ev, sizeof(ev));
        if (n < (ssize_t)sizeof(ev)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (ev.type == EV_ABS) {
            if (ev.code == ABS_MT_POSITION_X || ev.code == ABS_X) tx = ev.value;
            if (ev.code == ABS_MT_POSITION_Y || ev.code == ABS_Y) ty = ev.value;
        }
        if (ev.type == EV_KEY && ev.code == BTN_TOUCH && ev.value == 1) {
            g_touch_x = tx;
            g_touch_y = ty;
            g_tap     = true;
        }
    }
    close(fd);
}

static int tap_to_tile(int x, int y) {
    x = x * W / g_touch_max_x;
    y = y * H / g_touch_max_y;
    int col = std::min(x / (W / 2), 1);
    int row = std::min(y / (H / 2), 1);
    return row * 2 + col;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    if (!gst_init_display()) return 1;

    std::thread t(touch_thread);
    t.detach();

    int highlighted = -1;
    auto hi_until   = std::chrono::steady_clock::now();

    while (true) {
        // Check for tap
        if (g_tap.exchange(false)) {
            int tx = g_touch_x.load();
            int ty = g_touch_y.load();
            int tile = tap_to_tile(tx, ty);
            if (tile >= 0 && tile < (int)TILES.size()) {
                highlighted = tile;
                hi_until    = std::chrono::steady_clock::now()
                              + std::chrono::milliseconds(600);
                launch(tile);
            }
        }

        // Clear highlight after timeout
        if (highlighted >= 0 &&
            std::chrono::steady_clock::now() > hi_until) {
            highlighted = -1;
        }

        render(highlighted);
        push_frame();

        std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 fps
    }

    gst_element_set_state(g_pipeline, GST_STATE_NULL);
    gst_object_unref(g_pipeline);
    return 0;
}
