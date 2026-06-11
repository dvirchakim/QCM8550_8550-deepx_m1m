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

static const std::array<Tile, 3> TILES = {{
    { "Edge Art",        "DeepX Pose  +  Qualcomm GenAI",  "edge-art.service",        210,  60, 180 },
    { "OEM Reference",   "Qualcomm  +  DeepX Pipeline",    "imdt-deepx-demo.service",   0, 120, 180 },
    { "YOLO26 Parallel", "DeepX Det  ||  DeepX Seg",       "yolo26-parallel.service",  30, 160,  30 },
}};

// ── Service control ───────────────────────────────────────────────────────────
static std::string g_running_service;

static void launch(int tile_idx) {
    std::string prev = g_running_service;
    g_running_service = TILES[tile_idx].service;
    // Run in background thread so the render loop is never blocked.
    std::thread([prev, svc = g_running_service]() {
        if (!prev.empty()) {
            std::string cmd = "/bin/systemctl stop " + prev + " 2>/dev/null";
            system(cmd.c_str());
        }
        std::string cmd = "/bin/systemctl start " + svc + " 2>/dev/null";
        system(cmd.c_str());
    }).detach();
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
static constexpr int HDR  = 56;    // header height px
static constexpr int PAD  = 14;    // card inset px
static constexpr int TW   = W / 2; // 960
static constexpr int TH   = (H - HDR) / 2; // 512

static void render(int highlighted) {
    memset(g_frame, 12, sizeof(g_frame));           // #0c0c0c bg

    fill_rect(0, 0, W, HDR, 8, 8, 10);             // header near-black
    fill_rect(0, HDR, W, 1, 50, 44, 42);            // separator line

    for (int i = 0; i < 4; i++) {
        int col = i % 2, row = i / 2;
        int rx = col * TW + PAD;
        int ry = HDR + row * TH + PAD;
        int rw = TW - 2 * PAD;
        int rh = TH - 2 * PAD;

        if (i >= (int)TILES.size()) {
            // Empty slot — leave as background
            continue;
        }

        bool active = (g_running_service == TILES[i].service);
        bool hi     = (highlighted == i);
        uint8_t b = TILES[i].br, g = TILES[i].bg, r = TILES[i].bb;

        if (hi) {
            fill_rect(rx, ry, rw, rh,
                (uint8_t)std::min((int)b/3 + 30, 255),
                (uint8_t)std::min((int)g/3 + 30, 255),
                (uint8_t)std::min((int)r/3 + 30, 255));
        } else if (active) {
            fill_rect(rx, ry, rw, rh,
                (uint8_t)std::min((int)b/5 + 26, 255),
                (uint8_t)std::min((int)g/5 + 26, 255),
                (uint8_t)std::min((int)r/5 + 26, 255));
        } else {
            fill_rect(rx, ry, rw, rh, 28, 26, 25);
        }

        fill_rect(rx, ry, 5, rh, b, g, r);             // left accent bar
        fill_rect(rx, ry + rh - 4, rw, 4, b, g, r);    // bottom accent bar
        if (active)
            fill_rect(rx, ry, rw, 3,
                (uint8_t)std::min((int)b + 80, 255),
                (uint8_t)std::min((int)g + 80, 255),
                (uint8_t)std::min((int)r + 80, 255));   // active top bar
    }
}

// ── GStreamer display ─────────────────────────────────────────────────────────
static GstElement* g_pipeline = nullptr;
static GstElement* g_appsrc   = nullptr;
static std::atomic<bool> g_pipeline_error{false};

// Text layout (HDR=56, PAD=14, TW=960, TH=512, card rh=484):
//   col=0 left=36   col=1 left=996
//   row=0 ry=70:   tag=92   title=288(45%)  sub=360(60%)
//   row=1 ry=582:  tag=604  title=800(45%)  sub=872(60%)
//   title↔sub gap = 72px — robust against high-DPI font scaling
static const char* PIPE_STR =
    "appsrc name=src is-live=true format=time "
    "  caps=video/x-raw,format=BGR,width=1920,height=1080,framerate=30/1 "
    "! videoconvert "
    /* ── header ── */
    "! textoverlay text=\"AI Demo Station\""
    "  valignment=top halignment=left deltax=24 deltay=17"
    "  font-desc=\"Sans Bold 17\" color=0xffb8b8d0 "
    "! textoverlay text=\"QCS8550  ·  DeepX DX-M1\""
    "  valignment=top halignment=left deltax=1200 deltay=20"
    "  font-desc=\"Sans 13\" color=0xff606078 "
    /* ── tile 0  TL  Edge Art ── */
    "! textoverlay text=\"01\""
    "  valignment=top halignment=left deltax=36 deltay=92"
    "  font-desc=\"Sans Bold 11\" color=0xffd23cb4 "
    "! textoverlay text=\"Edge Art\""
    "  valignment=top halignment=left deltax=36 deltay=288"
    "  font-desc=\"Sans Bold 20\" color=0xffe8e8f2 "
    "! textoverlay text=\"DeepX Pose  +  Qualcomm GenAI\""
    "  valignment=top halignment=left deltax=36 deltay=360"
    "  font-desc=\"Sans 12\" color=0xffd23cb4 "
    /* ── tile 1  TR  OEM Reference ── */
    "! textoverlay text=\"02\""
    "  valignment=top halignment=left deltax=996 deltay=92"
    "  font-desc=\"Sans Bold 11\" color=0xff7890b4 "
    "! textoverlay text=\"OEM Reference\""
    "  valignment=top halignment=left deltax=996 deltay=288"
    "  font-desc=\"Sans Bold 20\" color=0xffe8e8f2 "
    "! textoverlay text=\"Qualcomm  +  DeepX Pipeline\""
    "  valignment=top halignment=left deltax=996 deltay=360"
    "  font-desc=\"Sans 12\" color=0xff7890b4 "
    /* ── tile 2  BL  YOLO26 Parallel ── */
    "! textoverlay text=\"03\""
    "  valignment=top halignment=left deltax=36 deltay=604"
    "  font-desc=\"Sans Bold 11\" color=0xff1ea01e "
    "! textoverlay text=\"YOLO26 Parallel\""
    "  valignment=top halignment=left deltax=36 deltay=800"
    "  font-desc=\"Sans Bold 20\" color=0xffe8e8f2 "
    "! textoverlay text=\"DeepX Det  ||  DeepX Seg\""
    "  valignment=top halignment=left deltax=36 deltay=872"
    "  font-desc=\"Sans 12\" color=0xff1ea01e "
    "! waylandsink fullscreen=true sync=false";

static gboolean bus_cb(GstBus*, GstMessage* msg, gpointer) {
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* e = nullptr; gchar* d = nullptr;
        gst_message_parse_error(msg, &e, &d);
        fprintf(stderr, "[picker] GST error: %s\n", e ? e->message : "?");
        g_error_free(e); g_free(d);
        g_pipeline_error = true;
    }
    return TRUE;
}

static bool build_pipeline() {
    GError* err = nullptr;
    g_pipeline = gst_parse_launch(PIPE_STR, &err);
    if (!g_pipeline || err) {
        fprintf(stderr, "[picker] parse error: %s\n", err ? err->message : "?");
        if (err) g_error_free(err);
        return false;
    }
    g_appsrc = GST_ELEMENT(gst_bin_get_by_name(GST_BIN(g_pipeline), "src"));
    GstBus* bus = gst_element_get_bus(g_pipeline);
    gst_bus_add_watch(bus, bus_cb, nullptr);
    gst_object_unref(bus);
    gst_element_set_state(g_pipeline, GST_STATE_PLAYING);
    g_pipeline_error = false;
    return true;
}

static void destroy_pipeline() {
    if (!g_pipeline) return;
    gst_element_set_state(g_pipeline, GST_STATE_NULL);
    if (g_appsrc) { gst_object_unref(g_appsrc); g_appsrc = nullptr; }
    gst_object_unref(g_pipeline);
    g_pipeline = nullptr;
}

// Rebuild after a Wayland error (called from main loop on g_pipeline_error).
// Sleep briefly to let the compositor settle before creating a new surface.
static void rebuild_pipeline() {
    fprintf(stderr, "[picker] rebuilding pipeline\n");
    destroy_pipeline();
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    build_pipeline();
}

static void push_frame() {
    if (!g_appsrc) return;
    constexpr size_t sz = (size_t)H * W * 3;
    GstBuffer* buf = gst_buffer_new_allocate(nullptr, sz, nullptr);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_WRITE);
    memcpy(map.data, g_frame, sz);
    gst_buffer_unmap(buf, &map);
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(g_appsrc), buf);
    if (ret != GST_FLOW_OK) g_pipeline_error = true;
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

    if (!build_pipeline()) return 1;

    std::thread t(touch_thread);
    t.detach();

    int highlighted = -1;
    auto hi_until   = std::chrono::steady_clock::now();

    while (true) {
        // Drain GStreamer bus events (picks up errors without a separate GMainLoop).
        g_main_context_iteration(nullptr, FALSE);

        // Recover from pipeline error (Wayland surface lost, etc.).
        if (g_pipeline_error) {
            rebuild_pipeline();
            highlighted = -1;
        }

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

    destroy_pipeline();
    return 0;
}
