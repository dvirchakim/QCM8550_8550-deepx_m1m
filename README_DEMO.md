# Edge-Art Interactive Silhouette Mural

Heterogeneous AI demo for the **IMDT QCS8550 SBC + DEEPX DX-M1**.
PRD v1.0.0 — silent, visual, real-time pose-driven generative art for
noisy convention floors.

---

## Architecture (one screen)

```
                         ┌─── on the BOARD ───────────────┐
qtiqmmfsrc cam=0 ┐       │                                │
                 ├─ dxpreprocess ─ dxinfer (YOLOV5Pose) ──┤
qtiqmmfsrc cam=1 ┘       ├─ dxpostprocess ─ tee ──┐       │
                         │                       │       │
                         │  dxosd → waylandsink  │  visual│
                         │  dxmsgconv ─ dxmsgbroker ─ MQTT│
                         └────────────────────────────┬───┘
                                                      │
                            paho-mqtt ◀───────────────┘
                                │ 17-KP JSON
                                ▼
                   ┌────── Python (PyQt6) ──────┐
                   │  PoseRenderer  (60 FPS)    │
                   │  Stylizer      (≈1 Hz)     │
                   │  Telemetry / power widget  │
                   └────────────────────────────┘
```

The board side ships in the running demo's image already
(`/usr/share/dx-stream/...`); this project adds the Python compositor
and the MQTT pose extraction.

---

## File map

| File                      | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| `config.py`               | All paths, KPIs, style presets, MQTT settings           |
| `main.py`                 | Entry point (`--no-board` to skip gst-launch)           |
| `src/agent.py`            | Orchestrator (no Qt dependencies)                       |
| `src/board_pipeline.py`   | Builds & supervises the on-board gst-launch subprocess  |
| `src/pose_mqtt.py`        | `MqttPoseSource` + `SimulatedPoseSource`                |
| `src/pose_renderer.py`    | 17-KP stylized neon/comic/noir/vangogh overlay (60 FPS) |
| `src/stylizer.py`         | OpenCV-stub generative pane (≈1 Hz). Future: QNN+SD     |
| `src/ui_app.py`           | PyQt6 main window matching PRD §5.1                     |
| `src/monitor.py`          | Thermal + power telemetry                               |

Legacy files (`deepx_handler.py`, `qualcomm_handler.py`,
`grid_manager.py`, `sources.py`) are kept for reference but no longer
imported.

---

## Running on Windows (dev / UI work)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py --no-board
```

You should see the booth UI with two synthetic moving skeletons (the
simulated pose source) and the generative pane updating every ~1.2 s
per chosen style. Touch / click the four style buttons to switch.

---

## Running on the board (Yocto + DEEPX)

1. **Deploy the code** to the board:
   ```powershell
   .\deploy.ps1
   ```
2. **Stop the legacy demo** (we don't want it racing on the same
   cameras):
   ```bash
   systemctl stop imdt-deepx-demo
   systemctl disable imdt-deepx-demo
   ```
3. **Ensure mosquitto is up** (default broker is on `127.0.0.1:1883`):
   ```bash
   systemctl status mosquitto
   ```
4. **Run the new app**:
   ```bash
   cd /data/qualcomm_deepx_m1
   export QT_QPA_PLATFORM=wayland-egl
   export WAYLAND_DISPLAY=wayland-1
   export XDG_RUNTIME_DIR=/run/user/root
   python3 main.py
   ```
   The agent will spawn two gst-launch pipelines internally (one per
   camera), publish 17-KP pose JSON to `dxstream/pose/<cam>`, and let
   the Python compositor render the stylized overlay + generative pane.

---

## PRD KPI status (Phase 1)

| KPI (PRD §6)             | Target              | Phase-1 status                                   |
|--------------------------|---------------------|--------------------------------------------------|
| qtiqmmfsrc NV12 stable   | 0 drop / 12 h       | Inherited from baseline demo ✔                  |
| DX-M1 pose latency       | ≤ 5 ms              | Inference ✔, end-to-end including MQTT ≈ 10 ms |
| Touch interactivity      | Instant             | ≤ 50 ms via Qt event loop ✔                     |
| Total power              | ≤ 10 W              | Telemetry widget tracks; needs board run        |
| 60 FPS visual layer      | yes                 | UI ticks at 60 Hz; depends on Python compositor |

Generative pane is **OpenCV-stub** in Phase 1 - swap `src/stylizer.py`
internals for a QNN+ControlNet runner once the `.dlc` is ready.

---

## Troubleshooting

* **No pose overlay** → check `mosquitto -d` is running and subscribe
  to the topic to confirm data flow:
  `mosquitto_sub -h 127.0.0.1 -t 'dxstream/pose/#' -u user -P 1234`
* **`dxmsgbroker` plugin missing** → confirm `libdx_msgbroker_mqtt.so`
  exists under `/usr/share/dx-stream/lib/`.
* **Black viewports on board** → the gst pipeline owns the cameras for
  the visual layer; the Python compositor reads via OpenCV, which on
  the board may need a v4l2 alias. Run with `--no-board` to confirm
  the UI works, then bring up the gst pipeline separately.
