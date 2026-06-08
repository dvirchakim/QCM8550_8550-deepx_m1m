"""
PyQt6 main window matching PRD section 5.1.

Layout (touch screen 1280 x 800):

  ┌────────────────────────────────────────────────────────────┐
  │  HEADER  IMDT QCS8550 + DEEPX HETEROGENEOUS AI PIPELINE   │   60 px
  ├──────────────────────────────────┬──────────────────────────┤
  │                                  │                          │
  │   Cam 0 viewport + pose overlay  │  Cam 1 viewport +        │
  │                                  │  pose overlay (top)      │   ~520 px
  │                                  ├──────────────────────────┤
  │                                  │  Generative mural pane   │
  │                                  │  (Stable-Diffusion-ish)  │
  ├──────────────────────────────────┴──────────────────────────┤
  │  [NEON] [VANGOGH] [COMIC] [NOIR]      POWER 4.8 W  TEMP 55C │   90 px
  └────────────────────────────────────────────────────────────┘

Designed for Wayland on the board (QT_QPA_PLATFORM=wayland-egl), works
identically on Windows for development.
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget, QFrame,
)

import config


# ---------------------------------------------------------------------------
def _to_qpixmap(bgr: np.ndarray) -> QPixmap:
    if bgr is None:
        return QPixmap()
    h, w, _ = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


# ---------------------------------------------------------------------------
class VideoPane(QLabel):
    """Auto-scaling video display widget with a label badge."""

    def __init__(self, badge: str) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            f"background:{config.COLOR_PANEL_BG}; color:{config.COLOR_ACCENT};"
            f"border:1px solid #1d2436; border-radius:6px;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._badge = badge
        self.setText(f"\u23F3  {badge}\n(waiting for stream)")
        font = QFont(); font.setPointSize(14); font.setBold(True)
        self.setFont(font)

    def show_frame(self, bgr: np.ndarray) -> None:
        if bgr is None:
            return
        pix = _to_qpixmap(bgr)
        self.setPixmap(pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))


# ---------------------------------------------------------------------------
class StyleButton(QPushButton):
    def __init__(self, label: str, style_id: str) -> None:
        super().__init__(label)
        self.style_id = style_id
        self.setCheckable(True)
        self.setMinimumHeight(60)
        self.setMinimumWidth(190)
        self.setStyleSheet(
            "QPushButton {"
            "  background:#101729; color:#ffffff; border:2px solid #1d2436;"
            "  border-radius:10px; font-size:14pt; font-weight:bold;"
            "  padding:4px 10px; }"
            "QPushButton:checked {"
            "  background:" + config.COLOR_HEADER_BG +
            "; border:2px solid " + config.COLOR_ACCENT + "; }"
            "QPushButton:pressed {"
            "  background:" + config.COLOR_ACCENT + "; color:#001020; }"
        )


# ---------------------------------------------------------------------------
class TelemetryWidget(QLabel):
    """Compact power/temp/FPS readout."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(310)
        self.setStyleSheet(
            f"background:#06101c; color:{config.COLOR_OK};"
            f"border:1px solid {config.COLOR_ACCENT}; border-radius:8px;"
            "padding:6px 14px; font-family:Consolas, monospace; font-size:12pt;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)

    def update_values(self, *, power_w: float, temp_c: float, fps: float,
                      gen_count: int, throttled: bool) -> None:
        col = config.COLOR_ERR if throttled or power_w > config.POWER_BUDGET_W else config.COLOR_OK
        self.setStyleSheet(self.styleSheet().split("color:")[0]
                           + f"color:{col};"
                           + ";".join(self.styleSheet().split(";")[1:]))
        self.setText(
            f"SYSTEM POWER  {power_w:5.2f} W   "
            f"TEMP {temp_c:4.1f}\u00B0C   "
            f"UI {fps:4.1f} FPS   "
            f"GEN #{gen_count}"
        )


# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """Touch-friendly booth UI."""

    style_changed = pyqtSignal(str)

    def __init__(self, agent) -> None:
        super().__init__()
        self.agent = agent
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self._build_ui()
        self._wire_signals()
        self._frame_count = 0
        self._fps = 0.0
        self._fps_t0 = time.time()

        # 60 FPS UI refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(int(1000 / config.TARGET_DISPLAY_FPS))

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(config.COLOR_PANEL_BG))
        self.setPalette(pal)

        root = QWidget(); self.setCentralWidget(root)
        root_lay = QVBoxLayout(root); root_lay.setSpacing(8); root_lay.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel(config.WINDOW_TITLE)
        header.setStyleSheet(
            f"background:{config.COLOR_HEADER_BG}; color:{config.COLOR_HEADER_FG};"
            "border-radius:8px; padding:10px;"
        )
        f = QFont(); f.setPointSize(16); f.setBold(True); header.setFont(f)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFixedHeight(config.HEADER_HEIGHT)
        root_lay.addWidget(header)

        # Centre row: left viewport | right column
        middle = QHBoxLayout(); middle.setSpacing(8)
        root_lay.addLayout(middle, stretch=1)

        self.cam0_pane = VideoPane("CAMERA 0 \u2192 DEEPX YOLOv5-Pose 17KP")
        middle.addWidget(self.cam0_pane, stretch=2)

        right_col = QVBoxLayout(); right_col.setSpacing(8)
        middle.addLayout(right_col, stretch=2)

        self.cam1_pane = VideoPane("CAMERA 1 \u2192 DEEPX YOLOv5-Pose 17KP")
        right_col.addWidget(self.cam1_pane, stretch=1)

        self.gen_pane = VideoPane("QCS8550 GPU \u2192 STYLIZED MURAL")
        right_col.addWidget(self.gen_pane, stretch=1)

        # Footer: style buttons + telemetry
        footer = QHBoxLayout(); footer.setSpacing(10)
        footer_w = QWidget(); footer_w.setLayout(footer)
        footer_w.setFixedHeight(config.FOOTER_HEIGHT)
        root_lay.addWidget(footer_w)

        self.style_buttons: list[StyleButton] = []
        for i, s in enumerate(config.STYLES):
            btn = StyleButton(s["label"], s["id"])
            if i == config.DEFAULT_STYLE_INDEX:
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked, sid=s["id"]: self._select_style(sid))
            footer.addWidget(btn)
            self.style_buttons.append(btn)

        footer.addStretch(1)
        self.telemetry = TelemetryWidget()
        footer.addWidget(self.telemetry)

    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        self.style_changed.connect(self.agent.on_style_change)

    def _select_style(self, style_id: str) -> None:
        for b in self.style_buttons:
            b.setChecked(b.style_id == style_id)
        self.style_changed.emit(style_id)

    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        frame0, frame1, gen = self.agent.render_frames()
        if frame0 is not None: self.cam0_pane.show_frame(frame0)
        if frame1 is not None: self.cam1_pane.show_frame(frame1)
        if gen    is not None: self.gen_pane.show_frame(gen)

        # FPS bookkeeping (rolling 30-frame window)
        self._frame_count += 1
        now = time.time()
        if now - self._fps_t0 >= 0.5:
            self._fps = self._frame_count / (now - self._fps_t0)
            self._frame_count = 0
            self._fps_t0 = now

        t = self.agent.monitor.snapshot()
        self.telemetry.update_values(
            power_w=t.power_w, temp_c=t.temp_c, fps=self._fps,
            gen_count=self.agent.stylizer.generations,
            throttled=t.throttled,
        )

    # ------------------------------------------------------------------
    def closeEvent(self, ev) -> None:
        self.agent.shutdown()
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
def run_app(agent) -> int:
    app = QApplication(sys.argv)
    win = MainWindow(agent)
    win.show()
    return app.exec()
