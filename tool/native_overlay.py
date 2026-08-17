import ctypes
import os
import platform
import sys
import time

import cv2
import numpy as np


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ============================================================
# Vision
# ============================================================

from common.config_loader import ConfigLoader

from recognition.vision_manager import (
    VisionManager
)


# ============================================================
# PyQt
# ============================================================

from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QRectF,
    QPointF,
)

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
)

from PyQt5.QtGui import (
    QPainter,
    QColor,
    QPen,
    QFont,
    QBrush,
    QImage,
)


# ============================================================
# Vision Worker
# ============================================================

class VisionWorker(QThread):

    data_signal = pyqtSignal(dict)

    TARGET_FPS = 30.0

    def __init__(
        self,
        vision: VisionManager,
    ):

        super().__init__()

        self.vision = vision

        self.running = True

        self.previous_anchor = None

    # =========================================================
    # Worker
    # =========================================================

    def run(self):

        interval = (
            1.0
            / self.TARGET_FPS
        )

        next_tick = (
            time.perf_counter()
        )

        while self.running:

            frame_start = (
                time.perf_counter()
            )

            # =================================================
            # Capture
            # =================================================

            try:

                frame = (
                    self.vision
                    .capturer
                    .grab_screen()
                )

            except Exception as exc:

                print(
                    f"[Capture] "
                    f"截图失败: {exc}"
                )

                time.sleep(0.05)

                next_tick = (
                    time.perf_counter()
                )

                continue

            # =================================================
            # HSV
            # =================================================

            frame_hsv = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2HSV,
            )

            # =================================================
            # 动态目标
            # =================================================

            anchor = (
                self.vision
                .indicators
                .find_locked_anchor(
                    frame_hsv,
                    previous_anchor=(
                        self.previous_anchor
                    ),
                )
            )

            # =================================================
            # 目标切换
            # =================================================

            target_switched = False

            if (
                anchor is not None
                and self.previous_anchor is not None
            ):

                dx = (
                    anchor[0]
                    - self.previous_anchor[0]
                )

                dy = (
                    anchor[1]
                    - self.previous_anchor[1]
                )

                jump_distance = (
                    (dx * dx + dy * dy)
                    ** 0.5
                )

                if (
                    jump_distance
                    >= self.vision
                    .TARGET_SWITCH_DISTANCE
                ):

                    target_switched = True

            if target_switched:

                print(
                    "[Vision] "
                    "检测到目标可能切换，"
                    "刷新舰船信息"
                )

                self.vision.notify_target_switch()

            # =================================================
            # 更新跟踪
            # =================================================

            if anchor is not None:

                self.previous_anchor = anchor

            else:

                self.previous_anchor = None

            # =================================================
            # Vision Manager
            # =================================================

            result = (
                self.vision
                .process_frame(
                    frame_bgr=frame,
                    frame_hsv=frame_hsv,
                    anchor=anchor,
                )
            )

            # =================================================
            # Vision latency
            # =================================================

            cost_ms = (
                time.perf_counter()
                - frame_start
            ) * 1000.0

            # =================================================
            # Debug
            # =================================================

            payload = {

                "anchor":
                    anchor,

                "result":
                    result,

                "cost_ms":
                    cost_ms,

                "ocr_ms":
                    self.vision
                    .ocr
                    .last_ocr_ms,

                "ocr_field":
                    self.vision
                    .ocr
                    .last_ocr_field,

                "ocr_text":
                    self.vision
                    .ocr
                    .last_ocr_text,

                "ocr_score":
                    self.vision
                    .ocr
                    .last_ocr_score,

                "debug_regions":
                    self.vision
                    .get_debug_regions(
                        anchor
                    ),
            }

            self.data_signal.emit(
                payload
            )

            # =================================================
            # FPS deadline
            # =================================================

            next_tick += interval

            remaining = (
                next_tick
                - time.perf_counter()
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )

            else:

                next_tick = (
                    time.perf_counter()
                )

    # =========================================================
    # Stop
    # =========================================================

    def stop(self):

        self.running = False

        self.wait()


# ============================================================
# Overlay
# ============================================================

class TransparentOverlay(QWidget):

    def __init__(self):

        super().__init__()

        # =====================================================
        # Vision
        # =====================================================

        self.config = ConfigLoader(
            "config.yaml"
        )

        self.vision = VisionManager(
            self.config
        )

        self.sw = self.vision.sw
        self.sh = self.vision.sh

        self.cx = self.vision.cx
        self.cy = self.vision.cy

        # =====================================================
        # Render
        # =====================================================

        self.render_data = None

        self.fps_tracker = 0.0

        self.last_frame_time = (
            time.perf_counter()
        )

        # =====================================================
        # Window
        # =====================================================

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput
            | Qt.Tool
        )

        self.setAttribute(
            Qt.WA_TranslucentBackground,
            True,
        )

        self.setGeometry(
            0,
            0,
            self.sw,
            self.sh,
        )

        # =====================================================
        # Windows Overlay
        # =====================================================

        self.hwnd = int(
            self.winId()
        )

        if platform.system() == "Windows":

            styles = (
                ctypes.windll.user32
                .GetWindowLongW(
                    self.hwnd,
                    -20,
                )
            )

            ctypes.windll.user32.SetWindowLongW(
                self.hwnd,
                -20,
                styles
                | 0x00000020
                | 0x00080000,
            )

            ctypes.windll.user32.SetWindowDisplayAffinity(
                self.hwnd,
                0x00000011,
            )

        # =====================================================
        # Worker
        # =====================================================

        self.worker = VisionWorker(
            self.vision
        )

        self.worker.data_signal.connect(
            self.on_data_received
        )

        self.worker.start()

    # =========================================================
    # Data
    # =========================================================

    def on_data_received(
        self,
        data,
    ):

        self.render_data = data

        now = (
            time.perf_counter()
        )

        delta = (
            now
            - self.last_frame_time
        )

        if delta > 0:

            instant_fps = (
                1.0 / delta
            )

            if self.fps_tracker <= 0:

                self.fps_tracker = instant_fps

            else:

                self.fps_tracker = (
                    self.fps_tracker * 0.85
                    + instant_fps * 0.15
                )

        self.last_frame_time = now

        self.update()

    # =========================================================
    # OCR Debug Preview
    # =========================================================

    @staticmethod
    def _build_qimage_from_cv(
        image: np.ndarray,
    ) -> QImage | None:

        if (
            image is None
            or image.size == 0
        ):

            return None

        # =====================================================
        # 确保是三通道 BGR
        # =====================================================

        if image.ndim == 2:

            bgr = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        elif (
            image.ndim == 3
            and image.shape[2] == 3
        ):

            bgr = image

        else:

            return None

        bgr = np.ascontiguousarray(
            bgr
        )

        # =====================================================
        # BGR → RGB
        # =====================================================

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        rgb = np.ascontiguousarray(
            rgb
        )

        h, w = (
            rgb.shape[:2]
        )

        qimage = QImage(
            rgb.data,
            w,
            h,
            rgb.strides[0],
            QImage.Format_RGB888,
        )

        # copy()：
        #
        # QImage 不继续引用 numpy 内存，
        # 避免函数退出后 rgb 被释放。
        return qimage.copy()

    # =========================================================
    # Paint
    # =========================================================

    def paintEvent(
        self,
        event,
    ):

        if self.render_data is None:
            return

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.Antialiasing,
            True,
        )

        painter.setRenderHint(
            QPainter.TextAntialiasing,
            True,
        )

        data = self.render_data

        anchor = data.get(
            "anchor"
        )

        result = data.get(
            "result"
        )

        debug_regions = data.get(
            "debug_regions",
            {}
        )

        cost_ms = data.get(
            "cost_ms",
            0.0
        )

        ocr_ms = data.get(
            "ocr_ms",
            0.0
        )

        ocr_field = data.get(
            "ocr_field",
            ""
        )

        ocr_text = data.get(
            "ocr_text",
            ""
        )

        ocr_score = data.get(
            "ocr_score",
            0.0
        )

        # =====================================================
        # Result 数据
        # =====================================================

        if result is None:

            state = None

            hud_raw = {}

            ship_name = "Unknown"

            max_speed = 0.0

            target_data = None

        else:

            state = result.get(
                "state"
            )

            hud_raw = result.get(
                "hud_raw",
                {}
            )

            ship_name = result.get(
                "ship_name",
                "Unknown",
            )

            max_speed = float(
                result.get(
                    "max_speed",
                    0.0,
                )
            )

            target_data = result.get(
                "target"
            )

        # =====================================================
        # 1. 固定 HUD
        # =====================================================

        fixed_fields = (
            "flight_time",
            "aim_distance",
            "ship_name",
            "max_speed",
            "enemy_angle",
            "our_angle",
        )

        painter.setFont(
            QFont(
                "Consolas",
                10,
                QFont.Bold,
            )
        )

        for name in fixed_fields:

            if name not in debug_regions:
                continue

            rect = debug_regions[name]["rect"]

            rx, ry, rw, rh = rect

            # -------------------------------------------------
            # 绿色透明框
            # -------------------------------------------------

            painter.setPen(
                QPen(
                    QColor(
                        0,
                        255,
                        127,
                        190,
                    ),
                    1,
                )
            )

            painter.setBrush(
                QBrush(
                    QColor(
                        0,
                        255,
                        127,
                        18,
                    )
                )
            )

            painter.drawRect(
                rx,
                ry,
                rw,
                rh,
            )

            value = hud_raw.get(
                name,
                "",
            )

            painter.setPen(
                QColor(
                    0,
                    255,
                    200,
                )
            )

            painter.drawText(
                rx,
                ry - 4,
                f"{name}: {value}",
            )

        # =====================================================
        # 2. 中央十字
        # =====================================================

        painter.setPen(
            QPen(
                QColor(
                    255,
                    255,
                    0,
                    150,
                ),
                1,
                Qt.DashLine,
            )
        )

        painter.drawLine(
            self.cx - 20,
            self.cy,
            self.cx + 20,
            self.cy,
        )

        painter.drawLine(
            self.cx,
            self.cy - 20,
            self.cx,
            self.cy + 20,
        )

        # =====================================================
        # 3. 动态目标
        # =====================================================

        if anchor is not None:

            (
                target_cx,
                target_cy,
                hx_left,
                hy_top,
                hw,
                hh,
            ) = anchor

            # -------------------------------------------------
            # 血条
            # -------------------------------------------------

            painter.setPen(
                QPen(
                    QColor(
                        255,
                        0,
                        0,
                        230,
                    ),
                    1.5,
                )
            )

            painter.setBrush(
                Qt.NoBrush
            )

            painter.drawRect(
                hx_left,
                hy_top,
                hw,
                hh,
            )

            # -------------------------------------------------
            # 血条左上角
            # -------------------------------------------------

            painter.setPen(
                QPen(
                    QColor(
                        0,
                        255,
                        255,
                        220,
                    ),
                    1.5,
                )
            )

            painter.setBrush(
                QBrush(
                    QColor(
                        0,
                        255,
                        255,
                    )
                )
            )

            painter.drawEllipse(
                QPointF(
                    hx_left,
                    hy_top,
                ),
                3,
                3,
            )

            # -------------------------------------------------
            # 血条 → 圆心
            # -------------------------------------------------

            painter.drawLine(
                hx_left,
                hy_top,
                target_cx,
                target_cy,
            )

            # -------------------------------------------------
            # 圆心十字
            # -------------------------------------------------

            painter.setPen(
                QPen(
                    QColor(
                        0,
                        150,
                        255,
                        255,
                    ),
                    2,
                )
            )

            painter.setBrush(
                Qt.NoBrush
            )

            painter.drawLine(
                target_cx - 8,
                target_cy,
                target_cx + 8,
                target_cy,
            )

            painter.drawLine(
                target_cx,
                target_cy - 8,
                target_cx,
                target_cy + 8,
            )

            # -------------------------------------------------
            # 航速圆环
            # -------------------------------------------------

            radius = (
                self.vision
                .indicators
                .RING_RADIUS
            )

            painter.setPen(
                QPen(
                    QColor(
                        255,
                        255,
                        255,
                        150,
                    ),
                    1,
                    Qt.DotLine,
                )
            )

            painter.setBrush(
                Qt.NoBrush
            )

            painter.drawEllipse(
                QPointF(
                    target_cx,
                    target_cy,
                ),
                radius,
                radius,
            )

            # -------------------------------------------------
            # 8 个人工采样点
            # -------------------------------------------------

            for index, (
                dx,
                dy,
                _angle,
                _radius,
            ) in enumerate(
                self.vision
                .indicators
                .ring_sample_offsets
            ):

                px = (
                    target_cx
                    + dx
                )

                py = (
                    target_cy
                    + dy
                )

                painter.setPen(
                    Qt.NoPen
                )

                painter.setBrush(
                    QBrush(
                        QColor(
                            255,
                            0,
                            255,
                            235,
                        )
                    )
                )

                painter.drawEllipse(
                    QPointF(
                        px,
                        py,
                    ),
                    2.5,
                    2.5,
                )

                painter.setPen(
                    QColor(
                        255,
                        255,
                        0,
                    )
                )

                painter.setFont(
                    QFont(
                        "Consolas",
                        8,
                        QFont.Bold,
                    )
                )

                painter.drawText(
                    px + 4,
                    py - 4,
                    str(index + 1),
                )

            # -------------------------------------------------
            # 状态灯框
            # -------------------------------------------------

            light_y = (
                target_cy
                - radius
                - self.vision
                .indicators
                .LIGHT_OFFSET
            )

            painter.setPen(
                QPen(
                    QColor(
                        255,
                        255,
                        0,
                        220,
                    ),
                    1,
                )
            )

            painter.setBrush(
                Qt.NoBrush
            )

            painter.drawRect(
                target_cx - 3,
                light_y - 3,
                6,
                6,
            )

            # -------------------------------------------------
            # 敌舰距离动态 ROI
            # -------------------------------------------------

            enemy_distance_region = (
                debug_regions.get(
                    "enemy_distance"
                )
            )

            if enemy_distance_region:

                ex, ey, ew, eh = (
                    enemy_distance_region[
                        "rect"
                    ]
                )

                painter.setPen(
                    QPen(
                        QColor(
                            255,
                            170,
                            0,
                            230,
                        ),
                        1.5,
                    )
                )

                painter.setBrush(
                    Qt.NoBrush
                )

                painter.drawRect(
                    ex,
                    ey,
                    ew,
                    eh,
                )

                distance_text = hud_raw.get(
                    "enemy_distance",
                    "",
                )

                painter.setFont(
                    QFont(
                        "Consolas",
                        9,
                        QFont.Bold,
                    )
                )

                painter.setPen(
                    QColor(
                        255,
                        190,
                        0,
                    )
                )

                painter.drawText(
                    ex,
                    ey - 4,
                    f"Enemy Dist: {distance_text}",
                )

        # =====================================================
        # 4. 数据面板
        # =====================================================

        panel_x = 25
        panel_y = 25
        panel_w = 520
        panel_h = 375

        painter.setPen(
            QPen(
                QColor(
                    0,
                    255,
                    180,
                    180,
                ),
                1.5,
            )
        )

        painter.setBrush(
            QBrush(
                QColor(
                    15,
                    15,
                    20,
                    210,
                )
            )
        )

        painter.drawRoundedRect(
            QRectF(
                panel_x,
                panel_y,
                panel_w,
                panel_h,
            ),
            8,
            8,
        )

        # -----------------------------------------------------
        # 标题
        # -----------------------------------------------------

        painter.setFont(
            QFont(
                "Microsoft YaHei",
                12,
                QFont.Bold,
            )
        )

        painter.setPen(
            QColor(
                0,
                255,
                255,
            )
        )

        painter.drawText(
            panel_x + 15,
            panel_y + 28,
            "WoWs AI Pre-Aim Data Inspector",
        )

        # =====================================================
        # 性能
        # =====================================================

        painter.setFont(
            QFont(
                "Consolas",
                9,
            )
        )

        locked = (
            state is not None
        )

        painter.setPen(
            QColor(
                160,
                160,
                160,
            )
        )

        painter.drawText(
            panel_x + 15,
            panel_y + 48,
            (
                f"FPS: {self.fps_tracker:4.1f} | "
                f"Vision: {cost_ms:4.1f}ms"
            ),
        )

        painter.drawText(
            panel_x + 15,
            panel_y + 64,
            (
                f"OCR: {ocr_ms:4.1f}ms | "
                f"Field: {ocr_field} | "
                f"Score: {ocr_score:.2f}"
            ),
        )

        painter.drawText(
            panel_x + 15,
            panel_y + 80,
            (
                f"Raw OCR: "
                f"{ocr_text[:48]}"
            ),
        )

        painter.setPen(
            QColor(
                0,
                255,
                127,
            )
            if locked
            else QColor(
                255,
                80,
                80,
            )
        )

        painter.drawText(
            panel_x + 420,
            panel_y + 48,
            (
                "LOCKED"
                if locked
                else "SEARCHING"
            ),
        )

        painter.setPen(
            QColor(
                70,
                70,
                70,
            )
        )

        painter.drawLine(
            panel_x + 15,
            panel_y + 92,
            panel_x + panel_w - 15,
            panel_y + 92,
        )

        # =====================================================
        # OCR Processed Preview
        #
        # 显示：
        #
        #     OCRParser._recognize()
        #         ↓
        #     _prepare_roi()
        #         ↓
        #     最终 image
        #         ↓
        #     RapidOCR
        #
        # 所以这里看到的就是 OCR 实际收到的图片。
        # =====================================================

        debug_image, debug_field = (
            self.vision
            .ocr
            .get_debug_image()
        )

        preview_x = (
            panel_x + 300
        )

        preview_y = (
            panel_y + 105
        )

        preview_w = 205
        preview_h = 100

        # -----------------------------------------------------
        # Preview 外框
        # -----------------------------------------------------

        painter.setPen(
            QPen(
                QColor(
                    100,
                    220,
                    255,
                    220,
                ),
                1,
            )
        )

        painter.setBrush(
            QBrush(
                QColor(
                    0,
                    0,
                    0,
                    170,
                )
            )
        )

        painter.drawRect(
            preview_x,
            preview_y,
            preview_w,
            preview_h,
        )

        # -----------------------------------------------------
        # 标题
        # -----------------------------------------------------

        painter.setFont(
            QFont(
                "Consolas",
                8,
                QFont.Bold,
            )
        )

        painter.setPen(
            QColor(
                120,
                220,
                255,
            )
        )

        painter.drawText(
            preview_x + 5,
            preview_y + 12,
            (
                "OCR Processed: "
                f"{debug_field or '-'}"
            ),
        )

        # -----------------------------------------------------
        # 图片
        # -----------------------------------------------------

        qimage = (
            self._build_qimage_from_cv(
                debug_image
            )
            if debug_image is not None
            else None
        )

        if qimage is not None:

            available_w = (
                preview_w - 10
            )

            available_h = (
                preview_h - 20
            )

            scaled = qimage.scaled(
                available_w,
                available_h,
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )

            image_x = (
                preview_x
                + (
                    preview_w
                    - scaled.width()
                )
                // 2
            )

            image_y = (
                preview_y
                + 18
                + (
                    available_h
                    - scaled.height()
                )
                // 2
            )

            painter.drawImage(
                image_x,
                image_y,
                scaled,
            )

        else:

            painter.setPen(
                QColor(
                    120,
                    120,
                    120,
                )
            )

            painter.drawText(
                preview_x + 10,
                preview_y + 55,
                "No OCR image yet",
            )

        # =====================================================
        # 数据
        # =====================================================

        painter.setFont(
            QFont(
                "Consolas",
                10,
            )
        )

        if state is not None:

            enemy_distance = float(
                self.vision
                .ocr
                .get_cache()[
                    "enemy_distance"
                ]
            )

            if state.direction_state == 1:
                direction = "FWD"

            elif state.direction_state == -1:
                direction = "REV"

            else:
                direction = "STOP"

            current_speed = (
                state.speed_fraction
                * max_speed
                * state.direction_state
            )

            lines = [

                (
                    "Target Ship:",
                    (
                        f"{ship_name} "
                        f"(Max: "
                        f"{max_speed:.1f} kts)"
                    ),
                ),

                (
                    "Engine Gear:",
                    (
                        f"{state.speed_fraction * 8:.0f}/8 "
                        f"[{direction}] -> "
                        f"{abs(current_speed):.1f} kts"
                    ),
                ),

                (
                    "Enemy Dist.:",
                    f"{enemy_distance:.1f} 公里",
                ),

                (
                    "Aim Distance:",
                    f"{state.aiming_distance:.2f} km",
                ),

                (
                    "Flight Time:",
                    f"{state.flight_time:.2f} s.",
                ),

                (
                    "Rel Angle  :",
                    f"{state.relative_angle:.1f}°",
                ),

                (
                    "State Dump :",
                    (
                        f"dir={state.direction_state}, "
                        f"frac="
                        f"{state.speed_fraction:.3f}"
                    ),
                ),

                (
                    "Lead Est.  :",
                    (
                        f"dx ≈ "
                        f"{abs(current_speed) * 0.5144 * state.flight_time:.1f}m"
                    ),
                ),
            ]

            for index, (
                label,
                value,
            ) in enumerate(lines):

                row_y = (
                    panel_y
                    + 115
                    + index * 29
                )

                painter.setPen(
                    QColor(
                        130,
                        200,
                        255,
                    )
                )

                painter.drawText(
                    panel_x + 18,
                    row_y,
                    label,
                )

                painter.setPen(
                    QColor(
                        255,
                        255,
                        255,
                    )
                )

                painter.drawText(
                    panel_x + 160,
                    row_y,
                    value,
                )

        else:

            painter.setPen(
                QColor(
                    150,
                    150,
                    150,
                )
            )

            painter.drawText(
                panel_x + 18,
                panel_y + 125,
                "Target not locked or out of view.",
            )

    # =========================================================
    # Close
    # =========================================================

    def closeEvent(
        self,
        event,
    ):

        self.worker.stop()

        self.vision.close()

        event.accept()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    overlay = TransparentOverlay()

    overlay.show()

    sys.exit(
        app.exec_()
    )