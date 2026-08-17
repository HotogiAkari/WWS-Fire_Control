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
# Overlay 调参区
#
# 这里只放 Overlay / Debug 显示相关参数。
# Recognition 算法参数不要放这里。
#
# OCR 参数：
#     recognition/vision_manager.py
#
# 血条 / ROI / 禁区：
#     recognition/indicator_parser.py
#
# OCR preprocessing：
#     recognition/ocr_parser.py
# ============================================================


# ============================================================
# 1. Vision Worker FPS
#
# 调高：
#     视觉更新更快
#     CPU 占用增加
#
# 调低：
#     CPU 占用降低
#     数据刷新变慢
# ============================================================

VISION_TARGET_FPS = 30.0


# ============================================================
# 2. Debug Panel
#
# 左上角数据面板。
# ============================================================

PANEL_X = 25
PANEL_Y = 25

PANEL_W = 760
PANEL_H = 520


# ============================================================
# 3. OCR Debug Preview
#
# OCR 处理后的图片横向排列在左上角面板内部。
# ============================================================

OCR_PREVIEW_GAP = 7

OCR_PREVIEW_CELL_H = 105


# ============================================================
# OCR Debug Preview 显示顺序
# ============================================================

OCR_PREVIEW_FIELDS = (

    "flight_time",

    "aim_distance",

    "enemy_angle",

    "our_angle",

    "enemy_distance",

    "max_speed",

    "ship_name",
)


# ============================================================
# OCR Debug Preview 显示名称
# ============================================================

OCR_PREVIEW_LABELS = {

    "flight_time":
        "Flight",

    "aim_distance":
        "Aim",

    "enemy_angle":
        "Enemy Ang",

    "our_angle":
        "Our Ang",

    "enemy_distance":
        "Enemy Dist",

    "max_speed":
        "Max Speed",

    "ship_name":
        "Ship Name",
}


# ============================================================
# 4. Performance / Debug 文字
# ============================================================

RAW_OCR_MAX_LENGTH = 70


# ============================================================
# 5. 数据面板第一行
#
# OCR 图片之后的数据从这里开始。
# ============================================================

DATA_START_Y = 255

DATA_ROW_HEIGHT = 29


# ============================================================
# 6. 中央十字大小
# ============================================================

CENTER_CROSS_SIZE = 20


# ============================================================
# 7. 目标中心十字大小
# ============================================================

TARGET_CROSS_SIZE = 8


# ============================================================
# 8. 状态灯 Debug 框大小
# ============================================================

LIGHT_DEBUG_SIZE = 6


# ============================================================
# 9. UI 禁区 Debug
# ============================================================

SHOW_UI_EXCLUSION_REGIONS = True


# ============================================================
# Imports
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

    TARGET_FPS = VISION_TARGET_FPS

    def __init__(
        self,
        vision: VisionManager,
    ):

        super().__init__()

        self.vision = vision

        self.running = True

        # =====================================================
        # 上一帧 Anchor
        #
        # 非常重要：
        #
        # 这里保存的必须始终是“上一帧”的结果。
        #
        # 当前帧处理完成以后，
        # 才会把 current anchor 写回来。
        # =====================================================

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
            # 保存“上一帧” Anchor
            #
            # 当前帧所有跟踪判断都必须使用这个值。
            # =================================================

            previous_anchor = (
                self.previous_anchor
            )

            # =================================================
            # 当前帧寻找目标
            # =================================================

            anchor = (
                self.vision
                .indicators
                .find_locked_anchor(
                    frame_hsv,
                    previous_anchor=(
                        previous_anchor
                    ),
                )
            )

            # =================================================
            # Target switch
            # =================================================

            target_switched = False

            if (
                anchor is not None
                and previous_anchor is not None
            ):

                dx = (
                    anchor[0]
                    - previous_anchor[0]
                )

                dy = (
                    anchor[1]
                    - previous_anchor[1]
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
            # VisionManager
            #
            # 注意：
            #
            # 这里必须传上一帧 Anchor。
            #
            # 不能提前执行：
            #
            #     self.previous_anchor = anchor
            #
            # 否则局部跟踪就会拿到当前帧自己。
            # =================================================

            try:

                result = (
                    self.vision
                    .process_frame(
                        frame_bgr=frame,
                        frame_hsv=frame_hsv,
                        anchor=anchor,
                        previous_anchor=(
                            previous_anchor
                        ),
                    )
                )

            except Exception as exc:

                print(
                    f"[Vision] "
                    f"处理帧失败: {exc}"
                )

                # 保持上一帧状态，
                # 下一帧继续尝试。
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

                continue

            # =================================================
            # 当前帧处理完成
            #
            # 现在才能更新 previous_anchor。
            # =================================================

            self.previous_anchor = anchor

            # =================================================
            # Vision latency
            # =================================================

            cost_ms = (
                time.perf_counter()
                - frame_start
            ) * 1000.0

            # =================================================
            # Debug Payload
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

                "ui_exclusion_regions":
                    self.vision
                    .get_ui_exclusion_regions(),
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

                self.fps_tracker = (
                    instant_fps
                )

            else:

                self.fps_tracker = (
                    self.fps_tracker * 0.85
                    + instant_fps * 0.15
                )

        self.last_frame_time = now

        self.update()

    # =========================================================
    # OpenCV Image → QImage
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

        # 必须 copy：
        # 否则 QImage 可能继续引用 Python ndarray 的内存。
        return qimage.copy()

    # =========================================================
    # UI 禁区 Debug
    # =========================================================

    def _draw_ui_exclusions(
        self,
        painter: QPainter,
    ):

        if not SHOW_UI_EXCLUSION_REGIONS:
            return

        regions = (
            self.vision
            .get_ui_exclusion_regions()
        )

        painter.setFont(
            QFont(
                "Consolas",
                8,
                QFont.Bold,
            )
        )

        for region in regions:

            name = region["name"]

            rx, ry, rw, rh = (
                region["rect"]
            )

            if rw <= 0 or rh <= 0:
                continue

            painter.setPen(
                QPen(
                    QColor(
                        255,
                        80,
                        80,
                        190,
                    ),
                    1,
                    Qt.DashLine,
                )
            )

            painter.setBrush(
                QBrush(
                    QColor(
                        255,
                        0,
                        0,
                        20,
                    )
                )
            )

            painter.drawRect(
                rx,
                ry,
                rw,
                rh,
            )

            painter.setPen(
                QColor(
                    255,
                    100,
                    100,
                )
            )

            painter.drawText(
                rx + 4,
                ry + 12,
                name,
            )

    # =========================================================
    # OCR Debug Preview
    # =========================================================

    def _draw_ocr_previews(
        self,
        painter: QPainter,
        panel_x: int,
        panel_y: int,
        panel_w: int,
    ):

        debug_images = (
            self.vision
            .ocr
            .get_debug_images()
        )

        # =====================================================
        # 标题
        # =====================================================

        title_y = (
            panel_y + 104
        )

        painter.setFont(
            QFont(
                "Microsoft YaHei",
                10,
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
            panel_x + 15,
            title_y,
            "OCR Processed Images",
        )

        # =====================================================
        # 图片区域
        # =====================================================

        image_y = (
            panel_y + 115
        )

        margin_left = (
            panel_x + 15
        )

        gap = OCR_PREVIEW_GAP

        total_width = (
            panel_w - 30
        )

        count = len(
            OCR_PREVIEW_FIELDS
        )

        if count <= 0:
            return

        cell_w = int(
            (
                total_width
                - gap * (count - 1)
            )
            / count
        )

        cell_h = (
            OCR_PREVIEW_CELL_H
        )

        for index, field in enumerate(
            OCR_PREVIEW_FIELDS
        ):

            x = (
                margin_left
                + index
                * (
                    cell_w
                    + gap
                )
            )

            y = image_y

            # =================================================
            # Cell
            # =================================================

            painter.setPen(
                QPen(
                    QColor(
                        90,
                        210,
                        240,
                        190,
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
                        180,
                    )
                )
            )

            painter.drawRect(
                x,
                y,
                cell_w,
                cell_h,
            )

            # =================================================
            # Label
            # =================================================

            label = (
                OCR_PREVIEW_LABELS.get(
                    field,
                    field,
                )
            )

            painter.setFont(
                QFont(
                    "Consolas",
                    7,
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
                x + 3,
                y + 11,
                label,
            )

            # =================================================
            # Image
            # =================================================

            image = (
                debug_images.get(
                    field
                )
            )

            if image is None:

                painter.setFont(
                    QFont(
                        "Consolas",
                        7,
                    )
                )

                painter.setPen(
                    QColor(
                        110,
                        110,
                        110,
                    )
                )

                painter.drawText(
                    x + 5,
                    y + 57,
                    "No image",
                )

                continue

            qimage = (
                self._build_qimage_from_cv(
                    image
                )
            )

            if qimage is None:
                continue

            available_w = (
                cell_w - 6
            )

            available_h = (
                cell_h - 18
            )

            if (
                available_w <= 0
                or available_h <= 0
            ):

                continue

            scaled = qimage.scaled(
                available_w,
                available_h,
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )

            image_x = (
                x
                + (
                    cell_w
                    - scaled.width()
                )
                // 2
            )

            image_y2 = (
                y
                + 14
                + (
                    available_h
                    - scaled.height()
                )
                // 2
            )

            painter.drawImage(
                image_x,
                image_y2,
                scaled,
            )

    # =========================================================
    # Paint
    # =========================================================

    def paintEvent(
        self,
        event,
    ):

        if self.render_data is None:
            return

        painter = QPainter(
            self
        )

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
        # UI 禁区
        # =====================================================

        self._draw_ui_exclusions(
            painter
        )

        # =====================================================
        # Result
        # =====================================================

        if result is None:

            state = None

            hud_raw = {}

            ship_name = "Unknown"

            max_speed = 0.0

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

        # =====================================================
        # Fixed HUD
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

            rect = (
                debug_regions[
                    name
                ]["rect"]
            )

            rx, ry, rw, rh = rect

            # ROI
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

            value = (
                hud_raw.get(
                    name,
                    "",
                )
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
        # Center Cross
        # =====================================================

        cross_size = CENTER_CROSS_SIZE

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
            self.cx - cross_size,
            self.cy,
            self.cx + cross_size,
            self.cy,
        )

        painter.drawLine(
            self.cx,
            self.cy - cross_size,
            self.cx,
            self.cy + cross_size,
        )

        # =====================================================
        # Target
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

            # =================================================
            # HP bar
            # =================================================

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

            # =================================================
            # HP bar anchor
            # =================================================

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

            # =================================================
            # HP → target center
            # =================================================

            painter.drawLine(
                hx_left,
                hy_top,
                target_cx,
                target_cy,
            )

            # =================================================
            # Target center
            # =================================================

            target_cross_size = (
                TARGET_CROSS_SIZE
            )

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
                target_cx
                - target_cross_size,
                target_cy,
                target_cx
                + target_cross_size,
                target_cy,
            )

            painter.drawLine(
                target_cx,
                target_cy
                - target_cross_size,
                target_cx,
                target_cy
                + target_cross_size,
            )

            # =================================================
            # Speed Ring
            # =================================================

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

            # =================================================
            # 8 sampling points
            # =================================================

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

            # =================================================
            # State Light
            # =================================================

            light_y = (
                target_cy
                - radius
                - self.vision
                .indicators
                .LIGHT_OFFSET
            )

            light_size = (
                LIGHT_DEBUG_SIZE
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
                target_cx
                - light_size // 2,
                light_y
                - light_size // 2,
                light_size,
                light_size,
            )

            # =================================================
            # Enemy Distance ROI
            # =================================================

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

                distance_text = (
                    hud_raw.get(
                        "enemy_distance",
                        "",
                    )
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
                    (
                        "Enemy Dist: "
                        f"{distance_text}"
                    ),
                )

        # =====================================================
        # Data Panel
        # =====================================================

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
                PANEL_X,
                PANEL_Y,
                PANEL_W,
                PANEL_H,
            ),
            8,
            8,
        )

        # =====================================================
        # Title
        # =====================================================

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
            PANEL_X + 15,
            PANEL_Y + 28,
            "WoWs AI Pre-Aim Data Inspector",
        )

        # =====================================================
        # Performance
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
            PANEL_X + 15,
            PANEL_Y + 48,
            (
                f"FPS: "
                f"{self.fps_tracker:4.1f} | "
                f"Vision: "
                f"{cost_ms:4.1f}ms"
            ),
        )

        painter.drawText(
            PANEL_X + 15,
            PANEL_Y + 64,
            (
                f"OCR: "
                f"{ocr_ms:4.1f}ms | "
                f"Field: "
                f"{ocr_field} | "
                f"Score: "
                f"{ocr_score:.2f}"
            ),
        )

        painter.drawText(
            PANEL_X + 15,
            PANEL_Y + 80,
            (
                "Raw OCR: "
                f"{ocr_text[:RAW_OCR_MAX_LENGTH]}"
            ),
        )

        # LOCKED / SEARCHING
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
            PANEL_X + PANEL_W - 90,
            PANEL_Y + 48,
            (
                "LOCKED"
                if locked
                else "SEARCHING"
            ),
        )

        # =====================================================
        # 分隔线
        # =====================================================

        painter.setPen(
            QColor(
                70,
                70,
                70,
            )
        )

        painter.drawLine(
            PANEL_X + 15,
            PANEL_Y + 92,
            PANEL_X + PANEL_W - 15,
            PANEL_Y + 92,
        )

        # =====================================================
        # OCR Images
        # =====================================================

        self._draw_ocr_previews(
            painter,
            PANEL_X,
            PANEL_Y,
            PANEL_W,
        )

        # =====================================================
        # Data
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

            if (
                state.direction_state
                == 1
            ):

                direction = "FWD"

            elif (
                state.direction_state
                == -1
            ):

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
                    PANEL_Y
                    + DATA_START_Y
                    + index
                    * DATA_ROW_HEIGHT
                )

                painter.setPen(
                    QColor(
                        130,
                        200,
                        255,
                    )
                )

                painter.drawText(
                    PANEL_X + 18,
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
                    PANEL_X + 160,
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
                PANEL_X + 18,
                PANEL_Y + DATA_START_Y,
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