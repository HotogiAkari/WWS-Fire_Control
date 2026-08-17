# recognition/vision_manager.py

import os
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from common.config_loader import ConfigLoader
from common.data_models import TargetState

from recognition.capture import ScreenCapturer
from recognition.indicator_parser import IndicatorParser
from recognition.ocr_parser import OCRParser


class VisionManager:
    """
    顶层视觉调度器。

    目标：

        native_overlay.py
            ↓
        VisionManager
            ├── Target / Indicator
            ├── OCR
            ├── MiniMap（以后）
            └── Ballistics（以后）

    UI 不直接关心底层算法实现。
    """

    def __init__(
        self,
        config: ConfigLoader,
    ):

        self.config = config

        # =====================================================
        # 当前模块
        # =====================================================

        self.capturer = ScreenCapturer()

        self.indicators = IndicatorParser()

        self.ocr = OCRParser()

        # =====================================================
        # 未来模块
        # =====================================================

        # 以后：
        #
        # from recognition.minimap_parser import MiniMapParser
        # self.minimap = MiniMapParser(...)
        #
        self.minimap = None

        # 以后：
        #
        # self.ballistics = BallisticsCalculator(...)
        #
        self.ballistics = None

        # =====================================================
        # 屏幕
        # =====================================================

        self.sw = int(
            self.capturer.monitor["width"]
        )

        self.sh = int(
            self.capturer.monitor["height"]
        )

        self.cx = self.sw // 2
        self.cy = self.sh // 2

        # =====================================================
        # 固定 HUD
        # =====================================================

        self.fixed_hud_regions = {

            "flight_time": (
                self.cx - 130,
                self.cy + 23,
                75,
                25,
            ),

            "aim_distance": (
                self.cx + 57,
                self.cy + 23,
                90,
                25,
            ),

            "ship_name": (
                self.cx - 300,
                self.cy + 83,
                270,
                30,
            ),

            "max_speed": (
                self.cx - 122,
                self.cy + 115,
                90,
                26,
            ),
        }

        # =====================================================
        # 目标切换距离
        # =====================================================

        self.TARGET_SWITCH_DISTANCE = 180.0

        # =====================================================
        # 当前目标
        # =====================================================

        self.current_anchor = None

    # =========================================================
    # 固定 HUD
    # =========================================================

    def get_fixed_regions(self) -> Dict:

        regions = dict(
            self.fixed_hud_regions
        )

        # 敌我角度也是固定 HUD
        angle_regions = (
            self.indicators
            .get_dynamic_ocr_rois(
                anchor=None,
                screen_cx=self.cx,
                screen_cy=self.cy,
            )
        )

        regions["enemy_angle"] = (
            angle_regions[
                "enemy_angle"
            ]
        )

        regions["our_angle"] = (
            angle_regions[
                "our_angle"
            ]
        )

        return regions

    # =========================================================
    # 动态区域
    # =========================================================

    def get_dynamic_regions(
        self,
        anchor,
    ) -> Dict:

        return (
            self.indicators
            .get_dynamic_ocr_rois(
                anchor=anchor,
                screen_cx=self.cx,
                screen_cy=self.cy,
            )
        )

    # =========================================================
    # OCR 区域
    # =========================================================

    def get_ocr_regions(
        self,
        anchor,
    ) -> Dict:

        regions = (
            self.get_fixed_regions()
        )

        dynamic = (
            self.get_dynamic_regions(
                anchor
            )
        )

        regions.update(
            dynamic
        )

        return regions

    # =========================================================
    # OCR 几何参数
    # =========================================================

    def get_ocr_geometry(self) -> Dict:

        return {

            "angle_crop_left":
                self.indicators.angle_crop_left,

            "angle_crop_top":
                self.indicators.angle_crop_top,

            "angle_crop_right":
                self.indicators.angle_crop_right,

            "angle_crop_bottom":
                self.indicators.angle_crop_bottom,
        }

    # =========================================================
    # Debug regions
    # =========================================================

    def get_debug_regions(
        self,
        anchor,
    ) -> Dict:

        regions = {}

        # -----------------------------------------------------
        # 固定 HUD
        # -----------------------------------------------------

        for name, rect in (
            self.get_fixed_regions()
        ).items():

            regions[name] = {
                "rect": rect,
                "type": "fixed",
            }

        # -----------------------------------------------------
        # 动态 ROI
        # -----------------------------------------------------

        if anchor is not None:

            for name, rect in (
                self.get_dynamic_regions(
                    anchor
                )
            ).items():

                if name in regions:
                    continue

                regions[name] = {
                    "rect": rect,
                    "type": "dynamic",
                }

        # -----------------------------------------------------
        # 目标血条
        # -----------------------------------------------------

        if anchor is not None:

            (
                cx,
                cy,
                hx,
                hy,
                hw,
                hh,
            ) = anchor

            regions[
                "target_hp_bar"
            ] = {
                "rect": (
                    hx,
                    hy,
                    hw,
                    hh,
                ),
                "type": "target",
            }

        return regions

    # =========================================================
    # 目标切换通知
    # =========================================================

    def notify_target_switch(self):

        self.ocr.force_static_refresh()

    # =========================================================
    # 处理一帧
    # =========================================================

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_hsv: np.ndarray,
        anchor,
    ) -> Dict:

        self.current_anchor = anchor

        # =====================================================
        # OCR
        # =====================================================

        self.ocr.update(
            frame_bgr=frame_bgr,
            roi_map=self.get_ocr_regions(
                anchor
            ),
            geometry=self.get_ocr_geometry(),
            locked=(
                anchor is not None
            ),
        )

        # =====================================================
        # MiniMap
        # =====================================================

        minimap_result = None

        if self.minimap is not None:

            minimap_result = (
                self.minimap.process(
                    frame_bgr
                )
            )

        # =====================================================
        # 没锁定
        # =====================================================

        if anchor is None:

            data = (
                self.ocr.get_cache()
            )

            return {
                "state": None,

                "ship_name":
                    data["ship_name"],

                "max_speed":
                    data["max_speed"],

                "hud_raw": {},

                "target": None,

                "minimap":
                    minimap_result,
            }

        # =====================================================
        # 航速
        # =====================================================

        (
            speed_fraction,
            direction,
            sample_pts,
        ) = (
            self.indicators
            .sample_speed_ring(
                frame_hsv,
                anchor[0],
                anchor[1],
            )
        )

        # =====================================================
        # OCR
        # =====================================================

        data = (
            self.ocr.get_cache()
        )

        flight_time = float(
            data["flight_time"]
        )

        aim_distance = float(
            data["aim_distance"]
        )

        enemy_distance = float(
            data["enemy_distance"]
        )

        enemy_angle = float(
            data["enemy_angle"]
        )

        our_angle = float(
            data["our_angle"]
        )

        ship_name = str(
            data["ship_name"]
        )

        max_speed = float(
            data["max_speed"]
        )

        # =====================================================
        # 格式化
        #
        # 单位完全由程序添加。
        # OCR 不负责单位。
        # =====================================================

        hud_raw = {

            "flight_time":
                f"{flight_time:.2f} s.",

            "aim_distance":
                f"{aim_distance:.2f} km",

            "ship_name":
                ship_name,

            "max_speed":
                f"{max_speed:.1f} kts",

            "enemy_distance":
                f"{enemy_distance:.1f} 公里",

            "enemy_angle":
                f"{enemy_angle:.1f}°",

            "our_angle":
                f"{our_angle:.1f}°",
        }

        # =====================================================
        # TargetState
        # =====================================================

        state = TargetState(

            timestamp=time.time(),

            # 动态敌舰距离
            distance=enemy_distance,

            # 固定 HUD 中的落点距离
            aiming_distance=aim_distance,

            flight_time=flight_time,

            relative_angle=enemy_angle,

            minimap_x=0.0,

            minimap_y=0.0,

            speed_fraction=speed_fraction,

            direction_state=direction,
        )

        # =====================================================
        # 返回统一结构
        # =====================================================

        return {

            "state":
                state,

            "ship_name":
                ship_name,

            "max_speed":
                max_speed,

            "hud_raw":
                hud_raw,

            "target": {

                "anchor":
                    anchor,

                "sample_pts":
                    sample_pts,

                "speed_fraction":
                    speed_fraction,

                "direction":
                    direction,
            },

            "minimap":
                minimap_result,
        }

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        if self.ocr is not None:
            self.ocr.close()

        if self.capturer is not None:
            self.capturer.close()

        if self.minimap is not None:

            try:
                self.minimap.close()
            except Exception:
                pass

        if self.ballistics is not None:

            try:
                self.ballistics.close()
            except Exception:
                pass