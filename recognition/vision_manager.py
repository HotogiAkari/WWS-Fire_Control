import os
import sys
import time
from typing import Any, Dict, Optional

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
# Vision 调参区
# ============================================================


# ============================================================
# 1. 固定 HUD ROI
#
# 格式：
#
#     (相对于屏幕中心的 x,
#      相对于屏幕中心的 y,
#      width,
#      height)
#
# 当前由于 WoWs HUD 位于屏幕中央，
# 使用相对中心坐标。
# ============================================================

FIXED_HUD_REGIONS = {

    "flight_time": (
        -130,
        23,
        75,
        25,
    ),

    "aim_distance": (
        57,
        23,
        90,
        25,
    ),

    "ship_name": (
        -300,
        83,
        270,
        30,
    ),

    "max_speed": (
        -122,
        115,
        90,
        26,
    ),
}


# ============================================================
# 2. 是否使用屏幕中心相对坐标
# ============================================================

FIXED_HUD_RELATIVE_TO_CENTER = True


# ============================================================
# 3. 目标切换距离
#
# 当前 anchor 与上一帧 anchor 的距离超过此值，
# 认为可能发生目标切换。
#
# 调大：
#     减少误触发
#
# 调小：
#     更容易刷新静态舰船数据
# ============================================================

TARGET_SWITCH_DISTANCE = 180.0


# ============================================================
# 4. OCR 配置
#
# 这里是 VisionManager 对 OCRParser 的正式运行配置。
#
# 修改 OCR 参数时优先修改这里。
# ============================================================

OCR_CONFIG = {

    # ========================================================
    # 基础放大
    # ========================================================

    "numeric_scale":
        2.0,

    "normal_scale_small":
        2.0,

    "normal_scale_large":
        1.5,

    "normal_text_height_threshold":
        30,

    # ========================================================
    # HSV
    # ========================================================

    "binary_max_saturation":
        100,

    # ========================================================
    # 局部背景
    # ========================================================

    "background_blur_kernel":
        (15, 15),

    # ========================================================
    # Top-Hat
    # ========================================================

    "tophat_kernel": (15, 15),

    "tophat_gain": 2.0,

    "tophat_post_blur_kernel": (3, 3),

    # ========================================================
    # Adaptive Threshold
    # ========================================================

    "adaptive_block_size": 11,

    "adaptive_c": -15,

    # ========================================================
    # 去噪
    # ========================================================

    "noise_open_kernel": (2, 2),

    "noise_open_iterations": 1,

    "post_morph_kernel": (2, 2),

    "post_morph_close_iterations": 1,

    "min_foreground_component_area": 15,

    # ========================================================
    # 相对角度圆环
    #
    # ⭐ 后续主要调这里
    # ========================================================

    "angle_ring_center_x":
        18.0,

    "angle_ring_center_y":
        18.0,

    "angle_ring_radius":
        13.0,

    "angle_ring_thickness":
        3.0,

    "angle_ring_mask_dilate":
        0,

    # ========================================================
    # Border
    # ========================================================

    "border_top":
        8,

    "border_bottom":
        8,

    "border_left":
        12,

    "border_right":
        12,
}

CROSSHAIR_CONFIG = {
    
    # 截图ROI：(x偏移, y偏移, 宽, 高)
    # 示例：从中心向右偏40像素开始搜索，宽度400，向上偏移12，高度8。
    "roi": (40, -10, 800, 10),
    
    # 亮度阈值(0-255)。刻度中心是纯白色，调高一点能无视深色背景
    "white_thresh": 240,
    
    # 垂直特征提取内核高度。必须小于roi的高度，但大于噪点高度
    "vertical_kernel_h": 4,
    
    # 刻度最大允许粗细(像素)
    "tick_max_width": 5,
    
    # 刻度间距的合理范围，用于过滤杂波
    "min_spacing": 20,
    "max_spacing": 400,
}

# ============================================================
# Imports
# ============================================================

from common.config_loader import ConfigLoader
from common.data_models import TargetState

from recognition.capture import ScreenCapturer
from recognition.indicator_parser import IndicatorParser
from recognition.ocr_parser import OCRParser
from recognition.crosshair_parser import CrosshairParser

# ============================================================
# VisionManager
# ============================================================

class VisionManager:
    """
    顶层视觉统合器。

    负责：

        - 获取基础视觉结果
        - 调用 OCR
        - 调用扩展视觉模块
        - 统一输出

    不负责具体算法。
    """

    def __init__(
        self,
        config: ConfigLoader,
    ):

        self.config = config

        # =====================================================
        # Core
        # =====================================================

        self.capturer = (
            ScreenCapturer()
        )

        self.indicators = (
            IndicatorParser()
        )

        self.ocr = (
            OCRParser(
                **OCR_CONFIG
            )
        )

        # =====================================================
        # Extension Modules
        #
        # 以后：
        #
        # self.register_module(
        #     "minimap",
        #     MiniMapParser(),
        # )
        # =====================================================

        self.modules: Dict[
            str,
            Any,
        ] = {}

        self.register_module(
            "crosshair",
            CrosshairParser(**CROSSHAIR_CONFIG),
        )

        # =====================================================
        # Screen
        # =====================================================

        self.sw = int(
            self.capturer.monitor[
                "width"
            ]
        )

        self.sh = int(
            self.capturer.monitor[
                "height"
            ]
        )

        self.cx = self.sw // 2
        self.cy = self.sh // 2

        # =====================================================
        # 固定 HUD
        # =====================================================

        if FIXED_HUD_RELATIVE_TO_CENTER:

            self.fixed_hud_regions = {

                name: (

                    self.cx + rect[0],

                    self.cy + rect[1],

                    rect[2],

                    rect[3],
                )

                for (
                    name,
                    rect,
                ) in FIXED_HUD_REGIONS.items()
            }

        else:

            self.fixed_hud_regions = dict(
                FIXED_HUD_REGIONS
            )

        # =====================================================
        # Target switch
        # =====================================================

        self.TARGET_SWITCH_DISTANCE = (
            TARGET_SWITCH_DISTANCE
        )

        # =====================================================
        # Current anchor
        # =====================================================

        self.current_anchor = None

    # =========================================================
    # Register module
    # =========================================================

    def register_module(
        self,
        name: str,
        module: Any,
    ):

        if not name:

            raise ValueError(
                "模块名称不能为空"
            )

        if module is None:

            raise ValueError(
                f"模块 {name} 不能为 None"
            )

        process = getattr(
            module,
            "process",
            None,
        )

        if not callable(process):

            raise TypeError(
                f"模块 {name} "
                "必须提供 process() 方法"
            )

        self.modules[name] = module

    # =========================================================
    # Unregister
    # =========================================================

    def unregister_module(
        self,
        name: str,
    ):

        module = self.modules.pop(
            name,
            None,
        )

        if module is None:
            return

        close = getattr(
            module,
            "close",
            None,
        )

        if callable(close):

            try:

                close()

            except Exception:

                pass

    # =========================================================
    # Get module
    # =========================================================

    def get_module(
        self,
        name: str,
    ) -> Optional[Any]:

        return self.modules.get(
            name
        )

    # =========================================================
    # Run extension modules
    # =========================================================

    def _run_registered_modules(
        self,
        frame_bgr: np.ndarray,
        frame_hsv: np.ndarray,
        context: Dict,
    ) -> Dict:

        outputs = {}

        for (
            name,
            module,
        ) in self.modules.items():

            try:

                outputs[name] = (
                    module.process(
                        frame_bgr=frame_bgr,
                        frame_hsv=frame_hsv,
                        context=context,
                    )
                )

            except Exception as exc:

                print(
                    f"[Vision] "
                    f"模块 {name} 执行失败: "
                    f"{exc}"
                )

                outputs[name] = {

                    "error":
                        str(exc),
                }

        return outputs

    # =========================================================
    # Fixed Regions
    # =========================================================

    def get_fixed_regions(
        self,
    ) -> Dict:

        regions = dict(
            self.fixed_hud_regions
        )

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
    # Dynamic Regions
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
    # OCR Regions
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
    # OCR Geometry
    # =========================================================

    def get_ocr_geometry(
        self,
    ) -> Dict:

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
    # Debug Regions
    # =========================================================

    def get_debug_regions(
        self,
        anchor,
    ) -> Dict:

        regions = {}

        for (
            name,
            rect,
        ) in self.get_fixed_regions().items():

            regions[name] = {

                "rect":
                    rect,

                "type":
                    "fixed",
            }

        if anchor is not None:

            for (
                name,
                rect,
            ) in self.get_dynamic_regions(
                anchor
            ).items():

                if name in regions:
                    continue

                regions[name] = {

                    "rect":
                        rect,

                    "type":
                        "dynamic",
                }

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

                "type":
                    "target",
            }

        return regions

    # =========================================================
    # UI Exclusion
    # =========================================================

    def get_ui_exclusion_regions(
        self,
    ):

        return (
            self.indicators
            .get_ui_exclusion_regions()
        )

    # =========================================================
    # Target switch
    # =========================================================

    def notify_target_switch(
        self,
    ):

        self.ocr.force_static_refresh()

    # =========================================================
    # Process Frame
    # =========================================================

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_hsv: np.ndarray,
        anchor=None,
        previous_anchor=None,
    ) -> Dict:

        # =====================================================
        # 1. Indicator
        # =====================================================

        indicator_result = (
            self.indicators.process(
                frame_bgr=frame_bgr,
                frame_hsv=frame_hsv,
                previous_anchor=previous_anchor,
                screen_cx=self.cx,
                screen_cy=self.cy,
            )
        )

        detected_anchor = (
            indicator_result[
                "anchor"
            ]
        )

        # =====================================================
        # 如果 Worker 已经检测好 anchor，
        # 沿用 Worker 的结果。
        # =====================================================

        if anchor is not None:

            detected_anchor = anchor

        self.current_anchor = (
            detected_anchor
        )

        # =====================================================
        # 2. OCR
        # =====================================================

        ocr_regions = (
            self.get_ocr_regions(
                detected_anchor
            )
        )

        ocr_cache = (
            self.ocr.process(
                frame_bgr=frame_bgr,
                roi_map=ocr_regions,
                geometry=self.get_ocr_geometry(),
                locked=(
                    detected_anchor is not None
                ),
            )
        )

        # =====================================================
        # 3. Context
        # =====================================================

        context = {

            "timestamp":
                time.time(),

            "screen": {

                "width":
                    self.sw,

                "height":
                    self.sh,

                "cx":
                    self.cx,

                "cy":
                    self.cy,
            },

            "anchor":
                detected_anchor,

            "indicator":
                indicator_result,

            "ocr":
                ocr_cache,

            "ocr_regions":
                ocr_regions,

            "ui_exclusion_regions":
                self.get_ui_exclusion_regions(),
        }

        # =====================================================
        # 4. Extension Modules
        # =====================================================

        module_outputs = (
            self._run_registered_modules(
                frame_bgr=frame_bgr,
                frame_hsv=frame_hsv,
                context=context,
            )
        )

        # =====================================================
        # 5. 未锁定
        # =====================================================

        if detected_anchor is None:

            return {

                "timestamp":
                    time.time(),

                "locked":
                    False,

                "state":
                    None,

                "ship_name":
                    ocr_cache[
                        "ship_name"
                    ],

                "max_speed":
                    ocr_cache[
                        "max_speed"
                    ],

                "hud_raw":
                    {},

                "target":
                    None,

                "indicator":
                    indicator_result,

                "ocr":
                    ocr_cache,

                "modules":
                    module_outputs,
            }

        # =====================================================
        # 6. Cache 数据
        # =====================================================

        speed_fraction = float(
            indicator_result[
                "speed_fraction"
            ]
        )

        direction = int(
            indicator_result[
                "direction"
            ]
        )

        flight_time = float(
            ocr_cache[
                "flight_time"
            ]
        )

        aim_distance = float(
            ocr_cache[
                "aim_distance"
            ]
        )

        enemy_distance = float(
            ocr_cache[
                "enemy_distance"
            ]
        )

        enemy_angle = float(
            ocr_cache[
                "enemy_angle"
            ]
        )

        our_angle = float(
            ocr_cache[
                "our_angle"
            ]
        )

        ship_name = str(
            ocr_cache[
                "ship_name"
            ]
        )

        max_speed = float(
            ocr_cache[
                "max_speed"
            ]
        )

        # =====================================================
        # 7. HUD Raw
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
        # 8. TargetState
        # =====================================================

        state = TargetState(

            timestamp=
                time.time(),

            distance=
                enemy_distance,

            aiming_distance=
                aim_distance,

            flight_time=
                flight_time,

            relative_angle=
                enemy_angle,

            minimap_x=0.0,

            minimap_y=0.0,

            speed_fraction=
                speed_fraction,

            direction_state=
                direction,
        )

        # =====================================================
        # 9. Unified result
        # =====================================================

        return {

            "timestamp":
                time.time(),

            "locked":
                True,

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
                    detected_anchor,

                "sample_pts":
                    indicator_result[
                        "sample_pts"
                    ],

                "speed_fraction":
                    speed_fraction,

                "direction":
                    direction,
            },

            "indicator":
                indicator_result,

            "ocr":
                ocr_cache,

            "modules":
                module_outputs,
        }

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        if self.ocr is not None:

            self.ocr.close()

        if self.capturer is not None:

            self.capturer.close()

        for (
            name,
            module,
        ) in list(
            self.modules.items()
        ):

            close = getattr(
                module,
                "close",
                None,
            )

            if callable(close):

                try:

                    close()

                except Exception:

                    pass

        self.modules.clear()