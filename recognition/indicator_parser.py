from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 类型
# ============================================================

Anchor = Tuple[
    int,  # center x
    int,  # center y
    int,  # hp left
    int,  # hp top
    int,  # hp width
    int,  # hp height
]

# ============================================================
# 1. 血条 HSV 颜色范围
#
# 用于从游戏画面中寻找红色血条。
#
# OpenCV HSV：
#
#     H: 0 ~ 179
#     S: 0 ~ 255
#     V: 0 ~ 255
#
# 如果血条无法检测：
#
#     适当扩大 H / S / V 范围。
#
# 如果误检测很多：
#
#     缩小范围。
# ============================================================

RED_LOWER_1 = (0, 140, 140)
RED_UPPER_1 = (10, 255, 255)

RED_LOWER_2 = (170, 140, 140)
RED_UPPER_2 = (180, 255, 255)


# ============================================================
# 2. 状态灯 HSV
#
# 用于确认候选血条上方确实存在游戏锁定状态灯。
# ============================================================

GREEN_HSV_MIN = (35, 100, 100)
GREEN_HSV_MAX = (85, 255, 255)

RED_HSV_MIN_1 = (0, 100, 100)
RED_HSV_MAX_1 = (10, 255, 255)

RED_HSV_MIN_2 = (170, 100, 100)
RED_HSV_MAX_2 = (180, 255, 255)

GREY_HSV_MIN = (0, 0, 60)
GREY_HSV_MAX = (180, 45, 185)


# ============================================================
# 3. 血条 → 目标圆心偏移
#
# 血条左上角：
#
#     (hp_x, hp_y)
#
# 目标圆心：
#
#     (hp_x + OFFSET_X,
#      hp_y - OFFSET_Y)
#
# 如果目标中心框整体偏左/右：
#
#     调 OFFSET_X
#
# 如果整体偏上/下：
#
#     调 OFFSET_Y
# ============================================================

OFFSET_X = 46
OFFSET_Y = 21


# ============================================================
# 4. 航速圆环
#
# RING_RADIUS：
#     航速圆环半径。
#
# LIGHT_OFFSET：
#     状态灯相对于圆环顶部的偏移。
# ============================================================

RING_RADIUS = 37
LIGHT_OFFSET = 8


# ============================================================
# 5. 动态目标搜索范围
#
# 当上一帧已经找到目标后，
# 下一帧优先只在目标附近搜索。
#
# 调大：
#     更不容易丢目标
#     但搜索区域更大
#
# 调小：
#     更快
#     但目标移动太快时可能丢失
# ============================================================

TRACK_SEARCH_RADIUS_X = 140
TRACK_SEARCH_RADIUS_Y = 100


# ============================================================
# 6. 航速圆环采样点
#
# 格式：
#
#     (角度, 半径)
#
# 一共 8 个点。
#
# 如果游戏 HUD 圆环位置改变，
# 可以重新调整这些点。
# ============================================================

RING_POINTS = [

    (292.5, 36.0),  # 1
    (340.0, 37.0),  # 2
    (20.0, 37.0),   # 3
    (50.0, 37.0),   # 4
    (130.0, 37.0),  # 5
    (170.0, 37.0),  # 6
    (190.0, 37.0),  # 7
    (247.5, 36.0),  # 8
]


# ============================================================
# 7. 航速采样
#
# SAMPLE_PATCH_RADIUS：
#     每个采样点周围取多少像素进行颜色判断。
#
# SAMPLE_GREEN_THRESHOLD：
#     绿色像素占 patch 的比例达到多少才认为是前进。
#
# SAMPLE_RED_THRESHOLD：
#     红色像素占 patch 的比例达到多少才认为是倒退。
# ============================================================

SAMPLE_PATCH_RADIUS = 1

SAMPLE_GREEN_THRESHOLD = 0.35
SAMPLE_RED_THRESHOLD = 0.35


# ============================================================
# 8. 敌舰距离 ROI
#
# 相对于：
#
#     敌舰血条左上角
#
# ROI：
#
#     x = hp_x + ENEMY_DISTANCE_OFFSET_X
#     y = hp_y + ENEMY_DISTANCE_OFFSET_Y
#
# width / height：
#     OCR 截取区域大小。
#
# 如果距离文字被切掉：
#
#     增大 width / height
#
# 如果周围干扰太多：
#
#     减小 width / height
# ============================================================

ENEMY_DISTANCE_OFFSET_X = 0
ENEMY_DISTANCE_OFFSET_Y = 33

ENEMY_DISTANCE_WIDTH = 95
ENEMY_DISTANCE_HEIGHT = 21


# ============================================================
# 9. 敌方角度 ROI
#
# 相对于屏幕中心。
#
# 格式：
#
#     x = screen_cx + offset_x
#     y = screen_cy + offset_y
#
# 如果整个框左右/上下偏移：
#     调 offset_x / offset_y
#
# 如果框太大：
#     减小 width / height
# ============================================================

ENEMY_ANGLE_OFFSET_X = -17
ENEMY_ANGLE_OFFSET_Y = 160

ENEMY_ANGLE_WIDTH = 36
ENEMY_ANGLE_HEIGHT = 36


# ============================================================
# 10. 我方角度 ROI
# ============================================================

OUR_ANGLE_OFFSET_X = -17
OUR_ANGLE_OFFSET_Y = 294

OUR_ANGLE_WIDTH = 36
OUR_ANGLE_HEIGHT = 36


# ============================================================
# 11. 相对角度裁切
#
# 原始角度 ROI：
#
#     36 × 36
#
# 实际 OCR 前：
#
#     左边裁 ANGLE_CROP_LEFT
#     上边裁 ANGLE_CROP_TOP
#     右边裁 ANGLE_CROP_RIGHT
#     下边裁 ANGLE_CROP_BOTTOM
#
# 用途：
#
#     去除数字周围的圆环。
#
# 如果圆环残留：
#
#     适当增大。
#
# 如果数字被切掉：
#
#     适当减小。
#
# 建议一次只改 1~2 px。
# ============================================================

ANGLE_CROP_LEFT = 10
ANGLE_CROP_TOP = 10
ANGLE_CROP_RIGHT = 10
ANGLE_CROP_BOTTOM = 10


# ============================================================
# 12. UI 禁区
#
# 格式：
#
#     (left, top, width, height)
#
# 这些区域不会参与敌舰血条候选检测。
#
# 当前预留 7 个：
#
#     1. 左侧成员板
#     2. 右侧成员板
#     3. 顶部计分条
#     4. 左下角罗盘
#     5. 底部装备栏
#     6. 右下角小地图
#     7. 右上角成就
#
# 数量不限。
#
# 以后添加：
#
#     EXCLUSION_ZONES.append(...)
#
# 即可。
#
# 注意：
#
#     目前坐标全部为 0，
#     防止未知分辨率下错误屏蔽 HUD。
# ============================================================

UI_EXCLUSION_REGIONS = [

    {
        "name": "left_team_panel",
        "rect": (
            0,
            0,
            50,
            50,
        ),
    },

    {
        "name": "right_team_panel",
        "rect": (
            0,
            0,
            50,
            50,
        ),
    },

    {
        "name": "top_score",
        "rect": (
            0,
            0,
            0,
            0,
        ),
    },

    {
        "name": "bottom_compass",
        "rect": (
            0,
            0,
            0,
            0,
        ),
    },

    {
        "name": "bottom_equipment",
        "rect": (
            0,
            0,
            0,
            0,
        ),
    },

    {
        "name": "right_minimap",
        "rect": (
            0,
            0,
            0,
            0,
        ),
    },

    {
        "name": "top_right_achievement",
        "rect": (
            0,
            0,
            0,
            0,
        ),
    },
]


# ============================================================
# IndicatorParser
# ============================================================

class IndicatorParser:
    """
    游戏 HUD 几何 / 颜色解析器。

    负责：

        - 血条搜索
        - 目标圆心
        - 航速
        - 动态 OCR ROI
        - UI 禁区

    不负责：

        - OCR
        - 小地图
        - 弹道计算
    """

    def __init__(self):

        # =====================================================
        # 血条颜色
        # =====================================================

        self.red_lower1 = np.array(
            RED_LOWER_1,
            dtype=np.uint8,
        )

        self.red_upper1 = np.array(
            RED_UPPER_1,
            dtype=np.uint8,
        )

        self.red_lower2 = np.array(
            RED_LOWER_2,
            dtype=np.uint8,
        )

        self.red_upper2 = np.array(
            RED_UPPER_2,
            dtype=np.uint8,
        )

        # =====================================================
        # 状态灯
        # =====================================================

        self.green_hsv_min = np.array(
            GREEN_HSV_MIN,
            dtype=np.uint8,
        )

        self.green_hsv_max = np.array(
            GREEN_HSV_MAX,
            dtype=np.uint8,
        )

        self.red_hsv_min1 = np.array(
            RED_HSV_MIN_1,
            dtype=np.uint8,
        )

        self.red_hsv_max1 = np.array(
            RED_HSV_MAX_1,
            dtype=np.uint8,
        )

        self.red_hsv_min2 = np.array(
            RED_HSV_MIN_2,
            dtype=np.uint8,
        )

        self.red_hsv_max2 = np.array(
            RED_HSV_MAX_2,
            dtype=np.uint8,
        )

        self.grey_hsv_min = np.array(
            GREY_HSV_MIN,
            dtype=np.uint8,
        )

        self.grey_hsv_max = np.array(
            GREY_HSV_MAX,
            dtype=np.uint8,
        )

        # =====================================================
        # 几何参数
        # =====================================================

        self.OFFSET_X = OFFSET_X
        self.OFFSET_Y = OFFSET_Y

        self.RING_RADIUS = RING_RADIUS
        self.LIGHT_OFFSET = LIGHT_OFFSET

        self.TRACK_SEARCH_RADIUS_X = (
            TRACK_SEARCH_RADIUS_X
        )

        self.TRACK_SEARCH_RADIUS_Y = (
            TRACK_SEARCH_RADIUS_Y
        )

        self.RING_POINTS = list(
            RING_POINTS
        )

        self.SAMPLE_PATCH_RADIUS = (
            SAMPLE_PATCH_RADIUS
        )

        self.SAMPLE_GREEN_THRESHOLD = (
            SAMPLE_GREEN_THRESHOLD
        )

        self.SAMPLE_RED_THRESHOLD = (
            SAMPLE_RED_THRESHOLD
        )

        self.enemy_distance_offset_x = (
            ENEMY_DISTANCE_OFFSET_X
        )

        self.enemy_distance_offset_y = (
            ENEMY_DISTANCE_OFFSET_Y
        )

        self.enemy_distance_width = (
            ENEMY_DISTANCE_WIDTH
        )

        self.enemy_distance_height = (
            ENEMY_DISTANCE_HEIGHT
        )

        self.enemy_angle_offset_x = (
            ENEMY_ANGLE_OFFSET_X
        )

        self.enemy_angle_offset_y = (
            ENEMY_ANGLE_OFFSET_Y
        )

        self.enemy_angle_width = (
            ENEMY_ANGLE_WIDTH
        )

        self.enemy_angle_height = (
            ENEMY_ANGLE_HEIGHT
        )

        self.our_angle_offset_x = (
            OUR_ANGLE_OFFSET_X
        )

        self.our_angle_offset_y = (
            OUR_ANGLE_OFFSET_Y
        )

        self.our_angle_width = (
            OUR_ANGLE_WIDTH
        )

        self.our_angle_height = (
            OUR_ANGLE_HEIGHT
        )

        self.angle_crop_left = (
            ANGLE_CROP_LEFT
        )

        self.angle_crop_top = (
            ANGLE_CROP_TOP
        )

        self.angle_crop_right = (
            ANGLE_CROP_RIGHT
        )

        self.angle_crop_bottom = (
            ANGLE_CROP_BOTTOM
        )

        # =====================================================
        # UI 禁区
        # =====================================================

        self.ui_exclusion_regions = [
            dict(region)
            for region in UI_EXCLUSION_REGIONS
        ]

        # =====================================================
        # 预计算航速采样点
        # =====================================================

        self.ring_sample_offsets = []

        for (
            angle_deg,
            radius,
        ) in self.RING_POINTS:

            angle_rad = np.deg2rad(
                angle_deg
            )

            dx = int(
                round(
                    radius
                    * np.cos(angle_rad)
                )
            )

            dy = int(
                round(
                    radius
                    * np.sin(angle_rad)
                )
            )

            self.ring_sample_offsets.append(
                (
                    dx,
                    dy,
                    float(angle_deg),
                    float(radius),
                )
            )

    # =========================================================
    # UI 禁区
    # =========================================================

    def get_ui_exclusion_regions(
        self,
    ) -> List[Dict]:

        return [
            dict(region)
            for region in
            self.ui_exclusion_regions
        ]

    def point_in_ui_exclusion(
        self,
        x: int,
        y: int,
    ) -> bool:

        for region in (
            self.ui_exclusion_regions
        ):

            rx, ry, rw, rh = (
                region["rect"]
            )

            if rw <= 0 or rh <= 0:
                continue

            if (
                rx <= x < rx + rw
                and
                ry <= y < ry + rh
            ):

                return True

        return False

    def rect_intersects_ui_exclusion(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> bool:

        if w <= 0 or h <= 0:
            return False

        x2 = x + w
        y2 = y + h

        for region in (
            self.ui_exclusion_regions
        ):

            rx, ry, rw, rh = (
                region["rect"]
            )

            if rw <= 0 or rh <= 0:
                continue

            rx2 = rx + rw
            ry2 = ry + rh

            if (
                x < rx2
                and x2 > rx
                and y < ry2
                and y2 > ry
            ):

                return True

        return False

    # =========================================================
    # 状态灯
    # =========================================================

    def evaluate_light_patch(
        self,
        frame_hsv: np.ndarray,
        cx: int,
        ly: int,
        patch_radius: int = 3,
    ) -> bool:

        h, w = frame_hsv.shape[:2]

        y1 = max(
            0,
            ly - patch_radius,
        )

        y2 = min(
            h,
            ly + patch_radius + 1,
        )

        x1 = max(
            0,
            cx - patch_radius,
        )

        x2 = min(
            w,
            cx + patch_radius + 1,
        )

        patch = frame_hsv[
            y1:y2,
            x1:x2,
        ]

        if patch.size == 0:
            return False

        if np.std(
            patch[:, :, 2]
        ) > 35.0:

            return False

        total = (
            patch.shape[0]
            * patch.shape[1]
        )

        # 绿色
        green_mask = cv2.inRange(
            patch,
            self.green_hsv_min,
            self.green_hsv_max,
        )

        if (
            cv2.countNonZero(
                green_mask
            )
            / total
            >= 0.60
        ):

            return True

        # 红色
        red1 = cv2.inRange(
            patch,
            self.red_hsv_min1,
            self.red_hsv_max1,
        )

        red2 = cv2.inRange(
            patch,
            self.red_hsv_min2,
            self.red_hsv_max2,
        )

        red = cv2.bitwise_or(
            red1,
            red2,
        )

        if (
            cv2.countNonZero(
                red
            )
            / total
            >= 0.60
        ):

            return True

        # 灰色
        grey = cv2.inRange(
            patch,
            self.grey_hsv_min,
            self.grey_hsv_max,
        )

        if (
            cv2.countNonZero(
                grey
            )
            / total
            >= 0.60
        ):

            return True

        return False

    # =========================================================
    # 搜索血条
    # =========================================================

    def _search_anchor_in_region(
        self,
        frame_hsv: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> Optional[Anchor]:

        height, width = (
            frame_hsv.shape[:2]
        )

        left = max(
            0,
            int(left),
        )

        top = max(
            0,
            int(top),
        )

        right = min(
            width,
            int(right),
        )

        bottom = min(
            height,
            int(bottom),
        )

        if (
            right <= left
            or bottom <= top
        ):

            return None

        roi = frame_hsv[
            top:bottom,
            left:right,
        ]

        mask1 = cv2.inRange(
            roi,
            self.red_lower1,
            self.red_upper1,
        )

        mask2 = cv2.inRange(
            roi,
            self.red_lower2,
            self.red_upper2,
        )

        mask = cv2.bitwise_or(
            mask1,
            mask2,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        for contour in contours:

            x, y, w, h = (
                cv2.boundingRect(
                    contour
                )
            )

            if w < 15:
                continue

            if h < 2 or h > 8:
                continue

            hx_left = (
                left + x
            )

            hy_top = (
                top + y
            )

            # UI 禁区
            if self.rect_intersects_ui_exclusion(
                hx_left,
                hy_top,
                w,
                h,
            ):

                continue

            cx = (
                hx_left
                + self.OFFSET_X
            )

            cy = (
                hy_top
                - self.OFFSET_Y
            )

            light_y = (
                cy
                - self.RING_RADIUS
                - self.LIGHT_OFFSET
            )

            if light_y < 0:
                continue

            if cx < 0 or cx >= width:
                continue

            if not self.evaluate_light_patch(
                frame_hsv,
                cx,
                light_y,
            ):

                continue

            return (
                cx,
                cy,
                hx_left,
                hy_top,
                w,
                h,
            )

        return None

    # =========================================================
    # 动态跟踪
    # =========================================================

    def find_locked_anchor(
        self,
        frame_hsv: np.ndarray,
        previous_anchor: Optional[Anchor] = None,
    ) -> Optional[Anchor]:

        height, width = (
            frame_hsv.shape[:2]
        )

        # =====================================================
        # 局部搜索
        # =====================================================

        if previous_anchor is not None:

            prev_cx = previous_anchor[0]
            prev_cy = previous_anchor[1]

            prev_hx = (
                prev_cx
                - self.OFFSET_X
            )

            prev_hy = (
                prev_cy
                + self.OFFSET_Y
            )

            # ★ 修复：
            # 必须把 frame_hsv 作为第一参数传入。
            anchor = (
                self._search_anchor_in_region(

                    frame_hsv,

                    prev_hx
                    - self.TRACK_SEARCH_RADIUS_X,

                    prev_hy
                    - self.TRACK_SEARCH_RADIUS_Y,

                    prev_hx
                    + self.TRACK_SEARCH_RADIUS_X,

                    prev_hy
                    + self.TRACK_SEARCH_RADIUS_Y,
                )
            )

            if anchor is not None:
                return anchor

        # =====================================================
        # 全屏 fallback
        # =====================================================

        return (
            self._search_anchor_in_region(
                frame_hsv,
                0,
                0,
                width,
                height,
            )
        )

    # =========================================================
    # OCR ROI
    # =========================================================

    def get_dynamic_ocr_rois(
        self,
        anchor: Optional[Anchor],
        screen_cx: int,
        screen_cy: int,
    ) -> Dict:

        regions = {

            "enemy_angle": (
                screen_cx
                + self.enemy_angle_offset_x,

                screen_cy
                + self.enemy_angle_offset_y,

                self.enemy_angle_width,
                self.enemy_angle_height,
            ),

            "our_angle": (
                screen_cx
                + self.our_angle_offset_x,

                screen_cy
                + self.our_angle_offset_y,

                self.our_angle_width,
                self.our_angle_height,
            ),
        }

        if anchor is None:
            return regions

        (
            cx,
            cy,
            hx_left,
            hy_top,
            hw,
            hh,
        ) = anchor

        regions["enemy_distance"] = (

            hx_left
            + self.enemy_distance_offset_x,

            hy_top
            + self.enemy_distance_offset_y,

            self.enemy_distance_width,
            self.enemy_distance_height,
        )

        return regions

    # =========================================================
    # 航速
    # =========================================================

    def sample_speed_ring(
        self,
        frame_hsv: np.ndarray,
        cx: int,
        cy: int,
    ) -> Tuple[
        float,
        int,
        List[Tuple[int, int]],
    ]:

        height, width = (
            frame_hsv.shape[:2]
        )

        forward = 0
        reverse = 0

        sample_pts = []

        for (
            dx,
            dy,
            _angle,
            _radius,
        ) in self.ring_sample_offsets:

            px = cx + dx
            py = cy + dy

            sample_pts.append(
                (px, py)
            )

            if (
                px < 0
                or px >= width
                or py < 0
                or py >= height
            ):

                continue

            r = self.SAMPLE_PATCH_RADIUS

            x1 = max(
                0,
                px - r,
            )

            x2 = min(
                width,
                px + r + 1,
            )

            y1 = max(
                0,
                py - r,
            )

            y2 = min(
                height,
                py + r + 1,
            )

            patch = frame_hsv[
                y1:y2,
                x1:x2,
            ]

            if patch.size == 0:
                continue

            green_mask = cv2.inRange(
                patch,
                self.green_hsv_min,
                self.green_hsv_max,
            )

            red1 = cv2.inRange(
                patch,
                self.red_hsv_min1,
                self.red_hsv_max1,
            )

            red2 = cv2.inRange(
                patch,
                self.red_hsv_min2,
                self.red_hsv_max2,
            )

            red_mask = cv2.bitwise_or(
                red1,
                red2,
            )

            total = (
                patch.shape[0]
                * patch.shape[1]
            )

            green_ratio = (
                cv2.countNonZero(
                    green_mask
                )
                / total
            )

            red_ratio = (
                cv2.countNonZero(
                    red_mask
                )
                / total
            )

            if (
                green_ratio
                >= self.SAMPLE_GREEN_THRESHOLD
            ):

                forward += 1

            elif (
                red_ratio
                >= self.SAMPLE_RED_THRESHOLD
            ):

                reverse += 1

        if forward > 0:

            speed = forward / 8.0
            direction = 1

        elif reverse > 0:

            speed = reverse / 8.0
            direction = -1

        else:

            speed = 0.0
            direction = 0

        return (
            speed,
            direction,
            sample_pts,
        )

    # =========================================================
    # 标准模块入口
    # =========================================================

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_hsv: np.ndarray,
        previous_anchor: Optional[Anchor],
        screen_cx: int,
        screen_cy: int,
    ) -> Dict:

        anchor = (
            self.find_locked_anchor(
                frame_hsv,
                previous_anchor,
            )
        )

        if anchor is None:

            return {

                "anchor": None,

                "locked": False,

                "speed_fraction": 0.0,

                "direction": 0,

                "sample_pts": [],
            }

        (
            speed_fraction,
            direction,
            sample_pts,
        ) = self.sample_speed_ring(
            frame_hsv,
            anchor[0],
            anchor[1],
        )

        return {

            "anchor": anchor,

            "locked": True,

            "speed_fraction":
                speed_fraction,

            "direction":
                direction,

            "sample_pts":
                sample_pts,
        }