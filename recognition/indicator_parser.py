# recognition/indicator_parser.py

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Anchor = Tuple[
    int,  # center x
    int,  # center y
    int,  # hp left
    int,  # hp top
    int,  # hp width
    int,  # hp height
]


class IndicatorParser:
    """
    游戏 HUD 几何/颜色解析器。

    负责：
        1. 动态血条追踪
        2. 计算目标圆心
        3. 航速圆环 8 点
        4. 动态敌舰距离 ROI
        5. 敌我角度 ROI
        6. 提供调试框

    不负责 OCR。
    """

    def __init__(self):

        # =====================================================
        # 血条颜色
        # =====================================================

        self.red_lower1 = np.array(
            [0, 140, 140],
            dtype=np.uint8,
        )

        self.red_upper1 = np.array(
            [10, 255, 255],
            dtype=np.uint8,
        )

        self.red_lower2 = np.array(
            [170, 140, 140],
            dtype=np.uint8,
        )

        self.red_upper2 = np.array(
            [180, 255, 255],
            dtype=np.uint8,
        )

        # =====================================================
        # 状态灯
        # =====================================================

        self.green_hsv_min = np.array(
            [35, 100, 100],
            dtype=np.uint8,
        )

        self.green_hsv_max = np.array(
            [85, 255, 255],
            dtype=np.uint8,
        )

        self.red_hsv_min1 = np.array(
            [0, 100, 100],
            dtype=np.uint8,
        )

        self.red_hsv_max1 = np.array(
            [10, 255, 255],
            dtype=np.uint8,
        )

        self.red_hsv_min2 = np.array(
            [170, 100, 100],
            dtype=np.uint8,
        )

        self.red_hsv_max2 = np.array(
            [180, 255, 255],
            dtype=np.uint8,
        )

        self.grey_hsv_min = np.array(
            [0, 0, 60],
            dtype=np.uint8,
        )

        self.grey_hsv_max = np.array(
            [180, 45, 185],
            dtype=np.uint8,
        )

        # =====================================================
        # 血条 → 圆心
        # =====================================================

        self.OFFSET_X = 46
        self.OFFSET_Y = 21

        # =====================================================
        # 航速圆环
        # =====================================================

        self.RING_RADIUS = 37
        self.LIGHT_OFFSET = 8

        # =====================================================
        # 动态跟踪范围
        # =====================================================

        self.TRACK_SEARCH_RADIUS_X = 140
        self.TRACK_SEARCH_RADIUS_Y = 100

        # =====================================================
        # 个航速点 (angle, radius)
        # =====================================================

        self.RING_POINTS = [

            (292.5, 36.0),  # 1
            (340.0, 37.0),  # 2
            (20.0, 37.0),   # 3
            (50.0, 37.0),   # 4
            (130.0, 37.0),  # 5
            (170.0, 37.0),  # 6
            (190.0, 37.0),  # 7
            (247.5, 36.0),  # 8
        ]

        self.SAMPLE_PATCH_RADIUS = 1

        self.SAMPLE_GREEN_THRESHOLD = 0.35
        self.SAMPLE_RED_THRESHOLD = 0.35

        # =====================================================
        # 敌舰距离
        #
        # 相对于：
        #     敌舰血条左上角
        # =====================================================

        self.enemy_distance_offset_x = 0
        self.enemy_distance_offset_y = 33

        self.enemy_distance_width = 95
        self.enemy_distance_height = 21

        # =====================================================
        # 敌方相对角度
        # =====================================================

        self.enemy_angle_offset_x = -17
        self.enemy_angle_offset_y = 160

        self.enemy_angle_width = 36
        self.enemy_angle_height = 36

        # =====================================================
        # 我方相对角度
        # =====================================================

        self.our_angle_offset_x = -17
        self.our_angle_offset_y = 294

        self.our_angle_width = 36
        self.our_angle_height = 36

        # =====================================================
        # 角度圆环裁切
        #
        # 36 × 36
        #     ↓
        # 20 × 20
        #
        # ⭐ 从 6px 调大到 8px：
        #     先物理裁掉更多圆环像素，
        #     配合 OCR 端的开运算
        #     （去细线）形成双重保险。
        #
        #     如果实测发现数字被裁掉了，
        #     适当调小；如果圆环残留还在，
        #     可以继续调大。
        # =====================================================

        self.angle_crop_left = 8
        self.angle_crop_top = 8
        self.angle_crop_right = 8
        self.angle_crop_bottom = 8

        # =====================================================
        # 预计算航速点
        # =====================================================

        self.ring_sample_offsets = []

        for angle_deg, radius in self.RING_POINTS:

            angle_rad = np.deg2rad(angle_deg)

            dx = int(
                round(
                    radius * np.cos(angle_rad)
                )
            )

            dy = int(
                round(
                    radius * np.sin(angle_rad)
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

        if np.std(patch[:, :, 2]) > 35.0:
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
            cv2.countNonZero(green_mask)
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
            cv2.countNonZero(red)
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
            cv2.countNonZero(grey)
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

        height, width = frame_hsv.shape[:2]

        left = max(0, int(left))
        top = max(0, int(top))
        right = min(width, int(right))
        bottom = min(height, int(bottom))

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

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 15:
                continue

            if h < 2 or h > 8:
                continue

            hx_left = left + x
            hy_top = top + y

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

        height, width = frame_hsv.shape[:2]

        # 局部搜索
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

            anchor = self._search_anchor_in_region(
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

            if anchor is not None:
                return anchor

        # 全屏 fallback
        return self._search_anchor_in_region(
            frame_hsv,
            0,
            0,
            width,
            height,
        )

    # =========================================================
    # OCR 动态区域
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

        # 没锁定没有敌舰距离
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
    # 航速圆环
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

        height, width = frame_hsv.shape[:2]

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

            x1 = max(0, px - r)
            x2 = min(width, px + r + 1)

            y1 = max(0, py - r)
            y2 = min(height, py + r + 1)

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