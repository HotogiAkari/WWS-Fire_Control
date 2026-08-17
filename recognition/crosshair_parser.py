# recognition/crosshair_parser.py

import cv2
import numpy as np
from typing import Dict


class CrosshairParser:
    """
    战舰世界 瞄准镜刻度解析器。
    采用“精准ROI + 垂直形态学 + 中位数滤波”的逻辑。
    所有参数均由 vision_manager.py 统一传入。
    """

    def __init__(
        self,
        roi: tuple,
        white_thresh: int,
        vertical_kernel_h: int,
        tick_max_width: int,
        min_spacing: int,
        max_spacing: int,
    ):
        self.roi_offset_x = int(roi[0])
        self.roi_offset_y = int(roi[1])
        self.roi_w = int(roi[2])
        self.roi_h = int(roi[3])
        
        self.white_thresh = int(white_thresh)
        self.vertical_kernel_h = int(vertical_kernel_h)
        self.tick_max_width = int(tick_max_width)
        
        self.min_spacing = int(min_spacing)
        self.max_spacing = int(max_spacing)
        
        # 垂直开运算内核 (宽1, 高N)
        self.vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (1, self.vertical_kernel_h)
        )

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_hsv: np.ndarray,
        context: Dict,
    ) -> Dict:
        
        screen = context.get("screen", {})
        cx = screen.get("cx")
        cy = screen.get("cy")
        
        if cx is None or cy is None:
            return self._empty_result()

        height, width = frame_bgr.shape[:2]

        # 1. 计算 ROI 绝对坐标
        x1 = cx + self.roi_offset_x
        y1 = cy + self.roi_offset_y
        x2 = x1 + self.roi_w
        y2 = y1 + self.roi_h

        # 越界保护
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            return self._empty_result()

        # 2. 截取 ROI 并二值化 (刻度是极亮的白色)
        roi = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)

        # 3. 垂直形态学开运算 (利用刻度是竖线的特征，直接杀掉横线和噪点)
        verticals = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.vertical_kernel)

        # 4. 提取竖线坐标
        contours, _ = cv2.findContours(verticals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tick_xs = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 过滤过粗的异常块
            if w <= self.tick_max_width and h >= self.vertical_kernel_h:
                # 记录绝对屏幕 X 坐标
                tick_xs.append(x1 + x + (w // 2))

        tick_xs.sort()

        # 5. 计算间距
        if len(tick_xs) < 2:
            return self._empty_result()

        diffs = np.diff(tick_xs)
        valid_diffs = [d for d in diffs if self.min_spacing <= d <= self.max_spacing]

        if not valid_diffs:
            return self._empty_result()

        # 采用中位数，无视任何因背景导致的多余检测或漏检
        spacing = float(np.median(valid_diffs))

        # 6. 利用对称性，从中心向两侧映射推导完美的绝对坐标
        # 生成左右各 15 个点 (覆盖绝大部分宽屏)
        all_ticks = []
        for i in range(1, 5):
            all_ticks.append((int(round(cx - i * spacing)), cy))
            all_ticks.append((int(round(cx + i * spacing)), cy))
            
        # 按 X 轴排序
        all_ticks.sort(key=lambda pt: pt[0])

        return {
            "valid": True,
            "spacing": spacing,
            "all_ticks": all_ticks,
            "roi_rect": (x1, y1, self.roi_w, self.roi_h), # 将实际ROI传出供Overlay绘画
        }

    def _empty_result(self) -> Dict:
        return {
            "valid": False,
            "spacing": 0.0,
            "all_ticks": [],
            "roi_rect": None,
        }

    def close(self):
        pass