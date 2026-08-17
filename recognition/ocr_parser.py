import concurrent.futures
import ctypes
import os
import platform
import re
import site
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# OCR 调参区
#
# 所有 OCR 图像处理参数集中在这里。
#
# VisionManager 可以通过 OCR_CONFIG 覆盖这些默认参数。
# ============================================================


# ============================================================
# 1. 字体放大
# ============================================================


# ------------------------------------------------------------
# 数字字段放大倍率
#
# 调大：
#     小数字更容易识别
#     OCR 输入尺寸增加
#     CPU 开销增加
#
# 调小：
#     更快
#     但小数字可能更难识别
# ------------------------------------------------------------

NUMERIC_SCALE = 2.0


# ------------------------------------------------------------
# 普通文字放大倍率
#
# 舰名等字段使用。
# ------------------------------------------------------------

NORMAL_SCALE_SMALL = 2.0
NORMAL_SCALE_LARGE = 1.5

# ROI 原始高度达到该值后使用 LARGE。
NORMAL_TEXT_HEIGHT_THRESHOLD = 30


# ============================================================
# 2. HSV 彩色抑制
#
# 这里只作为辅助。
#
# 白色 / 灰色文字：
#     饱和度通常较低
#
# 黄色 / 蓝色 / 红色背景：
#     饱和度通常较高
#
# 不要设置过低，否则可能损失浅黄色 / 灰白色文字。
# ============================================================

BINARY_MAX_SATURATION = 100


# ============================================================
# 3. 局部背景估计
#
# 用 Gaussian Blur 估计局部背景。
#
# 背景通常变化较慢，
# 文字是不透明的局部高亮结构。
# ============================================================

BACKGROUND_BLUR_KERNEL = (15, 15)


# ============================================================
# 4. Top-Hat
#
# Top-Hat =
#
#     原图 - Opening(原图)
#
# 用于提取局部亮结构。
#
# 建议尝试：
#
#     11 × 11
#     15 × 15
#     21 × 21
# ============================================================

TOPHAT_KERNEL = (15, 15)


# ============================================================
# 5. Top-Hat 增益
#
# 调大：
#     弱文字更加明显
#
# 调太大：
#     背景噪声也会被增强
# ============================================================

TOPHAT_GAIN = 4.0


# ============================================================
# 6. Top-Hat 后平滑
#
# 用于去除单像素噪声。
# ============================================================

TOPHAT_POST_BLUR_KERNEL = (3, 3)


# ============================================================
# 7. Adaptive Threshold
#
# block size 必须是奇数。
#
# 调大：
#     对大范围亮度变化更稳定
#
# 调小：
#     对细节更敏感
#
# C 调大：
#     更严格
#
# C 调小：
#     更容易保留弱文字
# ============================================================

ADAPTIVE_BLOCK_SIZE = 11

ADAPTIVE_C = -2


# ============================================================
# 8. 二值图去噪
#
# Opening：
#
#     去除孤立小噪点。
#
# Closing：
#
#     修复文字细小断裂。
#
# 不建议使用大核，
# 防止数字笔画变粗或字符粘连。
# ============================================================


# ------------------------------------------------------------
# Opening
# ------------------------------------------------------------

NOISE_OPEN_KERNEL = (2, 2)

NOISE_OPEN_ITERATIONS = 1


# ------------------------------------------------------------
# Closing
# ------------------------------------------------------------

POST_MORPH_KERNEL = (2, 2)

POST_MORPH_CLOSE_ITERATIONS = 1


# ============================================================
# 9. 小连通区域过滤
#
# 二值化后把非常小的前景区域删除。
#
# 例如：
#
#     零散 1~2 像素噪点
#
# 但不要设置太大，
# 否则小数字笔画会被一起删掉。
# ============================================================

MIN_FOREGROUND_COMPONENT_AREA = 3


# ============================================================
# 10. 相对角度圆环 Mask
#
# ⭐⭐⭐ 这里是你之后主要调节的位置。
#
# 角度 ROI 默认：
#
#     36 × 36
#
# 圆环中心默认位于 ROI 中心附近。
#
# 如果圆环还残留：
#
#     增大 ANGLE_RING_RADIUS
#
# 如果数字被圆环 Mask 误删：
#
#     减小 ANGLE_RING_RADIUS
#
# ------------------------------------------------------------
#
# ANGLE_RING_CENTER_X / Y
#
#     圆环中心相对于角度 ROI 左上角的位置。
#
# 当前默认：
#
#     18, 18
#
# ------------------------------------------------------------
#
# ANGLE_RING_RADIUS
#
#     ⭐ 圆环半径
#
# ------------------------------------------------------------
#
# ANGLE_RING_THICKNESS
#
#     圆环需要清除的厚度。
#
# 调大：
#     可以更彻底地删除圆环
#
# 调太大：
#     可能碰到数字
# ============================================================

ANGLE_RING_CENTER_X = 18.0

ANGLE_RING_CENTER_Y = 18.0


# ============================================================
# ⭐ 圆环半径
#
# 这是你以后最主要调整的参数。
# ============================================================

ANGLE_RING_RADIUS = 13.0


# ============================================================
# 圆环 Mask 厚度
# ============================================================

ANGLE_RING_THICKNESS = 3.0


# ============================================================
# 11. 圆环 Mask 边缘柔化
#
# 0：
#     不额外处理
#
# 1~2：
#     轻微扩大 Mask
#
# 如果圆环还有极少残留，
# 可以从 0 调到 1。
# ============================================================

ANGLE_RING_MASK_DILATE = 0


# ============================================================
# 12. OCR 输入边框
# ============================================================

BORDER_TOP = 8

BORDER_BOTTOM = 8

BORDER_LEFT = 12

BORDER_RIGHT = 12


# ============================================================
# 13. OCR Worker
# ============================================================

OCR_WORKER_COUNT = 1


# ============================================================
# 14. OCR 最小刷新间隔
# ============================================================

MIN_INTERVALS = {

    "flight_time": 0.040,

    "aim_distance": 0.040,

    "enemy_angle": 0.050,

    "our_angle": 0.050,

    "enemy_distance": 1.0,

    "ship_name": 0.0,

    "max_speed": 0.0,
}


# ============================================================
# 15. ROI 变化检测阈值
# ============================================================

CHANGE_THRESHOLDS = {

    "flight_time": 2.0,

    "aim_distance": 2.0,

    "enemy_angle": 1.5,

    "our_angle": 1.5,
}


# ============================================================
# 16. 静态字段刷新
# ============================================================

STATIC_REFRESH_INTERVAL = 2.0


# ============================================================
# 17. 数值范围保护
# ============================================================

NUMERIC_RANGES = {

    "flight_time": (
        0.0,
        99.99,
    ),

    "aim_distance": (
        0.0,
        999.99,
    ),

    "enemy_angle": (
        0.0,
        360.0,
    ),

    "our_angle": (
        0.0,
        360.0,
    ),

    "enemy_distance": (
        0.0,
        999.9,
    ),

    "max_speed": (
        0.0,
        999.9,
    ),
}


# ============================================================
# 18. Debug 图片
# ============================================================

DEBUG_FIELDS = (

    "flight_time",

    "aim_distance",

    "enemy_angle",

    "our_angle",

    "enemy_distance",

    "max_speed",

    "ship_name",
)


# ============================================================
# Windows / ONNX Runtime DLL
# ============================================================

def _prepare_windows_dll_environment():
    """
    在 RapidOCR / ONNX Runtime 导入前准备 Windows DLL 环境。
    """

    if platform.system() != "Windows":

        return

    dll_directories = set()

    # ========================================================
    # Python
    # ========================================================

    try:

        python_root = (
            Path(sys.executable)
            .resolve()
            .parent
        )

        dll_directories.add(
            python_root
        )

        python_dll_dir = (
            python_root
            / "DLLs"
        )

        if python_dll_dir.exists():

            dll_directories.add(
                python_dll_dir
            )

    except Exception:

        pass

    # ========================================================
    # site-packages
    # ========================================================

    try:

        site_packages = (
            site.getsitepackages()
        )

    except Exception:

        site_packages = []

    for path in site_packages:

        try:

            path_obj = Path(path)

            if path_obj.exists():

                dll_directories.add(
                    path_obj
                )

        except Exception:

            pass

    # ========================================================
    # ONNX Runtime
    # ========================================================

    for base in list(
        dll_directories
    ):

        ort_capi = (
            base
            / "onnxruntime"
            / "capi"
        )

        if ort_capi.exists():

            dll_directories.add(
                ort_capi
            )

    # ========================================================
    # Windows DLL search path
    # ========================================================

    add_dll_directory = getattr(
        os,
        "add_dll_directory",
        None,
    )

    if add_dll_directory is not None:

        for directory in (
            dll_directories
        ):

            try:

                if directory.exists():

                    add_dll_directory(
                        str(directory)
                    )

            except Exception:

                pass

    # ========================================================
    # MSVC runtime
    # ========================================================

    runtime_dlls = (

        "vcruntime140.dll",

        "msvcp140.dll",

        "vcruntime140_1.dll",
    )

    for dll_name in runtime_dlls:

        try:

            ctypes.CDLL(
                dll_name
            )

        except OSError:

            pass


_prepare_windows_dll_environment()


# ============================================================
# OCRParser
# ============================================================

class OCRParser:

    # ========================================================
    # Dynamic fields
    # ========================================================

    DYNAMIC_FIELDS = (

        "flight_time",

        "aim_distance",

        "enemy_angle",

        "our_angle",

        "enemy_distance",
    )

    # ========================================================
    # Static fields
    # ========================================================

    STATIC_FIELDS = (

        "ship_name",

        "max_speed",
    )

    def __init__(
        self,

        # =====================================================
        # 基础参数
        # =====================================================

        numeric_scale: float = (
            NUMERIC_SCALE
        ),

        normal_scale_small: float = (
            NORMAL_SCALE_SMALL
        ),

        normal_scale_large: float = (
            NORMAL_SCALE_LARGE
        ),

        normal_text_height_threshold: int = (
            NORMAL_TEXT_HEIGHT_THRESHOLD
        ),

        binary_max_saturation: int = (
            BINARY_MAX_SATURATION
        ),

        # =====================================================
        # Background
        # =====================================================

        background_blur_kernel=(
            BACKGROUND_BLUR_KERNEL
        ),

        # =====================================================
        # Top-Hat
        # =====================================================

        tophat_kernel=(
            TOPHAT_KERNEL
        ),

        tophat_gain: float = (
            TOPHAT_GAIN
        ),

        tophat_post_blur_kernel=(
            TOPHAT_POST_BLUR_KERNEL
        ),

        # =====================================================
        # Adaptive Threshold
        # =====================================================

        adaptive_block_size: int = (
            ADAPTIVE_BLOCK_SIZE
        ),

        adaptive_c: float = (
            ADAPTIVE_C
        ),

        # =====================================================
        # Noise Removal
        # =====================================================

        noise_open_kernel=(
            NOISE_OPEN_KERNEL
        ),

        noise_open_iterations: int = (
            NOISE_OPEN_ITERATIONS
        ),

        post_morph_kernel=(
            POST_MORPH_KERNEL
        ),

        post_morph_close_iterations: int = (
            POST_MORPH_CLOSE_ITERATIONS
        ),

        min_foreground_component_area: int = (
            MIN_FOREGROUND_COMPONENT_AREA
        ),

        # =====================================================
        # Angle Ring
        # =====================================================

        angle_ring_center_x: float = (
            ANGLE_RING_CENTER_X
        ),

        angle_ring_center_y: float = (
            ANGLE_RING_CENTER_Y
        ),

        angle_ring_radius: float = (
            ANGLE_RING_RADIUS
        ),

        angle_ring_thickness: float = (
            ANGLE_RING_THICKNESS
        ),

        angle_ring_mask_dilate: int = (
            ANGLE_RING_MASK_DILATE
        ),

        # =====================================================
        # Border
        # =====================================================

        border_top: int = (
            BORDER_TOP
        ),

        border_bottom: int = (
            BORDER_BOTTOM
        ),

        border_left: int = (
            BORDER_LEFT
        ),

        border_right: int = (
            BORDER_RIGHT
        ),
    ):

        # =====================================================
        # 保存参数
        # =====================================================

        self.numeric_scale = float(
            numeric_scale
        )

        self.normal_scale_small = float(
            normal_scale_small
        )

        self.normal_scale_large = float(
            normal_scale_large
        )

        self.normal_text_height_threshold = (
            int(
                normal_text_height_threshold
            )
        )

        self.binary_max_saturation = int(
            binary_max_saturation
        )

        # -----------------------------------------------------
        # Background
        # -----------------------------------------------------

        self.background_blur_kernel = (
            tuple(
                background_blur_kernel
            )
        )

        # -----------------------------------------------------
        # Top-Hat
        # -----------------------------------------------------

        self.tophat_kernel = (
            tuple(
                tophat_kernel
            )
        )

        self.tophat_gain = float(
            tophat_gain
        )

        self.tophat_post_blur_kernel = (
            tuple(
                tophat_post_blur_kernel
            )
        )

        # -----------------------------------------------------
        # Adaptive
        # -----------------------------------------------------

        self.adaptive_block_size = int(
            adaptive_block_size
        )

        self.adaptive_c = float(
            adaptive_c
        )

        # -----------------------------------------------------
        # Noise
        # -----------------------------------------------------

        self.noise_open_kernel = (
            tuple(
                noise_open_kernel
            )
        )

        self.noise_open_iterations = int(
            noise_open_iterations
        )

        self.post_morph_kernel = (
            tuple(
                post_morph_kernel
            )
        )

        self.post_morph_close_iterations = (
            int(
                post_morph_close_iterations
            )
        )

        self.min_foreground_component_area = (
            int(
                min_foreground_component_area
            )
        )

        # -----------------------------------------------------
        # Angle Ring
        # -----------------------------------------------------

        self.angle_ring_center_x = float(
            angle_ring_center_x
        )

        self.angle_ring_center_y = float(
            angle_ring_center_y
        )

        self.angle_ring_radius = float(
            angle_ring_radius
        )

        self.angle_ring_thickness = float(
            angle_ring_thickness
        )

        self.angle_ring_mask_dilate = int(
            angle_ring_mask_dilate
        )

        # -----------------------------------------------------
        # Border
        # -----------------------------------------------------

        self.border_top = int(
            border_top
        )

        self.border_bottom = int(
            border_bottom
        )

        self.border_left = int(
            border_left
        )

        self.border_right = int(
            border_right
        )

        # =====================================================
        # RapidOCR
        # =====================================================

        try:

            from rapidocr import (
                EngineType,
                LangRec,
                ModelType,
                OCRVersion,
                RapidOCR,
            )

        except ImportError as exc:

            raise RuntimeError(
                "\n"
                "RapidOCR / ONNX Runtime 导入失败。\n"
            ) from exc

        self.EngineType = EngineType
        self.LangRec = LangRec
        self.ModelType = ModelType
        self.OCRVersion = OCRVersion
        self.RapidOCR = RapidOCR

        print(
            "[OCR] 正在加载 RapidOCR..."
        )

        params = {

            "Global.use_det":
                False,

            "Global.use_cls":
                False,

            "Global.use_rec":
                True,

            "Global.use_preprocess_img":
                False,

            "Global.log_level":
                "critical",

            # -------------------------------------------------
            # Detection
            # -------------------------------------------------

            "Det.engine_type":
                EngineType.ONNXRUNTIME,

            "Det.model_type":
                ModelType.MOBILE,

            "Det.ocr_version":
                OCRVersion.PPOCRV5,

            # -------------------------------------------------
            # Recognition
            # -------------------------------------------------

            "Rec.engine_type":
                EngineType.ONNXRUNTIME,

            "Rec.lang_type":
                LangRec.CH,

            "Rec.model_type":
                ModelType.MOBILE,

            "Rec.ocr_version":
                OCRVersion.PPOCRV5,

            # -------------------------------------------------
            # ONNX Runtime
            # -------------------------------------------------

            "EngineConfig.onnxruntime.intra_op_num_threads":
                2,

            "EngineConfig.onnxruntime.inter_op_num_threads":
                1,
        }

        try:

            self.engine = self.RapidOCR(
                params=params
            )

        except Exception as exc:

            raise RuntimeError(
                "\n"
                "[OCR] RapidOCR 初始化失败。\n"
                "\n"
                f"原始错误：\n{exc}\n"
            ) from exc

        print(
            "[OCR] RapidOCR 加载完成。"
        )

        # =====================================================
        # Worker
        # =====================================================

        self.executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=OCR_WORKER_COUNT,
                thread_name_prefix="OCRWorker",
            )
        )

        self.future = None

        self.current_field: Optional[str] = None

        # =====================================================
        # Cache
        # =====================================================

        self.cache = {

            "flight_time": 0.0,

            "aim_distance": 0.0,

            "enemy_angle": 0.0,

            "our_angle": 0.0,

            "enemy_distance": 0.0,

            "ship_name": "Unknown",

            "max_speed": 0.0,
        }

        # =====================================================
        # ROI Probe
        # =====================================================

        self.last_probe = {}

        # =====================================================
        # OCR 时间
        # =====================================================

        self.last_submit_time = {}

        self.min_interval = dict(
            MIN_INTERVALS
        )

        self.change_threshold = dict(
            CHANGE_THRESHOLDS
        )

        # =====================================================
        # Static
        # =====================================================

        self.static_queue = []

        self.static_refresh_needed = True

        self.last_static_refresh = 0.0

        # =====================================================
        # Round Robin
        # =====================================================

        self.round_robin_index = 0

        # =====================================================
        # Lock
        # =====================================================

        self.was_locked = False

        # =====================================================
        # Debug
        # =====================================================

        self.last_ocr_ms = 0.0

        self.last_ocr_field = ""

        self.last_ocr_text = ""

        self.last_ocr_score = 0.0

        self.error_count = 0

        # =====================================================
        # Debug Images
        # =====================================================

        self._debug_image_lock = (
            threading.Lock()
        )

        self._debug_processed_images = {

            field: None

            for field in DEBUG_FIELDS
        }

    # =========================================================
    # RapidOCR
    # =========================================================

    def _run_recognition(
        self,
        image: np.ndarray,
    ):

        return self.engine(
            image,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )

    # =========================================================
    # OCR Result
    # =========================================================

    @staticmethod
    def _extract_result(
        result,
    ) -> Tuple[str, float]:

        if result is None:

            return (
                "",
                0.0,
            )

        txts = getattr(
            result,
            "txts",
            None,
        )

        scores = getattr(
            result,
            "scores",
            None,
        )

        if txts is None:

            return (
                "",
                0.0,
            )

        if isinstance(
            txts,
            str,
        ):

            text = txts.strip()

        else:

            text = "".join(
                str(x)
                for x in txts
            ).strip()

        score = 0.0

        if scores is not None:

            try:

                values = [
                    float(x)
                    for x in scores
                ]

                if values:

                    score = (
                        sum(values)
                        / len(values)
                    )

            except Exception:

                pass

        return (
            text,
            score,
        )

    # =========================================================
    # Resize
    # =========================================================

    @staticmethod
    def _resize_roi(
        roi: np.ndarray,
        scale: float,
    ) -> np.ndarray:

        if scale == 1.0:

            return np.ascontiguousarray(
                roi
            )

        return cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # =========================================================
    # Local Background
    # =========================================================

    def _normalize_local_background(
        self,
        value: np.ndarray,
    ) -> np.ndarray:

        background = cv2.GaussianBlur(
            value,
            self.background_blur_kernel,
            0,
        )

        value_i = (
            value.astype(
                np.int16
            )
        )

        background_i = (
            background.astype(
                np.int16
            )
        )

        normalized = (
            value_i
            - background_i
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        ).astype(
            np.uint8
        )

        return normalized

    # =========================================================
    # Top-Hat
    # =========================================================

    def _tophat(
        self,
        value: np.ndarray,
    ) -> np.ndarray:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.tophat_kernel,
        )

        tophat = cv2.morphologyEx(
            value,
            cv2.MORPH_TOPHAT,
            kernel,
        )

        if self.tophat_gain != 1.0:

            enhanced = cv2.convertScaleAbs(
                tophat,
                alpha=self.tophat_gain,
                beta=0,
            )

        else:

            enhanced = tophat

        if (
            self.tophat_post_blur_kernel
            != (1, 1)
        ):

            enhanced = cv2.GaussianBlur(
                enhanced,
                self.tophat_post_blur_kernel,
                0,
            )

        return enhanced

    # =========================================================
    # 自适应阈值
    # =========================================================

    def _adaptive_threshold(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        block_size = (
            self.adaptive_block_size
        )

        if block_size < 3:

            block_size = 3

        if block_size % 2 == 0:

            block_size += 1

        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.adaptive_c,
        )

    # =========================================================
    # Opening 去噪
    # =========================================================

    def _remove_small_noise(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self.noise_open_kernel,
        )

        cleaned = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=(
                self.noise_open_iterations
            ),
        )

        return cleaned

    # =========================================================
    # 小连通区域过滤
    # =========================================================

    def _remove_small_components(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:

        # =====================================================
        # 前景：
        #
        #     白色
        #
        # 此时 binary 还是：
        #
        #     白字黑底
        # =====================================================

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        cleaned = np.zeros_like(
            binary
        )

        for label in range(
            1,
            num_labels,
        ):

            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            if (
                area
                < self.min_foreground_component_area
            ):

                continue

            cleaned[
                labels == label
            ] = 255

        return cleaned

    # =========================================================
    # Closing
    # =========================================================

    def _repair_text(
        self,
        binary: np.ndarray,
        normal_text: bool,
    ) -> np.ndarray:

        if not normal_text:

            return binary

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.post_morph_kernel,
        )

        return cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=(
                self.post_morph_close_iterations
            ),
        )

    # =========================================================
    # 圆环 Mask
    # =========================================================

    def _create_angle_ring_mask(
        self,
        height: int,
        width: int,
    ) -> np.ndarray:

        """
        创建固定几何圆环 Mask。

        黑色：
            要删除的圆环区域。

        白色：
            保留区域。

        圆心、半径、厚度全部来自顶部调参区。
        """

        mask = np.ones(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        ) * 255

        center = (
            int(
                round(
                    self.angle_ring_center_x
                )
            ),
            int(
                round(
                    self.angle_ring_center_y
                )
            ),
        )

        outer_radius = (
            self.angle_ring_radius
            + self.angle_ring_thickness / 2.0
        )

        inner_radius = (
            max(
                0.0,
                self.angle_ring_radius
                - self.angle_ring_thickness / 2.0,
            )
        )

        # -----------------------------------------------------
        # 先画外圆
        # -----------------------------------------------------

        cv2.circle(
            mask,
            center,
            int(
                round(
                    outer_radius
                )
            ),
            0,
            thickness=-1,
        )

        # -----------------------------------------------------
        # 再把内部恢复为白色
        #
        # 最终只剩下一个圆环形黑色区域。
        # -----------------------------------------------------

        if inner_radius > 0:

            cv2.circle(
                mask,
                center,
                int(
                    round(
                        inner_radius
                    )
                ),
                255,
                thickness=-1,
            )

        # -----------------------------------------------------
        # 可选扩大 Mask
        # -----------------------------------------------------

        if (
            self.angle_ring_mask_dilate
            > 0
        ):

            kernel_size = (
                self.angle_ring_mask_dilate
                * 2
                + 1
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    kernel_size,
                    kernel_size,
                ),
            )

            inverse = cv2.bitwise_not(
                mask
            )

            inverse = cv2.dilate(
                inverse,
                kernel,
                iterations=1,
            )

            mask = cv2.bitwise_not(
                inverse
            )

        return mask

    # =========================================================
    # 应用角度圆环 Mask
    # =========================================================

    def _remove_angle_ring(
        self,
        binary_black_text: np.ndarray,
    ) -> np.ndarray:

        height, width = (
            binary_black_text.shape[:2]
        )

        ring_mask = (
            self._create_angle_ring_mask(
                height,
                width,
            )
        )

        # -----------------------------------------------------
        # ring_mask：
        #
        #     255 = 保留
        #     0   = 删除
        # -----------------------------------------------------

        result = cv2.bitwise_and(
            binary_black_text,
            ring_mask,
        )

        # -----------------------------------------------------
        # 被删除的圆环区域必须变成白色。
        #
        # 因为现在是：
        #
        #     黑字白底
        # -----------------------------------------------------

        removed = (
            ring_mask == 0
        )

        result[
            removed
        ] = 255

        return result

    # =========================================================
    # Border
    # =========================================================

    def _add_border(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        return cv2.copyMakeBorder(
            image,

            self.border_top,
            self.border_bottom,

            self.border_left,
            self.border_right,

            cv2.BORDER_CONSTANT,

            value=255,
        )

    # =========================================================
    # Gray → BGR
    # =========================================================

    @staticmethod
    def _gray_to_bgr(
        image: np.ndarray,
    ) -> np.ndarray:

        if image.ndim == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        return np.ascontiguousarray(
            image
        )

    # =========================================================
    # 通用 OCR preprocessing
    # =========================================================

    def _preprocess_binary(
        self,
        roi: np.ndarray,
        scale: float,
        normal_text: bool,
        remove_angle_ring: bool = False,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        # =====================================================
        # 1. Resize
        # =====================================================

        image = self._resize_roi(
            roi,
            scale,
        )

        # =====================================================
        # 2. HSV
        # =====================================================

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        saturation = (
            hsv[:, :, 1]
        )

        value = (
            hsv[:, :, 2]
        )

        # =====================================================
        # 3. Local Background
        # =====================================================

        normalized = (
            self._normalize_local_background(
                value
            )
        )

        # =====================================================
        # 4. Top-Hat
        # =====================================================

        tophat = (
            self._tophat(
                value
            )
        )

        # =====================================================
        # 5. 融合
        #
        # 两个增强结果取最大值。
        # =====================================================

        enhanced = np.maximum(
            normalized,
            tophat,
        )

        # =====================================================
        # 6. 彩色背景辅助抑制
        #
        # 不直接删除，
        # 只把明显高饱和区域的增强强度降低。
        # =====================================================

        low_saturation = (
            saturation
            <= self.binary_max_saturation
        )

        enhanced = enhanced.copy()

        enhanced[
            ~low_saturation
        ] = (
            enhanced[
                ~low_saturation
            ]
            // 3
        )

        # =====================================================
        # 7. Adaptive Threshold
        #
        # 输出：
        #
        #     白字黑底
        # =====================================================

        binary = (
            self._adaptive_threshold(
                enhanced
            )
        )

        # =====================================================
        # 8. Opening
        #
        # ⭐ 新增：
        #
        # 去除零散小噪点。
        # =====================================================

        binary = (
            self._remove_small_noise(
                binary
            )
        )

        # =====================================================
        # 9. 连通组件过滤
        #
        # 删除非常小的孤立前景区域。
        # =====================================================

        binary = (
            self._remove_small_components(
                binary
            )
        )

        # =====================================================
        # 10. 普通文字修复
        #
        # 数字默认不做 Closing，
        # 避免数字粘连。
        # =====================================================

        binary = (
            self._repair_text(
                binary,
                normal_text,
            )
        )

        # =====================================================
        # 11. 反色
        #
        # 当前：
        #
        #     白字黑底
        #
        # 变成：
        #
        #     黑字白底
        # =====================================================

        binary = cv2.bitwise_not(
            binary
        )

        # =====================================================
        # 12. 相对角度专用圆环 Mask
        #
        # 这一步必须发生在反色以后。
        #
        # 因为最终要保证：
        #
        #     圆环区域 = 白色
        # =====================================================

        if remove_angle_ring:

            binary = (
                self._remove_angle_ring(
                    binary
                )
            )

        # =====================================================
        # 13. Border
        # =====================================================

        binary = self._add_border(
            binary
        )

        # =====================================================
        # 14. BGR
        # =====================================================

        binary = self._gray_to_bgr(
            binary
        )

        return binary

    # =========================================================
    # Numeric
    # =========================================================

    def _preprocess_numeric(
        self,
        roi: np.ndarray,
        remove_angle_ring: bool = False,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        return self._preprocess_binary(

            roi,

            scale=self.numeric_scale,

            normal_text=False,

            remove_angle_ring=(
                remove_angle_ring
            ),
        )

    # =========================================================
    # Normal text
    # =========================================================

    def _preprocess_normal(
        self,
        roi: np.ndarray,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        h = roi.shape[0]

        if (
            h
            >= self.normal_text_height_threshold
        ):

            scale = (
                self.normal_scale_large
            )

        else:

            scale = (
                self.normal_scale_small
            )

        return self._preprocess_binary(

            roi,

            scale=scale,

            normal_text=True,

            remove_angle_ring=False,
        )

    # =========================================================
    # 根据字段选择 preprocessing
    # =========================================================

    def _prepare_roi(
        self,
        field: str,
        roi: np.ndarray,
        geometry: Optional[dict],
    ) -> np.ndarray:

        # =====================================================
        # 相对角度
        # =====================================================

        if field in (
            "enemy_angle",
            "our_angle",
        ):

            return self._preprocess_numeric(
                roi,
                remove_angle_ring=True,
            )

        # =====================================================
        # 普通数字
        # =====================================================

        if field in (
            "flight_time",
            "aim_distance",
            "enemy_distance",
            "max_speed",
        ):

            return self._preprocess_numeric(
                roi,
                remove_angle_ring=False,
            )

        # =====================================================
        # 舰名
        # =====================================================

        return self._preprocess_normal(
            roi
        )

    # =========================================================
    # Debug Image
    # =========================================================

    def _set_debug_image(
        self,
        field: str,
        image: np.ndarray,
    ):

        if (
            field not in DEBUG_FIELDS
            or image is None
            or image.size == 0
        ):

            return

        image_copy = (
            np.ascontiguousarray(
                image.copy()
            )
        )

        with self._debug_image_lock:

            self._debug_processed_images[
                field
            ] = image_copy

    def get_debug_images(
        self,
    ) -> Dict[
        str,
        Optional[np.ndarray],
    ]:

        with self._debug_image_lock:

            result = {}

            for (
                field,
                image,
            ) in (
                self
                ._debug_processed_images
                .items()
            ):

                if image is None:

                    result[field] = None

                else:

                    result[field] = (
                        image.copy()
                    )

            return result

    # =========================================================
    # OCR 单字段
    # =========================================================

    def _recognize(
        self,
        field: str,
        roi: np.ndarray,
        geometry: Optional[dict],
    ):

        start = (
            time.perf_counter()
        )

        # =====================================================
        # Processing
        # =====================================================

        image = (
            self._prepare_roi(
                field,
                roi,
                geometry,
            )
        )

        # =====================================================
        # Debug：
        #
        # 保存真正送给 OCR 的最终图片。
        # =====================================================

        self._set_debug_image(
            field,
            image,
        )

        # =====================================================
        # OCR
        # =====================================================

        result = (
            self._run_recognition(
                image
            )
        )

        text, score = (
            self._extract_result(
                result
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            field,
            text,
            score,
            elapsed,
        )

    # =========================================================
    # ROI 变化
    # =========================================================

    def _has_changed(
        self,
        field: str,
        roi: np.ndarray,
    ) -> bool:

        if (
            roi is None
            or roi.size == 0
        ):

            return False

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        old = (
            self.last_probe.get(
                field
            )
        )

        self.last_probe[field] = (
            gray
        )

        if old is None:

            return True

        if old.shape != gray.shape:

            return True

        difference = cv2.absdiff(
            gray,
            old,
        )

        return (
            float(
                difference.mean()
            )
            >= self.change_threshold[
                field
            ]
        )

    # =========================================================
    # Static Refresh
    # =========================================================

    def force_static_refresh(
        self,
    ):

        self.static_refresh_needed = True

        self.static_queue = [

            "ship_name",

            "max_speed",
        ]

        self.last_static_refresh = 0.0

    # =========================================================
    # Lock
    # =========================================================

    def set_locked(
        self,
        locked: bool,
    ):

        if (
            locked
            and not self.was_locked
        ):

            self.force_static_refresh()

        elif (
            not locked
            and self.was_locked
        ):

            self.force_static_refresh()

        self.was_locked = locked

    # =========================================================
    # Numeric Normalize
    # =========================================================

    @staticmethod
    def _normalize_numeric_text(
        text: str,
    ) -> str:

        return (
            text
            .replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("l", "1")
            .replace("|", "1")
        )

    # =========================================================
    # Extract Digits
    # =========================================================

    @classmethod
    def _extract_digits(
        cls,
        text: str,
    ) -> str:

        text = (
            cls._normalize_numeric_text(
                text
            )
        )

        return re.sub(
            r"\D",
            "",
            text,
        )

    # =========================================================
    # Numeric Range
    # =========================================================

    @staticmethod
    def _validate_numeric_value(
        field: str,
        value: float,
    ) -> bool:

        if field not in NUMERIC_RANGES:

            return True

        minimum, maximum = (
            NUMERIC_RANGES[field]
        )

        return (
            minimum
            <= value
            <= maximum
        )

    # =========================================================
    # OCR Poll
    # =========================================================

    def poll(self):

        if self.future is None:

            return

        if not self.future.done():

            return

        field = self.current_field

        try:

            (
                field,
                text,
                score,
                cost_ms,
            ) = self.future.result()

            # =================================================
            # Debug
            # =================================================

            self.last_ocr_ms = (
                cost_ms
            )

            self.last_ocr_field = (
                field
            )

            self.last_ocr_text = (
                text
            )

            self.last_ocr_score = (
                score
            )

            # =================================================
            # Flight Time
            # =================================================

            if field == "flight_time":

                digits = (
                    self._extract_digits(
                        text
                    )
                )

                if digits:

                    value = (
                        int(digits)
                        / 100.0
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[
                            field
                        ] = round(
                            value,
                            2,
                        )

            # =================================================
            # Aim Distance
            # =================================================

            elif field == "aim_distance":

                digits = (
                    self._extract_digits(
                        text
                    )
                )

                if digits:

                    value = (
                        int(digits)
                        / 100.0
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[
                            field
                        ] = round(
                            value,
                            2,
                        )

            # =================================================
            # Enemy Angle
            # =================================================

            elif field == "enemy_angle":

                digits = (
                    self._extract_digits(
                        text
                    )
                )

                if digits:

                    value = float(
                        int(digits)
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[
                            field
                        ] = value

            # =================================================
            # Our Angle
            # =================================================

            elif field == "our_angle":

                digits = (
                    self._extract_digits(
                        text
                    )
                )

                if digits:

                    value = float(
                        int(digits)
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[
                            field
                        ] = value

            # =================================================
            # Enemy Distance
            # =================================================

            elif field == "enemy_distance":

                digits = (
                    self._extract_digits(
                        text
                    )
                )

                if digits:

                    value = (
                        int(digits)
                        / 10.0
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[
                            field
                        ] = round(
                            value,
                            1,
                        )

            # =================================================
            # Ship Name
            # =================================================

            elif field == "ship_name":

                clean_text = (
                    text.strip()
                )

                if clean_text:

                    self.cache[
                        field
                    ] = clean_text

            # =================================================
            # Max Speed
            # =================================================

            elif field == "max_speed":

                digits = (
                    self._extract_digits(
                        text
                    )
                )

                if digits:

                    value = (
                        int(digits)
                        / 10.0
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[
                            field
                        ] = round(
                            value,
                            1,
                        )

        except Exception as exc:

            self.error_count += 1

            print(
                f"[OCR] "
                f"{field} 识别失败: {exc}"
            )

        finally:

            self.future = None

            self.current_field = None

    # =========================================================
    # Submit
    # =========================================================

    def _submit(
        self,
        field: str,
        roi: np.ndarray,
        geometry: Optional[dict],
    ) -> bool:

        if self.future is not None:

            return False

        now = (
            time.perf_counter()
        )

        last_submit = (
            self.last_submit_time.get(
                field,
                0.0,
            )
        )

        if (
            now
            - last_submit
            < self.min_interval[field]
        ):

            return False

        image = roi.copy()

        self.future = (
            self.executor.submit(
                self._recognize,
                field,
                image,
                geometry,
            )
        )

        self.current_field = field

        self.last_submit_time[
            field
        ] = now

        return True

    # =========================================================
    # Update
    # =========================================================

    def update(
        self,
        frame_bgr: np.ndarray,
        roi_map: Dict,
        geometry: Optional[dict],
        locked: bool,
    ):

        # =====================================================
        # Poll
        # =====================================================

        self.poll()

        # =====================================================
        # Lock
        # =====================================================

        self.set_locked(
            locked
        )

        if not locked:

            return

        # =====================================================
        # Worker busy
        # =====================================================

        if self.future is not None:

            return

        now = (
            time.perf_counter()
        )

        # =====================================================
        # Static fields
        # =====================================================

        static_due = (

            self.static_refresh_needed

            or

            (
                now
                - self.last_static_refresh
                >= STATIC_REFRESH_INTERVAL
            )
        )

        if static_due:

            if not self.static_queue:

                self.static_queue = [

                    "ship_name",

                    "max_speed",
                ]

            field = (
                self.static_queue[0]
            )

            if field in roi_map:

                rx, ry, rw, rh = (
                    roi_map[field]
                )

                roi = frame_bgr[
                    ry:ry + rh,
                    rx:rx + rw,
                ]

                if roi.size > 0:

                    if self._submit(
                        field,
                        roi,
                        geometry,
                    ):

                        self.static_queue.pop(
                            0
                        )

                        if not self.static_queue:

                            self.static_refresh_needed = (
                                False
                            )

                            self.last_static_refresh = (
                                now
                            )

                        return

            else:

                self.static_queue.pop(
                    0
                )

            if not self.static_queue:

                self.static_refresh_needed = (
                    False
                )

                self.last_static_refresh = (
                    now
                )

        # =====================================================
        # Dynamic fields
        # =====================================================

        field_count = len(
            self.DYNAMIC_FIELDS
        )

        for i in range(
            field_count
        ):

            index = (
                self.round_robin_index
                + i
            ) % field_count

            field = (
                self.DYNAMIC_FIELDS[
                    index
                ]
            )

            if field not in roi_map:

                continue

            rx, ry, rw, rh = (
                roi_map[field]
            )

            roi = frame_bgr[
                ry:ry + rh,
                rx:rx + rw,
            ]

            if roi.size == 0:

                continue

            # -------------------------------------------------
            # Enemy Distance
            # -------------------------------------------------

            if field == "enemy_distance":

                if self._submit(
                    field,
                    roi,
                    geometry,
                ):

                    self.round_robin_index = (
                        index + 1
                    ) % field_count

                    return

                continue

            # -------------------------------------------------
            # 普通动态字段
            # -------------------------------------------------

            if not self._has_changed(
                field,
                roi,
            ):

                continue

            if self._submit(
                field,
                roi,
                geometry,
            ):

                self.round_robin_index = (
                    index + 1
                ) % field_count

                return

    # =========================================================
    # Standard Module Entry
    # =========================================================

    def process(
        self,
        frame_bgr: np.ndarray,
        roi_map: Dict,
        geometry: Optional[dict],
        locked: bool,
    ) -> Dict:

        self.update(
            frame_bgr=frame_bgr,
            roi_map=roi_map,
            geometry=geometry,
            locked=locked,
        )

        return self.get_cache()

    # =========================================================
    # Cache
    # =========================================================

    def get_cache(
        self,
    ) -> Dict:

        return dict(
            self.cache
        )

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        if self.executor is not None:

            self.executor.shutdown(
                wait=True,
                cancel_futures=True,
            )

        self.executor = None

        self.future = None

        with self._debug_image_lock:

            for field in DEBUG_FIELDS:

                self._debug_processed_images[
                    field
                ] = None


# ============================================================
# Standalone Debug
# ============================================================

if __name__ == "__main__":

    print(
        "[OCR] OCRParser 模块加载测试..."
    )

    try:

        parser = OCRParser()

        print(
            "[OCR] OCRParser 初始化成功。"
        )

    except Exception as exc:

        print(
            "[OCR] OCRParser 初始化失败："
        )

        print(
            exc
        )