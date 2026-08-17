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
# VisionManager 可以通过 OCR_CONFIG 覆盖这些默认值。
# ============================================================


# ============================================================
# 1. 数字放大倍率
#
# 所有数字字段统一使用。
#
# 调大：
#     小数字更容易识别
#     OCR 输入图像更大
#     CPU 开销增加
#
# 调小：
#     更快
#     但小字体可能识别困难
# ============================================================

NUMERIC_SCALE = 2.0


# ============================================================
# 2. 普通文字放大倍率
#
# 舰名使用。
# ============================================================

NORMAL_SCALE_SMALL = 2.0
NORMAL_SCALE_LARGE = 1.5

# ROI 高度达到该值后使用 LARGE。
NORMAL_TEXT_HEIGHT_THRESHOLD = 30


# ============================================================
# 3. 白 / 灰文字 HSV 筛选
#
#     V >= BINARY_MIN_VALUE
#     S <= BINARY_MAX_SATURATION
#
# 白色 / 灰色：
#     S 通常较低
#
# 黄色 / 蓝色 / 红色：
#     S 通常较高
# ============================================================

BINARY_MIN_VALUE = 150
BINARY_MAX_SATURATION = 80


# ============================================================
# 4. 舰名形态学 CLOSE
#
# 数字不做 CLOSE。
# 普通文字使用轻度 CLOSE。
# ============================================================

NORMAL_CLOSE_KERNEL = (2, 2)


# ============================================================
# 5. OCR 输入边框
# ============================================================

BORDER_TOP = 8
BORDER_BOTTOM = 8
BORDER_LEFT = 12
BORDER_RIGHT = 12


# ============================================================
# 6. OCR Worker 数量
#
# 当前建议保持 1。
# ============================================================

OCR_WORKER_COUNT = 1


# ============================================================
# 7. OCR 最小刷新间隔
#
# 单位：秒。
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
# 8. ROI 变化阈值
#
# 数值越小：
#     更敏感
#     OCR 更频繁
#
# 数值越大：
#     更省 CPU
# ============================================================

CHANGE_THRESHOLDS = {

    "flight_time": 2.0,

    "aim_distance": 2.0,

    "enemy_angle": 1.5,

    "our_angle": 1.5,
}


# ============================================================
# 9. 静态信息刷新
# ============================================================

STATIC_REFRESH_INTERVAL = 2.0


# ============================================================
# 10. OCR 数值合法范围
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
# 11. Debug 图片字段
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
# Windows / ONNX Runtime DLL 处理
# ============================================================

def _prepare_windows_dll_environment():
    """
    尽量在 RapidOCR / ONNX Runtime 导入前准备 Windows DLL 环境。

    解决部分 Windows + Python 3.12 环境中：

        onnxruntime_pybind11_state
        DLL initialization routine failed

    的问题。

    这里只做环境准备，不改变 OCR 算法。
    """

    if platform.system() != "Windows":
        return

    # ========================================================
    # 1. Python 自身 DLL 路径
    # ========================================================

    dll_directories = set()

    python_root = (
        Path(sys.executable)
        .resolve()
        .parent
    )

    dll_directories.add(
        python_root
    )

    python_dll_dir = (
        python_root / "DLLs"
    )

    if python_dll_dir.exists():

        dll_directories.add(
            python_dll_dir
        )

    # ========================================================
    # 2. site-packages 路径
    # ========================================================

    try:

        site_packages = site.getsitepackages()

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
    # 3. ONNX Runtime capi
    # ========================================================

    for base in list(dll_directories):

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
    # 4. os.add_dll_directory
    #
    # Python 3.8+ Windows 支持。
    # ========================================================

    add_dll_directory = getattr(
        os,
        "add_dll_directory",
        None,
    )

    if add_dll_directory is not None:

        for directory in dll_directories:

            try:

                if directory.exists():

                    add_dll_directory(
                        str(directory)
                    )

            except Exception:

                pass

    # ========================================================
    # 5. 预加载 MSVC Runtime
    #
    # ONNX Runtime 官方文档也建议 Windows
    # 环境确保 MSVC runtime 可用。
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

            # 可能本身已经存在但无法通过当前路径加载。
            # 不在这里直接终止，让后面的详细错误处理。
            pass


# ============================================================
# 在 cv2 / RapidOCR 导入前准备 DLL
# ============================================================

_prepare_windows_dll_environment()


class OCRParser:

    DYNAMIC_FIELDS = (

        "flight_time",

        "aim_distance",

        "enemy_angle",

        "our_angle",

        "enemy_distance",
    )

    STATIC_FIELDS = (

        "ship_name",

        "max_speed",
    )

    def __init__(
        self,

        # =====================================================
        # 可由 VisionManager 覆盖的参数
        # =====================================================

        numeric_scale: float = NUMERIC_SCALE,

        normal_scale_small: float = NORMAL_SCALE_SMALL,

        normal_scale_large: float = NORMAL_SCALE_LARGE,

        normal_text_height_threshold: int = (
            NORMAL_TEXT_HEIGHT_THRESHOLD
        ),

        binary_min_value: int = BINARY_MIN_VALUE,

        binary_max_saturation: int = (
            BINARY_MAX_SATURATION
        ),

        border_top: int = BORDER_TOP,

        border_bottom: int = BORDER_BOTTOM,

        border_left: int = BORDER_LEFT,

        border_right: int = BORDER_RIGHT,
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

        self.normal_text_height_threshold = int(
            normal_text_height_threshold
        )

        self.binary_min_value = int(
            binary_min_value
        )

        self.binary_max_saturation = int(
            binary_max_saturation
        )

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
                "\n"
                "如果错误包含：\n"
                "DLL initialization routine failed\n"
                "\n"
                "请检查：\n"
                "1. Microsoft Visual C++ Redistributable x64\n"
                "2. Python / NumPy / ONNX Runtime 位数是否一致\n"
                "3. onnxruntime 是否能单独 import\n"
            ) from exc

        self.EngineType = EngineType
        self.LangRec = LangRec
        self.ModelType = ModelType
        self.OCRVersion = OCRVersion
        self.RapidOCR = RapidOCR

        print(
            "[OCR] 正在加载 RapidOCR..."
        )

        # =====================================================
        # RapidOCR 参数
        #
        # 注意：
        #
        # RapidOCR 当前版本的构造函数会初始化
        # Det / Cls / Rec 三个模型，即使最终调用时
        # use_det=False / use_cls=False。
        #
        # 因此 ONNX Runtime 必须在构造阶段可用。
        # =====================================================

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
            #
            # 当前虽然不运行，
            # RapidOCR 仍会在初始化时创建 detector。
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
                "当前错误来自 ONNX Runtime / Windows DLL，\n"
                "而不是 OCR ROI 或图像预处理。\n"
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
        # OCR Submit 时间
        # =====================================================

        self.last_submit_time = {}

        self.min_interval = dict(
            MIN_INTERVALS
        )

        self.change_threshold = dict(
            CHANGE_THRESHOLDS
        )

        # =====================================================
        # Static Queue
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

            return "", 0.0

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

            return "", 0.0

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
    # 白 / 灰文字
    # =========================================================

    def _white_text_binary(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        lower = np.array(
            [
                0,
                0,
                self.binary_min_value,
            ],
            dtype=np.uint8,
        )

        upper = np.array(
            [
                179,
                self.binary_max_saturation,
                255,
            ],
            dtype=np.uint8,
        )

        return cv2.inRange(
            hsv,
            lower,
            upper,
        )

    # =========================================================
    # 普通文字 CLOSE
    # =========================================================

    def _clean_normal_binary(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            NORMAL_CLOSE_KERNEL,
        )

        return cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

    # =========================================================
    # 反色
    # =========================================================

    @staticmethod
    def _invert(
        image: np.ndarray,
    ) -> np.ndarray:

        return cv2.bitwise_not(
            image
        )

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
    # BGR
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
    # 通用 preprocessing
    # =========================================================

    def _preprocess_binary(
        self,
        roi: np.ndarray,
        scale: float,
        normal_text: bool,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        # 1. 放大
        image = self._resize_roi(
            roi,
            scale,
        )

        # 2. 白 / 灰文字提取
        binary = (
            self._white_text_binary(
                image
            )
        )

        # 3. 普通文字才进行 CLOSE
        if normal_text:

            binary = (
                self._clean_normal_binary(
                    binary
                )
            )

        # 4. 反色
        binary = self._invert(
            binary
        )

        # 5. 加白边
        binary = self._add_border(
            binary
        )

        # 6. 转 BGR
        binary = self._gray_to_bgr(
            binary
        )

        return binary

    # =========================================================
    # 数字
    # =========================================================

    def _preprocess_numeric(
        self,
        roi: np.ndarray,
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
        )

    # =========================================================
    # 普通文字
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
        )

    # =========================================================
    # 字段 → preprocessing
    # =========================================================

    def _prepare_roi(
        self,
        field: str,
        roi: np.ndarray,
        geometry: Optional[dict],
    ) -> np.ndarray:

        if field in (

            "flight_time",

            "aim_distance",

            "enemy_angle",

            "our_angle",

            "enemy_distance",

            "max_speed",
        ):

            return self._preprocess_numeric(
                roi
            )

        return self._preprocess_normal(
            roi
        )

    # =========================================================
    # Debug
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

        image_copy = np.ascontiguousarray(
            image.copy()
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

            for field, image in (
                self._debug_processed_images.items()
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

        image = (
            self._prepare_roi(
                field,
                roi,
                geometry,
            )
        )

        # 保存真正送入 OCR 的图片
        self._set_debug_image(
            field,
            image,
        )

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

        cost_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            field,
            text,
            score,
            cost_ms,
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

        self.last_probe[field] = gray

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
    # 静态刷新
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
    # 锁定
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
    # 数字纠正
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
    # 数值检查
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
    # OCR 完成
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

            self.last_ocr_ms = cost_ms

            self.last_ocr_field = field

            self.last_ocr_text = text

            self.last_ocr_score = score

            # =================================================
            # Flight Time
            # =================================================

            if field == "flight_time":

                digits = self._extract_digits(
                    text
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

                        self.cache[field] = (
                            round(
                                value,
                                2,
                            )
                        )

            # =================================================
            # Aim Distance
            # =================================================

            elif field == "aim_distance":

                digits = self._extract_digits(
                    text
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

                        self.cache[field] = (
                            round(
                                value,
                                2,
                            )
                        )

            # =================================================
            # Enemy Angle
            # =================================================

            elif field == "enemy_angle":

                digits = self._extract_digits(
                    text
                )

                if digits:

                    value = float(
                        int(digits)
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[field] = value

            # =================================================
            # Our Angle
            # =================================================

            elif field == "our_angle":

                digits = self._extract_digits(
                    text
                )

                if digits:

                    value = float(
                        int(digits)
                    )

                    if self._validate_numeric_value(
                        field,
                        value,
                    ):

                        self.cache[field] = value

            # =================================================
            # Enemy Distance
            # =================================================

            elif field == "enemy_distance":

                digits = self._extract_digits(
                    text
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

                        self.cache[field] = (
                            round(
                                value,
                                1,
                            )
                        )

            # =================================================
            # Ship Name
            # =================================================

            elif field == "ship_name":

                clean_text = text.strip()

                if clean_text:

                    self.cache[field] = (
                        clean_text
                    )

            # =================================================
            # Max Speed
            # =================================================

            elif field == "max_speed":

                digits = self._extract_digits(
                    text
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

                        self.cache[field] = (
                            round(
                                value,
                                1,
                            )
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
            now - last_submit
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

        self.last_submit_time[field] = now

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

        self.poll()

        self.set_locked(
            locked
        )

        if not locked:
            return

        if self.future is not None:
            return

        now = (
            time.perf_counter()
        )

        # =====================================================
        # Static
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

                        # 防止静态 OCR 永远循环，
                        # 让动态 OCR 获得执行机会。
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
        # Dynamic
        # =====================================================

        field_count = len(
            self.DYNAMIC_FIELDS
        )

        for i in range(field_count):

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
            # 其它动态字段
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
    # 标准入口
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