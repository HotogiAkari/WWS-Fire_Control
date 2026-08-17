import concurrent.futures
import re
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


try:

    from rapidocr import (
        EngineType,
        LangRec,
        ModelType,
        OCRVersion,
        RapidOCR,
    )

    HAS_RAPID_OCR = True

except ImportError:

    HAS_RAPID_OCR = False

    RapidOCR = None
    EngineType = None
    LangRec = None
    ModelType = None
    OCRVersion = None


class OCRParser:
    """
    OCR 识别模块。

    职责：

        1. 对已经确定位置的 ROI 做预处理
        2. 调用 RapidOCR recognition-only
        3. 解析文字 / 数字
        4. 缓存最近一次有效结果
        5. 控制不同字段的 OCR 频率
        6. 保存最近一次真正送入 OCR 的处理后图像，
           供 native_overlay 调试显示

    不负责：

        - ROI 定位
        - 血条追踪
        - 航速圆环
        - 小地图
        - 目标定位

    当前设计：

        游戏 HUD：

            白色文字
            +
            黑色阴影 / 描边

        数字：

            原始 ROI
                ↓
            放大
                ↓
            灰度
                ↓
            固定亮度阈值
                ↓
            不做 CLOSE
                ↓
            反色
                ↓
            白色边框
                ↓
            BGR
                ↓
            RapidOCR recognition-only

        舰名：

            原始 ROI
                ↓
            放大
                ↓
            灰度
                ↓
            固定亮度阈值
                ↓
            轻度 CLOSE
                ↓
            反色
                ↓
            白色边框
                ↓
            BGR
                ↓
            RapidOCR
    """

    # =========================================================
    # 动态字段
    # =========================================================

    DYNAMIC_FIELDS = (
        "flight_time",
        "aim_distance",
        "enemy_angle",
        "our_angle",
        "enemy_distance",
    )

    # =========================================================
    # 静态字段
    # =========================================================

    STATIC_FIELDS = (
        "ship_name",
        "max_speed",
    )

    # =========================================================
    # 静态信息刷新周期
    # =========================================================

    STATIC_REFRESH_INTERVAL = 2.0

    # =========================================================
    # HUD 二值化
    #
    # 游戏文字是白色，因此不再使用 Otsu。
    #
    # 直接保留高亮像素：
    #
    #     gray >= threshold
    #         ↓
    #       白色
    #
    #     gray < threshold
    #         ↓
    #       黑色
    #
    # 这样可以更直接地排除：
    #
    #     游戏背景
    #     黑色阴影
    #     深色描边
    #
    # =========================================================

    HUD_BINARY_THRESHOLD = 180

    # =========================================================
    # 舰名使用的轻度 CLOSE
    #
    # 数字不会使用这个处理。
    # =========================================================

    NORMAL_CLOSE_KERNEL = (2, 2)

    # =========================================================
    # OCR 输入边缘
    # =========================================================

    BORDER_TOP = 8
    BORDER_BOTTOM = 8
    BORDER_LEFT = 12
    BORDER_RIGHT = 12

    # =========================================================
    # Debug Preview
    # =========================================================
    #
    # 保存“最终送入 RapidOCR 的图像”。
    #
    # 注意：
    #
    #     保存的是 _recognize() 内部经过 preprocessing
    #     后的最终 BGR ndarray。
    #
    # 所以 native_overlay 显示的内容与 OCR 实际收到的
    # 图像完全一致。
    # =========================================================

    def __init__(self):

        if not HAS_RAPID_OCR:

            raise RuntimeError(
                "\n"
                "未安装 RapidOCR。\n"
                "\n"
                "请运行：\n"
                "pip install -U rapidocr onnxruntime\n"
            )

        print(
            "[OCR] 正在加载 RapidOCR..."
        )

        # =====================================================
        # RapidOCR
        #
        # ROI 已经由 IndicatorParser / VisionManager 确定，
        # 因此只使用 Recognition。
        # =====================================================

        params = {

            "Global.use_det": False,

            "Global.use_cls": False,

            "Global.use_rec": True,

            "Global.use_preprocess_img": False,

            "Global.log_level":
                "critical",

            # =================================================
            # Recognition
            # =================================================

            "Rec.engine_type":
                EngineType.ONNXRUNTIME,

            "Rec.lang_type":
                LangRec.CH,

            "Rec.model_type":
                ModelType.MOBILE,

            "Rec.ocr_version":
                OCRVersion.PPOCRV5,

            # =================================================
            # ONNX Runtime
            # =================================================

            "EngineConfig.onnxruntime.intra_op_num_threads": 2,

            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }

        self.engine = RapidOCR(
            params=params
        )

        print(
            "[OCR] RapidOCR 加载完成。"
        )

        # =====================================================
        # OCR Worker
        # =====================================================

        self.executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="OCRWorker",
            )
        )

        self.future = None

        self.current_field: Optional[str] = None

        # =====================================================
        # OCR Cache
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
        # ROI 变化检测缓存
        # =====================================================

        self.last_probe = {}

        # =====================================================
        # 最近一次提交 OCR 的时间
        # =====================================================

        self.last_submit_time = {}

        # =====================================================
        # 各字段 OCR 最小间隔
        # =====================================================

        self.min_interval = {

            "flight_time": 0.040,

            "aim_distance": 0.040,

            "enemy_angle": 0.050,

            "our_angle": 0.050,

            "enemy_distance": 1.0,

            "ship_name": 0.0,

            "max_speed": 0.0,
        }

        # =====================================================
        # ROI 变化触发阈值
        # =====================================================

        self.change_threshold = {

            "flight_time": 2.0,

            "aim_distance": 2.0,

            "enemy_angle": 1.5,

            "our_angle": 1.5,

            "enemy_distance": 1.0,
        }

        # =====================================================
        # 静态信息 OCR 队列
        # =====================================================

        self.static_queue = []

        self.static_refresh_needed = True

        self.last_static_refresh = 0.0

        # =====================================================
        # 动态 OCR Round Robin
        # =====================================================

        self.round_robin_index = 0

        # =====================================================
        # 当前目标是否锁定
        # =====================================================

        self.was_locked = False

        # =====================================================
        # Debug / Performance
        # =====================================================

        self.last_ocr_ms = 0.0

        self.last_ocr_field = ""

        self.last_ocr_text = ""

        self.last_ocr_score = 0.0

        self.error_count = 0

        # =====================================================
        # Debug OCR Preview
        #
        # OCR Worker 和 Qt GUI 不在同一个线程，
        # 所以使用 Lock 保护。
        # =====================================================

        self._debug_image_lock = (
            threading.Lock()
        )

        self._debug_processed_image = None

        self._debug_processed_field = ""

    # =========================================================
    # RapidOCR
    # =========================================================

    def _run_recognition(
        self,
        image: np.ndarray,
    ):

        """
        只执行 Recognition。

        image：
            BGR ndarray
        """

        return self.engine(
            image,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )

    # =========================================================
    # OCR 输出解析
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

        # =====================================================
        # 文本
        # =====================================================

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

        # =====================================================
        # Score
        # =====================================================

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

                score = 0.0

        return (
            text,
            score,
        )

    # =========================================================
    # 放大
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
    # 灰度化
    # =========================================================

    @staticmethod
    def _to_gray(
        image: np.ndarray,
    ) -> np.ndarray:

        if image.ndim == 2:

            return image

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    # =========================================================
    # HUD 固定阈值二值化
    #
    # 不再使用 Otsu。
    #
    # 原因：
    #
    #     HUD 文字明确是高亮白色，
    #     我们需要直接提取“亮像素”，
    #     而不是让 Otsu 根据整个 ROI 的直方图
    #     猜测前景和背景。
    # =========================================================

    @classmethod
    def _hud_binary(
        cls,
        gray: np.ndarray,
        threshold: Optional[int] = None,
    ) -> np.ndarray:

        if threshold is None:

            threshold = (
                cls.HUD_BINARY_THRESHOLD
            )

        _, binary = cv2.threshold(
            gray,
            int(threshold),
            255,
            cv2.THRESH_BINARY,
        )

        return binary

    # =========================================================
    # 舰名轻度 CLOSE
    #
    # 只用于普通文字。
    #
    # 数字不调用这里。
    # =========================================================

    @classmethod
    def _clean_normal_binary(
        cls,
        binary: np.ndarray,
    ) -> np.ndarray:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            cls.NORMAL_CLOSE_KERNEL,
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
    # 添加边缘
    # =========================================================

    @classmethod
    def _add_border(
        cls,
        image: np.ndarray,
    ) -> np.ndarray:

        return cv2.copyMakeBorder(
            image,
            cls.BORDER_TOP,
            cls.BORDER_BOTTOM,
            cls.BORDER_LEFT,
            cls.BORDER_RIGHT,
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
    # 通用二值 OCR preprocessing
    #
    # numeric=True：
    #
    #     不做 CLOSE
    #
    # normal：
    #
    #     允许轻度 CLOSE
    # =========================================================

    @classmethod
    def _preprocess_binary(
        cls,
        roi: np.ndarray,
        scale: float,
        numeric: bool = False,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        # =====================================================
        # 1. 放大
        # =====================================================

        image = cls._resize_roi(
            roi,
            scale,
        )

        # =====================================================
        # 2. 灰度
        # =====================================================

        gray = cls._to_gray(
            image
        )

        # =====================================================
        # 3. 固定亮度阈值
        # =====================================================

        binary = cls._hud_binary(
            gray
        )

        # =====================================================
        # 4. 数字不做 CLOSE
        #
        # 数字笔画较细，
        # CLOSE 容易产生：
        #
        #     1 → 7
        #     3 → 8
        #     5 → 6
        #
        # 等粘连问题。
        # =====================================================

        if not numeric:

            binary = cls._clean_normal_binary(
                binary
            )

        # =====================================================
        # 5. 反色
        #
        #     白色数字
        #         ↓
        #     黑色数字
        #
        #     黑色背景
        #         ↓
        #     白色背景
        # =====================================================

        binary = cls._invert(
            binary
        )

        # =====================================================
        # 6. 白色边框
        # =====================================================

        binary = cls._add_border(
            binary
        )

        # =====================================================
        # 7. BGR
        # =====================================================

        binary = cls._gray_to_bgr(
            binary
        )

        return binary

    # =========================================================
    # 普通字段
    #
    # 主要用于舰名。
    # =========================================================

    @classmethod
    def _preprocess_normal(
        cls,
        roi: np.ndarray,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        h = roi.shape[0]

        if h >= 30:

            scale = 1.5

        else:

            scale = 2.0

        return cls._preprocess_binary(
            roi,
            scale=scale,
            numeric=False,
        )

    # =========================================================
    # 数字字段
    #
    # 数字使用独立处理：
    #
    #     1. 更高放大倍率
    #     2. 固定高亮阈值
    #     3. 不使用 CLOSE
    # =========================================================

    @classmethod
    def _preprocess_numeric(
        cls,
        roi: np.ndarray,
        scale: float = 2.5,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        return cls._preprocess_binary(
            roi,
            scale=scale,
            numeric=True,
        )

    # =========================================================
    # 敌舰距离
    # =========================================================

    @classmethod
    def _preprocess_enemy_distance(
        cls,
        roi: np.ndarray,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        return cls._preprocess_binary(
            roi,
            scale=3.0,
            numeric=True,
        )

    # =========================================================
    # 角度
    #
    # 原始：
    #
    #     36 × 36
    #
    # 原版本裁掉 8px 后：
    #
    #     20 × 20
    #
    # 现在改为裁 5px：
    #
    #     26 × 26
    #
    # 这样保留更多数字本体。
    # =========================================================

    @classmethod
    def _preprocess_angle(
        cls,
        roi: np.ndarray,
        geometry: dict,
    ) -> np.ndarray:

        if (
            roi is None
            or roi.size == 0
        ):

            return roi

        h, w = roi.shape[:2]

        # =====================================================
        # 角度裁切
        # =====================================================

        left = max(
            0,
            int(
                geometry.get(
                    "angle_crop_left",
                    5,
                )
            ),
        )

        top = max(
            0,
            int(
                geometry.get(
                    "angle_crop_top",
                    5,
                )
            ),
        )

        right = max(
            0,
            int(
                geometry.get(
                    "angle_crop_right",
                    5,
                )
            ),
        )

        bottom = max(
            0,
            int(
                geometry.get(
                    "angle_crop_bottom",
                    5,
                )
            ),
        )

        # =====================================================
        # 防止裁剪参数导致 ROI 崩溃
        # =====================================================

        x1 = min(
            left,
            max(0, w - 1),
        )

        y1 = min(
            top,
            max(0, h - 1),
        )

        x2 = max(
            x1 + 1,
            w - right,
        )

        y2 = max(
            y1 + 1,
            h - bottom,
        )

        x2 = min(
            x2,
            w,
        )

        y2 = min(
            y2,
            h,
        )

        cropped = roi[
            y1:y2,
            x1:x2,
        ]

        # =====================================================
        # 数字较小
        #
        # 仍然使用 3×，
        # 但不进行 CLOSE。
        # =====================================================

        return cls._preprocess_numeric(
            cropped,
            scale=3.0,
        )

    # =========================================================
    # 根据字段选择预处理
    # =========================================================

    def _prepare_roi(
        self,
        field: str,
        roi: np.ndarray,
        geometry: Optional[dict],
    ) -> np.ndarray:

        # -----------------------------------------------------
        # 敌舰距离
        # -----------------------------------------------------

        if field == "enemy_distance":

            return (
                self._preprocess_enemy_distance(
                    roi
                )
            )

        # -----------------------------------------------------
        # 敌我角度
        # -----------------------------------------------------

        if field in (
            "enemy_angle",
            "our_angle",
        ):

            return (
                self._preprocess_angle(
                    roi,
                    geometry or {},
                )
            )

        # -----------------------------------------------------
        # 普通数字
        # -----------------------------------------------------

        if field in (
            "flight_time",
            "aim_distance",
            "max_speed",
        ):

            return (
                self._preprocess_numeric(
                    roi,
                    scale=2.5,
                )
            )

        # -----------------------------------------------------
        # 舰船名称
        # -----------------------------------------------------

        return (
            self._preprocess_normal(
                roi
            )
        )

    # =========================================================
    # 设置 Debug Preview
    # =========================================================

    def _set_debug_image(
        self,
        field: str,
        image: np.ndarray,
    ):

        if (
            image is None
            or image.size == 0
        ):

            return

        image_copy = np.ascontiguousarray(
            image.copy()
        )

        with self._debug_image_lock:

            self._debug_processed_image = (
                image_copy
            )

            self._debug_processed_field = (
                str(field)
            )

    # =========================================================
    # 获取 Debug Preview
    # =========================================================

    def get_debug_image(
        self,
    ) -> Tuple[
        Optional[np.ndarray],
        str,
    ]:

        with self._debug_image_lock:

            if (
                self._debug_processed_image
                is None
            ):

                return (
                    None,
                    self._debug_processed_field,
                )

            return (
                self._debug_processed_image.copy(),
                self._debug_processed_field,
            )

    # =========================================================
    # 单字段 OCR
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
        # preprocessing
        # =====================================================

        image = (
            self._prepare_roi(
                field,
                roi,
                geometry,
            )
        )

        # =====================================================
        # 保存真正送给 OCR 的最终图像
        #
        # 注意：
        #
        #     这里发生在 _run_recognition()
        #     之前，因此预览内容与 RapidOCR
        #     实际输入完全相同。
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
    # ROI 变化检测
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

        old = self.last_probe.get(
            field
        )

        self.last_probe[field] = (
            gray
        )

        # 第一次必须识别
        if old is None:

            return True

        # ROI 尺寸发生变化
        if old.shape != gray.shape:

            return True

        difference = cv2.absdiff(
            gray,
            old,
        )

        mean_difference = float(
            difference.mean()
        )

        return (
            mean_difference
            >= self.change_threshold[field]
        )

    # =========================================================
    # 强制刷新静态信息
    # =========================================================

    def force_static_refresh(self):

        self.static_refresh_needed = True

        self.static_queue = [
            "ship_name",
            "max_speed",
        ]

        self.last_static_refresh = 0.0

    # =========================================================
    # 锁定状态
    # =========================================================

    def set_locked(
        self,
        locked: bool,
    ):

        # =====================================================
        # 未锁定 → 锁定
        # =====================================================

        if (
            locked
            and not self.was_locked
        ):

            self.force_static_refresh()

        # =====================================================
        # 锁定 → 丢失
        # =====================================================

        elif (
            not locked
            and self.was_locked
        ):

            self.force_static_refresh()

        self.was_locked = locked

    # =========================================================
    # 数字字符修正
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
    # 提取数字
    # =========================================================

    @classmethod
    def _extract_digits(
        cls,
        text: str,
    ) -> str:

        text = cls._normalize_numeric_text(
            text
        )

        return re.sub(
            r"\D",
            "",
            text,
        )

    # =========================================================
    # OCR 数字范围校验
    #
    # 防止 OCR 偶尔把背景/噪声识别成非常离谱的数字。
    #
    # 返回：
    #
    #     True  -> 接受
    #     False -> 忽略这次 OCR
    # =========================================================

    @staticmethod
    def _validate_numeric_value(
        field: str,
        value: float,
    ) -> bool:

        ranges = {

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

        if field not in ranges:

            return True

        minimum, maximum = (
            ranges[field]
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

        field = (
            self.current_field
        )

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

            self.last_ocr_ms = cost_ms

            self.last_ocr_field = field

            self.last_ocr_text = text

            self.last_ocr_score = score

            # =================================================
            # 炮弹飞行时间
            #
            # 1234
            # ↓
            # 12.34
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
                        "flight_time",
                        value,
                    ):

                        self.cache[
                            "flight_time"
                        ] = round(
                            value,
                            2,
                        )

            # =================================================
            # 落点距离
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
                        "aim_distance",
                        value,
                    ):

                        self.cache[
                            "aim_distance"
                        ] = round(
                            value,
                            2,
                        )

            # =================================================
            # 敌方角度
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
                        "enemy_angle",
                        value,
                    ):

                        self.cache[
                            "enemy_angle"
                        ] = value

            # =================================================
            # 我方角度
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
                        "our_angle",
                        value,
                    ):

                        self.cache[
                            "our_angle"
                        ] = value

            # =================================================
            # 敌舰距离
            #
            # 123
            # ↓
            # 12.3
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
                        "enemy_distance",
                        value,
                    ):

                        self.cache[
                            "enemy_distance"
                        ] = round(
                            value,
                            1,
                        )

            # =================================================
            # 舰名
            # =================================================

            elif field == "ship_name":

                clean_text = text.strip()

                if clean_text:

                    self.cache[
                        "ship_name"
                    ] = clean_text

            # =================================================
            # 最大航速
            #
            # 210
            # ↓
            # 21.0
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
                        "max_speed",
                        value,
                    ):

                        self.cache[
                            "max_speed"
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
    # 提交 OCR
    # =========================================================

    def _submit(
        self,
        field: str,
        roi: np.ndarray,
        geometry: Optional[dict],
    ) -> bool:

        # OCR worker 正在工作
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

        # =====================================================
        # 最小间隔
        # =====================================================

        if (
            now - last_submit
            < self.min_interval[field]
        ):

            return False

        # =====================================================
        # copy
        #
        # worker 必须拥有自己的图像副本。
        # =====================================================

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

        # =====================================================
        # 先处理已经完成的 OCR
        # =====================================================

        self.poll()

        # =====================================================
        # 更新 Lock 状态
        # =====================================================

        self.set_locked(
            locked
        )

        if not locked:

            return

        # OCR worker 忙
        if self.future is not None:

            return

        # =====================================================
        # 静态字段
        # =====================================================

        now = (
            time.perf_counter()
        )

        static_due = (
            self.static_refresh_needed
            or
            (
                now
                - self.last_static_refresh
                >= self.STATIC_REFRESH_INTERVAL
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

                        return

            else:

                self.static_queue.pop(
                    0
                )

            if not self.static_queue:

                self.static_refresh_needed = False

                self.last_static_refresh = now

        # =====================================================
        # 动态字段
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

            # =================================================
            # 敌舰距离
            #
            # 跟随目标移动，因此不做 ROI 内容变化检测。
            # =================================================

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

            # =================================================
            # 普通动态字段
            # =================================================

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
    # 读取 Cache
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

            self._debug_processed_image = None

            self._debug_processed_field = ""