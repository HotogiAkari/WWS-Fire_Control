# recognition/capture.py

import threading
import time

import cv2
import mss
import numpy as np


class ScreenCapturer:
    """
    高速 Windows 屏幕抓取器。

    设计：
        - MSS 在实际抓屏线程中懒加载
        - BitBlt 失败时自动重建
        - 主显示器信息在初始化时读取
    """

    def __init__(self):
        # 不在这里长期持有 MSS，
        # 避免创建线程和使用线程不同。
        self._sct = None
        self._sct_thread_id = None

        # 获取主显示器信息
        with mss.MSS(backend="gdi") as sct:
            self.monitor = dict(sct.primary_monitor)

        self.width = int(self.monitor["width"])
        self.height = int(self.monitor["height"])

        self.capture_errors = 0

    # =========================================================
    # 确保当前线程有 MSS
    # =========================================================

    def _ensure_sct(self):
        thread_id = threading.get_ident()

        if (
            self._sct is not None
            and self._sct_thread_id == thread_id
        ):
            return

        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass

        self._sct = mss.MSS(backend="gdi")
        self._sct_thread_id = thread_id

    # =========================================================
    # 重建
    # =========================================================

    def _recreate_sct(self):
        try:
            if self._sct is not None:
                self._sct.close()
        except Exception:
            pass

        self._sct = None
        self._sct_thread_id = None

        self._ensure_sct()

    # =========================================================
    # 抓屏
    # =========================================================

    def grab_screen(
        self,
        bbox: list[int] | None = None,
    ) -> np.ndarray:

        self._ensure_sct()

        if bbox is None:
            monitor = self.monitor

        else:
            monitor = {
                "top": int(bbox[1]),
                "left": int(bbox[0]),
                "width": int(bbox[2]),
                "height": int(bbox[3]),
            }

        try:
            screenshot = self._sct.grab(monitor)

        except mss.exception.ScreenShotError:
            self.capture_errors += 1

            # BitBlt 失败，重建一次
            self._recreate_sct()

            screenshot = self._sct.grab(monitor)

        frame = np.asarray(
            screenshot,
            dtype=np.uint8,
        )[:, :, :3]

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        return frame

    # =========================================================
    # 关闭
    # =========================================================

    def close(self):

        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass

        self._sct = None
        self._sct_thread_id = None


# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":

    print("正在启动屏幕抓取测试...")

    capturer = ScreenCapturer()

    start_time = time.perf_counter()
    frames_count = 0

    try:

        while True:

            frame = capturer.grab_screen()

            frames_count += 1

            elapsed = (
                time.perf_counter() - start_time
            )

            fps = (
                frames_count / elapsed
                if elapsed > 0
                else 0.0
            )

            display_frame = cv2.resize(
                frame,
                (960, 540),
                interpolation=cv2.INTER_AREA,
            )

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                f"Capture Errors: {capturer.capture_errors}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Screen Capture Debug",
                display_frame,
            )

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:

        capturer.close()
        cv2.destroyAllWindows()