# 透明窗口：基于 PyQt/PySide 或 Direct2D 创建无边框穿透窗口
import sys
import os
import time
import cv2
import numpy as np
import platform
import ctypes
from dataclasses import asdict

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QRect, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
from PyQt5.QtWidgets import QApplication, QMainWindow

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.data_models import TargetState
from common.config_loader import ConfigLoader
from recognition.vision_manager import VisionManager

# ================= 🚀 后台视觉处理工作线程 =================
class VisionWorker(QThread):
    """后台高速处理图像，不卡顿前台 60FPS 渲染"""
    data_updated = pyqtSignal(dict)

    def __init__(self, vision_manager: VisionManager):
        super().__init__()
        self.vision = vision_manager
        self.running = True

    def run(self):
        while self.running:
            start_time = time.time()
            frame = self.vision.capturer.grab_screen()
            
            # 运行核心识别
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            anchor = self.vision.indicators.find_locked_anchor(frame_hsv)
            result = self.vision.process_frame(frame)

            # 读取 HUD 原始识别文字
            hud_values = {}
            for name, (rx, ry, rw, rh) in self.vision.hud_offsets.items():
                roi = frame[ry:ry+rh, rx:rx+rw]
                if "angle" in name:
                    val = self.vision.ocr.extract_number(roi, is_angle=True)
                elif "name" in name:
                    val = self.vision.ocr.extract_ship_name(roi)
                else:
                    val = self.vision.ocr.extract_number(roi)
                hud_values[name] = val

            process_fps = 1.0 / (time.time() - start_time + 1e-6)

            # 组装数据包发送给前台
            payload = {
                "locked": result is not None,
                "anchor": anchor,
                "result": result,
                "hud_values": hud_values,
                "process_fps": process_fps
            }
            self.data_updated.emit(payload)
            
            # 限制识别频率在 30-60 FPS，避免 GPU 占满
            time.sleep(0.015)

    def stop(self):
        self.running = False
        self.wait()

# ================= 🪟 1:1 原生全屏透明穿透窗口 =================
class TransparentOverlay(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = ConfigLoader("config.yaml")
        self.vision = VisionManager(self.config)

        self.sw = self.vision.sw
        self.sh = self.vision.sh
        self.cx = self.vision.cx
        self.cy = self.vision.cy

        self.latest_data = {
            "locked": False,
            "anchor": None,
            "result": None,
            "hud_values": {},
            "process_fps": 0.0
        }

        self.init_window()
        self.init_win32_passthrough()

        # 启动后台识别线程
        self.worker = VisionWorker(self.vision)
        self.worker.data_updated.connect(self.on_data_received)
        self.worker.start()

        # 60 FPS 平滑刷新定时器
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update)
        self.render_timer.start(16) # ~60 FPS

    def init_window(self):
        """初始化全屏无边框透明窗口"""
        self.setGeometry(0, 0, self.sw, self.sh)
        self.setWindowTitle("WoWs AI Aim Overlay")

        # Qt 窗口属性：无边框 + 顶层置顶 + 穿透背景 + 工具窗口(不占任务栏)
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.SubWindow |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForInput, True)

    def init_win32_passthrough(self):
        """Windows 底层 API：100% 鼠标键盘穿透 + 截屏隐身"""
        if platform.system() == "Windows":
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            
            # 1. 截屏隐身 (WDA_EXCLUDEFROMCAPTURE = 0x11)，防止 mss 自己截到自己画的线
            user32.SetWindowDisplayAffinity(hwnd, 0x00000011)

            # 2. 鼠标键盘绝对穿透 (WS_EX_TRANSPARENT | WS_EX_LAYERED)
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00000080
            WS_EX_NOACTIVATE = 0x08000000

            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE)

    def on_data_received(self, data):
        self.latest_data = data

    def paintEvent(self, event):
        """绘制所有原生 1:1 高清 UI"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        locked = self.latest_data["locked"]
        anchor = self.latest_data["anchor"]
        result = self.latest_data["result"]
        hud_values = self.latest_data["hud_values"]
        fps = self.latest_data["process_fps"]

        # ---------------- 1. 绘制准星中央 HUD 绿色高亮框与读数 ----------------
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        
        for name, (rx, ry, rw, rh) in self.vision.hud_offsets.items():
            val = hud_values.get(name, "")
            
            # 绘制极细半透明绿色边框
            painter.setPen(QPen(QColor(0, 255, 128, 200), 1))
            painter.setBrush(QBrush(QColor(0, 255, 128, 25))) # 微弱底色
            painter.drawRect(rx, ry, rw, rh)

            # 绘制悬浮数值
            painter.setPen(QPen(QColor(0, 255, 128, 255), 1))
            painter.drawText(rx, ry - 4, f"{val}")

        # ---------------- 2. 绘制锁定敌舰头顶专属动态雷达 ----------------
        if locked and anchor:
            target_cx, target_cy, hx_left, hy_top = anchor
            state, ship_name, max_speed = result
            
            ring_r = self.vision.indicators.RING_RADIUS
            light_off = self.vision.indicators.LIGHT_OFFSET

            # (A) 绘制血条锁定左锚点与中心引导线
            painter.setPen(QPen(QColor(0, 255, 255, 180), 1, Qt.DashLine))
            painter.drawLine(hx_left, hy_top, target_cx, target_cy)
            painter.setBrush(QBrush(QColor(0, 255, 255, 255)))
            painter.drawEllipse(QPoint(hx_left, hy_top), 3, 3)

            # (B) 绘制几何圆心十字 (青蓝色)
            painter.setPen(QPen(QColor(0, 200, 255, 255), 2))
            painter.drawLine(target_cx - 8, target_cy, target_cx + 8, target_cy)
            painter.drawLine(target_cx, target_cy - 8, target_cx, target_cy + 8)

            # (C) 绘制航速圆环
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(target_cx, target_cy), ring_r, ring_r)

            # (D) 绘制大头针指示灯锁定框 (亮黄)
            light_y = target_cy - ring_r - light_off
            painter.setPen(QPen(QColor(255, 220, 0, 255), 1.5))
            painter.drawRect(target_cx - 5, light_y - 5, 10, 10)

            # (E) 在敌舰头上绘制悬浮状态卡片
            dir_str = "FWD" if state.direction_state == 1 else "REV" if state.direction_state == -1 else "STOP"
            card_text = f"🎯 {ship_name} | {state.speed_fraction*8:.0f}/8 ({dir_str}) | {state.distance:.1f}km"
            
            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            # 绘制黑底发光字
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.drawText(target_cx - 60, light_y - 12, card_text)
            painter.setPen(QPen(QColor(0, 255, 200, 255), 1))
            painter.drawText(target_cx - 60, light_y - 12, card_text)

        # ---------------- 3. 屏幕左上角数据结构实时仪表盘 (HUD Card) ----------------
        self.draw_dashboard(painter, locked, result, fps)

    def draw_dashboard(self, painter: QPainter, locked: bool, result: tuple, fps: float):
        """屏幕左上方半透明数据监视看板"""
        card_x, card_y = 25, 25
        card_w, card_h = 320, 180

        # 半透明磨砂黑底色
        painter.setPen(QPen(QColor(0, 255, 128, 180) if locked else QColor(100, 100, 100, 100), 1))
        painter.setBrush(QBrush(QColor(15, 15, 15, 210)))
        painter.drawRoundedRect(QRect(card_x, card_y, card_w, card_h), 6, 6)

        # 标题与 FPS
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255, 240)))
        painter.drawText(card_x + 15, card_y + 25, "AI PRE-AIM FCS MONITOR")
        
        painter.setFont(QFont("Consolas", 9))
        painter.setPen(QPen(QColor(0, 255, 128, 255) if locked else QColor(255, 80, 80, 255)))
        status_text = f"[{'LOCKED' if locked else 'SEARCHING'}] {fps:.1f} FPS"
        painter.drawText(card_x + card_w - 120, card_y + 25, status_text)

        # 分割线
        painter.setPen(QPen(QColor(60, 60, 60, 150), 1))
        painter.drawLine(card_x + 10, card_y + 35, card_x + card_w - 10, card_y + 35)

        # 详细 TargetState 读数
        painter.setFont(QFont("Segoe UI", 9))
        if locked and result:
            state, ship_name, max_speed = result
            dir_str = "FORWARD" if state.direction_state == 1 else "REVERSE" if state.direction_state == -1 else "STOP"
            
            lines = [
                f"Target Ship   : {ship_name} ({max_speed:.1f} kts)",
                f"Flight Time   : {state.flight_time:.2f} s",
                f"Distance (Aim): {state.distance:.2f} km",
                f"Relative Angle: {state.relative_angle:.1f}°",
                f"Engine State  : {state.speed_fraction*8:.0f}/8 ({dir_str})"
            ]
            for i, line in enumerate(lines):
                painter.setPen(QPen(QColor(200, 200, 200, 240)))
                painter.drawText(card_x + 15, card_y + 60 + i * 22, line)
        else:
            painter.setPen(QPen(QColor(140, 140, 140, 200)))
            painter.drawText(card_x + 15, card_y + 70, "Waiting for target lock...")
            painter.drawText(card_x + 15, card_y + 95, "Aim with telescope & press 'X'")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = TransparentOverlay()
    overlay.show()
    sys.exit(app.exec_())