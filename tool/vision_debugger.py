import sys
import os
import cv2
import numpy as np
import mss
import platform

if platform.system() == "Windows":
    import ctypes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.config_loader import ConfigLoader
from recognition.vision_manager import VisionManager

def run_debugger():
    window_name = "Full Vision Pipeline Debugger"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    if platform.system() == "Windows":
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)

    # 加载配置与视觉管理器
    config = ConfigLoader("config.yaml")
    vision = VisionManager(config)

    print("全量视觉信息提取大盘已启动... (按 'Q' 退出)")

    while True:
        frame = vision.capturer.grab_screen()
        debug_frame = frame.copy()
        
        # 1. 运行全量提取
        result = vision.process_frame(frame)
        
        # 2. 绘制静态 UI 框
        for name in ["impact_info", "enemy_relative_angle", "enemy_max_speed"]:
            bbox = config.get_roi(name)
            if bbox and bbox[2] > 0:
                x, y, w, h = bbox
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 150, 0), 2)
                cv2.putText(debug_frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)

        # 3. 绘制动态目标信息
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        anchor = vision.find_locked_anchor(frame_hsv)
        
        if anchor:
            hx, hy, hw, hh, cx, cy = anchor
            
            # 画出血条与圆心
            cv2.rectangle(debug_frame, (hx, hy), (hx+hw, hy+hh), (0, 0, 255), 2)
            cv2.drawMarker(debug_frame, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
            
            # 画出外扩采样点 (紫色)
            _, _, sample_pts = vision.indicators.sample_speed_ring(frame_hsv, cx, cy)
            for px, py in sample_pts:
                cv2.circle(debug_frame, (px, py), 2, (255, 0, 255), -1)
                
            # 画出距离提取框 (青色)
            cv2.rectangle(debug_frame, (cx - 45, hy + 26), (cx + 45, hy + 48), (255, 255, 0), 1)
            
            # 画出指示灯检测点
            light_y = cy - vision.indicators.RING_RADIUS - vision.indicators.LIGHT_OFFSET
            cv2.rectangle(debug_frame, (cx-3, light_y-3), (cx+3, light_y+3), (0, 255, 255), 1)

        # 4. 在屏幕左上角打印解析结果看板
        if result:
            state, ship_name = result
            dir_str = "前进" if state.direction_state == 1 else "倒退" if state.direction_state == -1 else "停船"
            
            dashboard = [
                f"【目标】: {ship_name}",
                f"【航速档位】: {state.speed_fraction * 8:.0f}/8 ({dir_str})",
                f"【目标距离】: {state.distance:.1f} km (落点: {state.aiming_distance:.1f} km)",
                f"【飞行时间】: {state.flight_time:.2f} s",
                f"【相对角度】: {state.relative_angle:.1f}°"
            ]
            
            # 绘制信息底板
            cv2.rectangle(debug_frame, (20, 20), (450, 180), (0, 0, 0), -1)
            cv2.rectangle(debug_frame, (20, 20), (450, 180), (0, 255, 0), 2)
            
            for i, line in enumerate(dashboard):
                # 针对中文字符串使用基础绘制或拼音显示
                cv2.putText(debug_frame, line, (35, 55 + i * 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(debug_frame, "STATUS: SEARCHING / UNLOCKED", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_debugger()