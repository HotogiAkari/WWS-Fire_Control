# [模块接口] 接收历史帧，调用引擎，输出最终的 TrajectoryPath
from typing import List
from common.data_models import TargetState, ShipStaticData, TrajectoryPath
from common.config_loader import ConfigLoader

class PredictionManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.history_window: List[TargetState] = []
        self.max_history = self.config.get_param("history_frames", 30)
        
    def reset_history(self):
        """当丢失锁定或切换目标时，清空历史数据"""
        self.history_window.clear()

    def update_and_predict(self, current_state: TargetState, ship_data: ShipStaticData) -> TrajectoryPath:
        """
        核心接口：接收最新状态，放入滑动窗口处理防抖，并输出预测轨迹
        """
        # 1. 维护滑动窗口
        self.history_window.append(current_state)
        if len(self.history_window) > self.max_history:
            self.history_window.pop(0)
            
        # 2. 待实现：速度防抖与滤波逻辑 (解决 n/8 跳变问题)
        # filtered_speed = self._apply_speed_filter(self.history_window)
        
        # 3. 待实现：调用底层动力学模型计算轨迹
        # path = kinematics.calculate(..., filtered_speed, ship_data)
        
        # 暂时返回一个空轨迹作为占位
        return TrajectoryPath(base_timestamp=current_state.timestamp, is_fallback=True, points=[])