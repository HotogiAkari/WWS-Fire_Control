# [模块接口] 接收 TrajectoryPath 和实时参数，刷新屏幕显示
from common.data_models import TargetState, TrajectoryPath
from common.config_loader import ConfigLoader

class DisplayManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.screen_width = self.config.get_param("screen_width", 1920)
        self.screen_height = self.config.get_param("screen_height", 1080)
        
        # 内部状态锁存
        self.cached_scale_factor = 1.0 # 冻结的准星比例尺 K 值
        
        # 待实现：初始化透明窗口 (Overlay) 和渲染器

    def clear_overlay(self):
        """清空屏幕上的所有绘制内容 (如：未锁定状态时)"""
        pass

    def update_display(self, path: TrajectoryPath, current_state: TargetState):
        """
        核心接口：接收轨迹和当前UI状态，解算瞄点并渲染
        """
        # 1. 待实现：校验 current_state.aiming_distance 与 distance
        #    决定是更新 cached_scale_factor 还是继续冻结
        
        # 2. 待实现：根据 current_state.flight_time，从 path 中采样物理位移 (dx, dy)
        
        # 3. 待实现：将物理位移映射为屏幕像素 (X, Y)
        
        # 4. 待实现：在 Overlay 窗口上调用渲染器，平滑移动到 (X, Y) 绘制预瞄圈
        pass