# 配置加载器：解析 config.yaml
from typing import Dict, Any
import yaml

class ConfigLoader:
    """全局配置加载器，供所有模块读取设置"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        # 实际开发时用 yaml.safe_load 读取，这里返回空字典作为骨架
        return {}

    def get_roi(self, key: str) -> tuple[int, int, int, int]:
        """获取感兴趣区域的坐标 (x1, y1, x2, y2)"""
        return self.config_data.get("ROIs", {}).get(key, (0, 0, 0, 0))
        
    def get_param(self, key: str, default: Any = None) -> Any:
        """获取系统通用参数 (如防抖帧数、颜色阈值等)"""
        return self.config_data.get("Params", {}).get(key, default)