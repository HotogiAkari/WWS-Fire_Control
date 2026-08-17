# [核心] 定义 TargetState, TrajectoryPath 等数据类及存取函数
import json
from dataclasses import dataclass, asdict
from typing import List, Dict

# ================= 1. 数据结构定义 (Data Classes) =================

@dataclass
class ShipStaticData:
    """静态数据库：舰船白板数据"""
    ship_id: str               # 舰船名称或ID (如 "Yamato")
    ship_class: str            # 舰种 ("BB", "CA", "DD", "CV", "SS")
    max_speed_base: float      # 最大航速 (节)
    turning_radius: float      # 转弯半径 (米)
    engine_acceleration: float # 引擎推重比/加速度常数 (决定加减速性能)

@dataclass
class TargetState:
    """动态数据库：单帧实时目标状态"""
    timestamp: float           # 截屏时间戳
    
    # 距离参数
    distance: float            # 敌舰的实际距离 (公里，来自Mod)
    aiming_distance: float     # 当前准星落点距离 (公里，来自游戏原生UI) -> 用于锁存防抖校验
    
    # 状态参数
    flight_time: float         # 炮弹飞行时间 (秒)
    relative_angle: float      # 敌舰相对我方的角度 (度)
    
    # 小地图坐标 (后续用于AI轨迹辅助)
    minimap_x: float           # 小地图绝对/相对 X 坐标
    minimap_y: float           # 小地图绝对/相对 Y 坐标
    
    # 动力参数
    speed_fraction: float      # 当前航速比 (0.0 到 1.0，步长 1/8) -> 由进度条UI提取
    direction_state: int       # 航向状态: 1(前进), 0(停船), -1(倒车) -> 由指示灯提取

@dataclass
class TrajectoryPoint:
    """轨迹点：用于构建未来路径的单个节点"""
    delta_time: float          # 相对 base_timestamp 的未来时间 (秒)
    dx: float                  # 相对当前时刻的横向物理位移 (米)
    dy: float                  # 相对当前时刻的纵深物理位移 (米)

@dataclass
class TrajectoryPath:
    """敌方路径数据：预测模块产出的核心结果"""
    base_timestamp: float            # 路径生成的基准时间戳
    is_fallback: bool                # 是否为保底直线预测 (True表示预测模块降级或数据不足)
    points: List[TrajectoryPoint]    # 预测的离散轨迹点集合


# ================= 2. 静态数据库存取函数 =================

def save_static_db(filepath: str, db: Dict[str, ShipStaticData]) -> None:
    """将静态数据库保存为 JSON"""
    serializable_db = {k: asdict(v) for k, v in db.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_db, f, indent=4, ensure_ascii=False)

def load_static_db(filepath: str) -> Dict[str, ShipStaticData]:
    """从 JSON 加载静态数据库"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_db = json.load(f)
    return {k: ShipStaticData(**v) for k, v in raw_db.items()}


# ================= 3. 动态状态历史存取函数 (用于回放/AI训练) =================

def save_target_history(filepath: str, history: List[TargetState]) -> None:
    """将连续的时序数据保存为 JSON，用于后期训练 AI"""
    serializable_data = [asdict(state) for state in history]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=4)

def load_target_history(filepath: str) -> List[TargetState]:
    """加载历史时序数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_list = json.load(f)
    return [TargetState(**item) for item in raw_list]


# ================= 4. 轨迹数据存取 (辅助调试用) =================

def save_trajectory(filepath: str, trajectory: TrajectoryPath) -> None:
    """保存生成的轨迹 (调试验证时使用)"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(asdict(trajectory), f, indent=4)

def load_trajectory(filepath: str) -> TrajectoryPath:
    """读取轨迹文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 将字典形式的 points 还原为 TrajectoryPoint 实例
    points = [TrajectoryPoint(**p) for p in data.pop('points', [])]
    return TrajectoryPath(**data, points=points)