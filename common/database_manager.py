# 数据库管理器：负责在内存中加载和查询 static_ships.json
from typing import Optional
from common.data_models import ShipStaticData, load_static_db

class DatabaseManager:
    """静态船只数据库管理器"""
    def __init__(self, db_path: str = "data/static_ships.json"):
        # 启动时将数据库加载到内存 (字典形式，O(1) 查询速度)
        self.db: dict[str, ShipStaticData] = load_static_db(db_path)

    def get_ship_data(self, ship_id: str) -> Optional[ShipStaticData]:
        """根据识别到的船只名称，返回白板物理数据"""
        return self.db.get(ship_id)