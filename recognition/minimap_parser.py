# recognition/minimap_parser.py

from typing import Optional

import numpy as np


class MiniMapParser:
    """
    小地图识别模块。

    当前为预留接口。

    后续可以逐步实现：

        1. 小地图 ROI
        2. 玩家位置
        3. 敌舰图标
        4. 友军图标
        5. 航向
        6. 与屏幕目标建立对应关系
    """

    def __init__(
        self,
        roi=None,
    ):

        self.roi = roi

    def process(
        self,
        frame_bgr: np.ndarray,
    ) -> dict:

        if frame_bgr is None:
            return {}

        # =====================================================
        # TODO:
        #
        # 以后在这里实现：
        #
        # roi = frame_bgr[y:y+h, x:x+w]
        #
        # 玩家：
        #   ...
        #
        # 敌人：
        #   ...
        #
        # 最终返回统一数据结构
        # =====================================================

        return {

            "player": None,

            "enemies": [],

            "allies": [],

            "heading": None,
        }

    def close(self):
        pass