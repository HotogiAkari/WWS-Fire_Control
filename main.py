import time
from common.config_loader import ConfigLoader
from common.database_manager import DatabaseManager
from recognition.vision_manager import VisionManager
from prediction.prediction_manager import PredictionManager
from display.display_manager import DisplayManager

def main():
    # 1. 初始化各模块
    config = ConfigLoader("config.yaml")
    db = DatabaseManager("data/static_ships.json")
    
    vision = VisionManager(config)
    predictor = PredictionManager(config)
    display = DisplayManager(config)

    print("系统初始化完成，进入主循环...")

    # 2. 主循环
    while True:
        start_time = time.time()

        # [步骤A] 获取画面并解析
        frame = vision.capture_screen()
        vision_result = vision.process_frame(frame)

        if not vision_result:
            # 没锁定目标：清空预测历史，清空屏幕
            predictor.reset_history()
            display.clear_overlay()
        else:
            current_state, ship_id = vision_result
            
            # [步骤B] 查询静态数据库
            ship_data = db.get_ship_data(ship_id)
            if not ship_data:
                # 兼容处理：如果没有这艘船的数据，可以使用通用默认数据
                pass 
                
            # [步骤C] 轨迹预测
            path = predictor.update_and_predict(current_state, ship_data)
            
            # [步骤D] 瞄点解算与显示
            display.update_display(path, current_state)

        # 帧率控制 (例如限制在 60 FPS)
        process_time = time.time() - start_time
        time.sleep(max(0, 1/60 - process_time))

if __name__ == "__main__":
    main()