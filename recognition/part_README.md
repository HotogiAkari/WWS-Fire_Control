可以。按照你现在的整体架构，建议把这一部分的开发规范明确成 **「模块独立运行 → `vision_manager` 统一调用 → 标准化返回结果 → 后续统一写入 `data`」**。这样以后增加小地图、锁定框、炮弹落点等功能时，不需要改动已有识别模块。

下面我分成两部分：

1. **当前识别代码所有需要调参的变量及含义**
2. **可以直接放进 README 的 `Recognition` 开发手册**

---

# 一、Recognition 调参变量总表

建议以后把所有调参变量集中放在模块最上方的 `CONFIG` / 配置区域，不要把数字散落在函数内部。

## 1. OCR / 图片预处理参数

### `OCR_SCALE`

**含义：**
送入 OCR 之前对截取区域进行放大的倍率。

例如：

```python
OCR_SCALE = 3
```

表示：

```text
原始 ROI
   ↓
放大 3 倍
   ↓
图片处理
   ↓
OCR
```

**调大：**

* 小字体更加容易识别
* OCR 对细小数字更加敏感
* CPU/GPU 开销增加
* 处理时间增加
* 过大可能放大噪声

**调小：**

* 更快
* 但小字体可能无法识别

---

### `OCR_THRESHOLD`

**含义：**
二值化时使用的阈值。

典型形式：

```python
cv2.threshold(gray, OCR_THRESHOLD, 255, cv2.THRESH_BINARY)
```

例如：

```python
OCR_THRESHOLD = 150
```

意味着灰度值高于 150 的像素变成白色，否则变成黑色。

**调大：**

* 更多区域会被判定为黑色
* 可能丢失较暗的文字

**调小：**

* 更多区域会被判定为白色
* 可能保留背景噪声

---

### `OCR_INVERT`

**含义：**
是否反转二值化结果。

例如：

```python
OCR_INVERT = True
```

最终：

```text
黑字白底
```

或者：

```text
白字黑底
```

需要根据 OCR 对当前文字区域的要求决定。

---

### `OCR_DENOISE_KERNEL`

**含义：**
图片去噪/形态学处理的核大小。

例如：

```python
OCR_DENOISE_KERNEL = 2
```

用于去除一些非常小的杂点。

**调大：**

* 去噪更强
* 但容易把小数字笔画一起消掉

**调小：**

* 保留更多文字细节
* 但可能留下 UI 噪声

---

## 2. 各识别区域 ROI 参数

这部分实际上是**最重要的调参区域**。

每一个识别项目都应该有自己的：

```text
x
y
width
height
```

或者：

```text
x1
y1
x2
y2
```

例如：

```python
RELATIVE_ANGLE_ROI = (x, y, width, height)
```

其中：

* `x`：ROI 左上角 X
* `y`：ROI 左上角 Y
* `width`：截取宽度
* `height`：截取高度

---

### `x / x1`

控制识别框**左右移动**。

```text
← 减小       增大 →
```

如果识别区域左边切多了：

```text
减小 width
```

如果整个区域需要向右移动：

```text
增大 x
```

---

### `y / y1`

控制识别框**上下移动**。

```text
↑ 减小

↓ 增大
```

---

### `width`

控制识别框的**水平尺寸**。

如果你之前说：

> 相对角度截取大小无法完全去除数字周围的圆圈

那么首先应该调整的就是：

```python
RELATIVE_ANGLE_ROI_WIDTH
```

**不要优先修改 OCR 参数。**

因为圆圈属于 ROI 本身截取范围过大造成的背景干扰。

---

### `height`

控制识别框的**垂直尺寸**。

如果上下有多余 UI：

```text
减小 height
```

如果数字顶部/底部被切掉：

```text
增大 height
```

---

# 3. UI 禁区参数

你现在的结构建议固定为：

```python
EXCLUSION_ZONES = [
    ...
]
```

而不是：

```python
ZONE1 = ...
ZONE2 = ...
ZONE3 = ...
```

这样以后可以直接：

```python
EXCLUSION_ZONES.append(...)
```

增加禁区。

---

## `EXCLUSION_ZONES`

**含义：**

屏幕中明确不能被识别为目标的区域。

目前包括：

1. 左侧成员板
2. 右侧成员板
3. 顶部计分条
4. 左下角罗盘
5. 底部装备栏
6. 右上角成就

你之前提到的小地图也应该作为：

7. 右下角小地图

但因为你现在准备把小地图交给独立模块，所以这里有两种设计：

### 方案 A：Recognition 中直接排除小地图

```python
EXCLUSION_ZONES = [
    LEFT_MEMBER_PANEL,
    RIGHT_MEMBER_PANEL,
    SCORE_BAR,
    COMPASS,
    EQUIPMENT_BAR,
    ACHIEVEMENT,
    MINIMAP,
]
```

### 方案 B：把禁区交给统一视觉框架

我更推荐 **B**。

即：

```text
vision_manager
       │
       ├── 全局 UI exclusion zones
       │
       ├── ship_recognition
       ├── angle_recognition
       ├── lock_recognition
       └── minimap_recognition
```

这样以后小地图模块自己需要完整访问小地图区域时，可以明确声明：

```python
requires_excluded_zone = MINIMAP
```

而不是让模块之间互相硬编码。

---

# 4. 禁区坐标

每个禁区本质上还是：

```python
(x, y, width, height)
```

例如：

```python
{
    "name": "score_bar",
    "x": 0,
    "y": 0,
    "width": 1920,
    "height": 100,
}
```

### `x`

禁区左边界。

### `y`

禁区上边界。

### `width`

禁区宽度。

### `height`

禁区高度。

---

# 5. 检测框参数

如果某个识别模块需要先找：

```text
血条
锁定框
船名框
角度框
```

通常会存在：

```python
MIN_WIDTH
MAX_WIDTH
MIN_HEIGHT
MAX_HEIGHT
```

以及：

```python
MIN_AREA
MAX_AREA
```

---

### `MIN_WIDTH`

允许检测目标的最小宽度。

太小：

```text
噪声
```

一般会被过滤。

---

### `MAX_WIDTH`

允许检测目标的最大宽度。

防止把大型 UI 当成目标。

---

### `MIN_HEIGHT`

允许检测目标的最小高度。

---

### `MAX_HEIGHT`

允许检测目标的最大高度。

---

### `MIN_AREA`

最小轮廓面积。

---

### `MAX_AREA`

最大轮廓面积。

---

# 6. 血条 / 锁定框颜色参数

如果通过 HSV / RGB 颜色识别目标，那么需要调整：

```python
LOWER_COLOR
UPPER_COLOR
```

例如：

```python
LOWER_HSV = (...)
UPPER_HSV = (...)
```

分别代表 HSV 空间中的：

```text
H：Hue 色相
S：Saturation 饱和度
V：Value 明度
```

---

## 为什么黄色角度会出现黑底白字？

你之前观察到：

> 敌方角度是黄底白字，有时候处理后变成黑底白字。

这个现象非常合理。

**不一定是“黄色太亮”这么简单。**

更准确地说，是你的预处理过程实际上在判断：

```text
这个像素应该属于背景还是文字？
```

而黄色半透明 UI 和后面的场景颜色发生了混合：

```text
最终像素
=
黄色 UI × alpha
+
背景 × (1-alpha)
```

所以黄色区域的实际 RGB/灰度值会随着背景变化。

例如同一个黄色框：

```text
背景：天空
→ 混合后比较亮

背景：海面
→ 混合后比较暗

背景：建筑
→ 又是另一种值
```

然后你再做：

```text
RGB → Gray → Threshold
```

就可能出现：

```text
黄色背景 → 被判定成黑色
白色文字 → 被判定成白色
```

于是 OCR 看到：

```text
黑底白字
```

而我方灰色背景的亮度变化相对较小，所以更稳定。

因此，**这并不简单等于“黄色太亮”。**

真正的问题是：

> 半透明黄色背景经过场景混合后，其灰度值可能跨越你的二值化阈值。

这也是为什么：

> 天空作为背景时其它区域仍然可以正常处理

并不矛盾。

---

# 二、README：Recognition 开发手册

下面这一部分可以直接放进你的 README。

---

# Recognition 开发手册

## 1. 模块定位

Recognition 是整个视觉系统中的**功能识别层**。

项目采用模块化设计：

```text
                 ┌────────────────────┐
                 │   vision_manager   │
                 │    视觉总管理器     │
                 └─────────┬──────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
      ship_recognition  angle_recognition  minimap
             │             │             │
             ▼             ▼             ▼
           船舶信息       角度信息        小地图信息
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                         data
```

每个 Recognition 模块只负责一个明确的视觉功能。

例如：

```text
ship_recognition
    ↓
识别船名、最大航速等

angle_recognition
    ↓
识别敌我相对角度

lock_recognition
    ↓
识别锁定目标

minimap_recognition
    ↓
识别小地图中的所有目标和方向
```

---

# 2. 核心设计原则

每一个识别模块必须满足：

> **独立、可测试、可调用、可扩展。**

也就是说：

```python
python ship_recognition.py
```

应该可以单独运行。

同时：

```python
vision_manager.py
```

又可以直接调用：

```python
result = ship_recognition.run(frame)
```

而不需要模拟整个程序环境。

---

# 3. 每个模块必须提供统一入口

推荐所有 Recognition 模块都提供：

```python
def run(frame, config=None):
    ...
    return result
```

其中：

### `frame`

当前游戏画面。

通常为：

```python
numpy.ndarray
```

格式为 OpenCV 图像：

```text
BGR
```

---

### `config`

模块自己的配置。

例如：

```python
def run(frame, config=None):
    scale = config.get("scale", 3)
```

如果不需要动态配置，可以使用模块内部默认配置。

---

# 4. 返回值规范

模块不得直接修改其它模块的数据。

例如：

**不要这样：**

```python
data["enemy_angle"] = angle
```

Recognition 模块只负责：

```python
return angle
```

或者：

```python
return {
    "enemy_angle": angle
}
```

最终由：

```text
vision_manager
```

统一整理。

---

# 5. 推荐返回结构

简单识别：

```python
return {
    "value": value,
    "confidence": confidence
}
```

例如：

```python
{
    "value": 32.5,
    "confidence": 0.94
}
```

多值识别：

```python
return {
    "ship_name": "Yamato",
    "max_speed": 27.0,
    "confidence": {
        "ship_name": 0.98,
        "max_speed": 0.95
    }
}
```

---

# 6. 失败时必须返回明确结果

不要：

```python
return None
```

然后让 `vision_manager` 猜发生了什么。

推荐：

```python
return {
    "value": None,
    "confidence": 0.0
}
```

或者：

```python
return {
    "success": False,
    "value": None,
    "confidence": 0.0
}
```

这样可以区分：

```text
成功识别
```

和：

```text
没有识别到
```

---

# 7. Recognition 模块内部结构

推荐：

```text
recognition/
│
├── ship_recognition.py
├── angle_recognition.py
├── lock_recognition.py
├── minimap_recognition.py
│
└── ...
```

一个模块内部推荐按照以下结构：

```python
# =========================
# Configuration
# =========================

CONFIG = {
    ...
}


# =========================
# Image Processing
# =========================

def preprocess(...):
    ...


# =========================
# Detection
# =========================

def detect(...):
    ...


# =========================
# OCR
# =========================

def recognize(...):
    ...


# =========================
# Public API
# =========================

def run(frame, config=None):
    ...


# =========================
# Standalone Test
# =========================

if __name__ == "__main__":
    ...
```

---

# 8. 只有 `run()` 属于公共接口

其它函数：

```python
preprocess()
detect()
recognize()
```

均属于模块内部实现。

外部程序不应该依赖这些函数。

例如：

```python
result = ship_recognition.run(frame)
```

而不是：

```python
roi = ship_recognition.preprocess(frame)
result = ship_recognition.recognize(roi)
```

这样以后修改模块内部算法时：

```text
preprocess()
detect()
recognize()
```

可以随意重构。

只要：

```python
run()
```

的接口没有变化，`vision_manager` 就不需要修改。

---

# 9. 新建 Recognition 模块的流程

假设现在需要添加：

```text
炮弹落点识别
```

首先建立：

```text
recognition/shell_impact.py
```

然后：

```python
CONFIG = {
    ...
}
```

实现内部处理：

```python
def preprocess(frame, config):
    ...


def detect(frame, config):
    ...


def recognize(frame, config):
    ...
```

最后提供：

```python
def run(frame, config=None):
    ...
    return result
```

---

# 10. 独立测试

开发模块时，首先应该保证：

```python
python shell_impact.py
```

能够正常运行。

独立测试程序可以：

```text
读取测试截图
    ↓
调用 run()
    ↓
显示处理后的 ROI
    ↓
显示检测结果
```

开发阶段不要一开始就接入：

```text
vision_manager
```

否则出现错误时，很难判断到底是：

```text
模块错误
```

还是：

```text
vision_manager 调用错误
```

---

# 11. 加入 vision_manager

模块独立测试完成后，再在：

```text
vision_manager.py
```

中注册。

推荐：

```python
from recognition import ship_recognition
from recognition import angle_recognition
from recognition import minimap_recognition
```

然后：

```python
def process_frame(frame):

    results = {}

    results["ship"] = ship_recognition.run(
        frame,
        ship_recognition.CONFIG
    )

    results["angle"] = angle_recognition.run(
        frame,
        angle_recognition.CONFIG
    )

    results["minimap"] = minimap_recognition.run(
        frame,
        minimap_recognition.CONFIG
    )

    return results
```

最终：

```python
results = process_frame(frame)
```

得到：

```python
{
    "ship": {...},
    "angle": {...},
    "minimap": {...}
}
```

---

# 12. 新功能的最低接入要求

以后增加一个新功能，只需要完成：

```text
① 创建模块
② 实现 CONFIG
③ 实现 run()
④ 独立测试
⑤ 在 vision_manager 注册
```

也就是说：

```text
new_feature.py
```

至少需要：

```python
CONFIG = {}


def run(frame, config=None):
    ...
    return result
```

然后：

```python
results["new_feature"] = new_feature.run(
    frame,
    new_feature.CONFIG
)
```

即可。

---

# 13. 禁止模块之间直接调用

例如：

```text
ship_recognition
        ↓
angle_recognition
```

这种设计应该避免。

正确结构应该是：

```text
             vision_manager
              /     |     \
             ↓      ↓      ↓
           ship   angle   minimap
```

如果两个模块需要共享数据：

```text
module A
    ↓
vision_manager
    ↓
module B
```

而不是：

```text
module A → module B
```

这样可以避免模块之间形成复杂依赖。

---

# 14. 图片处理规范

所有图像处理都应尽量局限于模块自己的 ROI。

不要无理由修改完整游戏画面：

```python
frame[:] = ...
```

推荐：

```python
roi = frame[y:y+h, x:x+w]
processed = preprocess(roi)
```

这样不同模块之间不会互相污染输入。

---

# 15. OCR 模块规范

OCR 前的处理必须尽可能独立。

推荐：

```text
原始 frame
   ↓
ROI
   ↓
放大
   ↓
灰度化
   ↓
二值化
   ↓
形态学处理
   ↓
OCR
   ↓
结果
```

调参变量应该集中在：

```python
CONFIG
```

例如：

```python
CONFIG = {
    "scale": 3,
    "threshold": 150,
    "invert": False,
}
```

而不是：

```python
resize(..., 3)
threshold(..., 150)
```

把参数直接写死在代码中。

---

# 16. Debug 输出规范

开发阶段可以提供：

```python
debug=False
```

例如：

```python
def run(frame, config=None, debug=False):
```

当：

```python
debug=True
```

时，可以输出：

```text
原始 ROI
处理后 ROI
检测框
OCR 输入图片
OCR 输出
```

正式运行：

```python
debug=False
```

避免影响性能。

---

# 17. 不要在 Recognition 模块中保存状态

原则上：

```python
result = run(frame)
```

应该只由：

```text
当前 frame
+
当前 config
```

决定。

不要依赖上一次识别：

```python
previous_angle
previous_ship
previous_frame
```

除非该模块明确需要时序信息。

如果确实需要，例如：

```text
目标跟踪
```

应该显式建立：

```python
Tracker
```

而不是偷偷使用全局变量。

---

# 18. Minimap 模块特殊规范

小地图与普通 Recognition 不同。

小地图本身建立自己的坐标系：

```text
             Y
             ↑
             │
             │
       ┌─────┼─────┐
       │     │     │
       │     │     │
       │─────O─────→ X
       │           │
       └───────────┘
```

小地图模块内部负责：

```text
① 建立小地图坐标系
② 识别我舰
③ 识别锁定敌舰
④ 识别我方视角方向
⑤ 识别敌方行驶方向
⑥ 识别炮弹落点
⑦ 转换敌我二维坐标
```

最终向 `vision_manager` 提供统一结果。

例如：

```python
{
    "player": {
        "x": ...,
        "y": ...,
        "heading": ...
    },

    "enemies": [
        {
            "x": ...,
            "y": ...,
            "heading": ...,
            "locked": True
        }
    ],

    "shell_impacts": [
        {
            "x": ...,
            "y": ...
        }
    ]
}
```

**小地图内部可以使用自己的坐标系统，但不得把像素坐标直接当成游戏坐标输出。**

---

# 19. vision_manager 的职责

`vision_manager` 是**调度器，而不是识别器**。

它主要负责：

```text
读取图像
    ↓
调用 Recognition
    ↓
收集结果
    ↓
统一结果结构
    ↓
输出 data
```

不应该把大量：

```python
cv2.threshold()
cv2.findContours()
OCR()
```

直接写进 `vision_manager`。

否则最终会重新变成一个巨大的单体程序。

---

# 20. 最终推荐架构

最终项目可以逐渐发展成：

```text
vision/
│
├── vision_manager.py
│
├── recognition/
│   │
│   ├── ship/
│   │   ├── __init__.py
│   │   └── ship_recognition.py
│   │
│   ├── angle/
│   │   ├── __init__.py
│   │   └── angle_recognition.py
│   │
│   ├── lock/
│   │   ├── __init__.py
│   │   └── lock_recognition.py
│   │
│   └── minimap/
│       ├── __init__.py
│       └── minimap_recognition.py
│
└── data/
    └── ...
```

最终调用关系：

```text
                       vision_manager
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
           Ship             Angle          Minimap
        Recognition       Recognition     Recognition
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                            results
                              │
                              ▼
                             data
```

这样以后你想增加：

```text
炮弹飞行时间
敌舰速度
敌舰航向
锁定状态
血量
距离
目标类型
```

都只需要：

```text
新建一个 Recognition 模块
        ↓
实现 run()
        ↓
独立测试
        ↓
vision_manager 加一行调用
        ↓
加入统一 data
```

而不需要重构整个程序。

---

## 最后给你一个非常重要的建议

你现在这个项目已经开始从“一个能跑的识别脚本”向真正的**视觉系统**发展了，所以从现在开始最好严格区分这四层：

```text
┌─────────────────────────────┐
│        vision_manager       │
│          调度层              │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│        Recognition          │
│          识别层              │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Image Processing       │
│          图像处理层          │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│            data             │
│          数据层              │
└─────────────────────────────┘
```

尤其是 **`vision_manager` 不要承担算法逻辑**。它只负责“叫谁干活、把结果收回来、统一放在哪里”。

这样你后面做小地图时会非常舒服：小地图甚至可以发展成一个相对独立的二维视觉引擎，而 `vision_manager` 完全不需要知道它内部究竟用了颜色分割、轮廓检测、模板匹配还是 OCR。它只需要知道：

```python
minimap_result = minimap.run(frame)
```

然后拿结果即可。
