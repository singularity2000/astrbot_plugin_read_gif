"""纯配置解析：不写回用户配置，GIF/视频使用独立的自动档切换点。"""
import math

GIF_THRESHOLDS = (1.0, 10.0, 20.0)
VIDEO_THRESHOLDS = (5.0, 15.0, 30.0)
THRESHOLD_KEYS = ("to_9", "to_16", "to_25")


def parse_thresholds(value, defaults):
    """返回 (有效切换点, 是否回退)。缺少新配置时使用默认值。"""
    if value is None:
        return defaults, False
    try:
        raw = [value.get(key, default) for key, default in zip(THRESHOLD_KEYS, defaults)]
        if any(isinstance(item, bool) for item in raw):
            raise ValueError
        values = tuple(float(item) for item in raw)
        if not all(math.isfinite(item) and item > 0 for item in values):
            raise ValueError
        if not values[0] < values[1] < values[2]:
            raise ValueError
        return values, False
    except (AttributeError, TypeError, ValueError, OverflowError):
        return defaults, True


def parse_video_limit(value="60"):
    """空字符串不限；有限正数有效；缺失、布尔、零、负数、异常值回退 60 秒。"""
    if isinstance(value, str) and not value.strip():
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except (TypeError, ValueError, OverflowError):
        pass
    return 60.0
