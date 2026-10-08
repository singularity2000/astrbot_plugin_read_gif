"""三项缓存限额：事件触发检查、单次到期唤醒、安全删除与有界退避。"""
import asyncio
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import stat
import time


@dataclass(frozen=True)
class CacheLimits:
    max_age_seconds: float | None = None
    max_bytes: float | None = None
    max_files: int | None = None

    @property
    def enabled(self) -> bool:
        return any(value is not None for value in (
            self.max_age_seconds, self.max_bytes, self.max_files,
        ))


def parse_cache_limits(config: dict) -> CacheLimits:
    """空值不限；任一非法上限抛出 ValueError，不将错误解释成零。"""
    if not isinstance(config, dict):
        raise ValueError("cache_management")
    values = []
    for key, scale in (("max_age_hours", 3600), ("max_size_mb", 1024 * 1024), ("max_files", 1)):
        raw = config.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            values.append(None)
            continue
        try:
            if isinstance(raw, bool):
                raise ValueError
            text = str(raw).strip()
            if key == "max_files":
                if not re.fullmatch(r"[0-9]+", text):
                    raise ValueError
                number = int(text)
            else:
                number = float(text) * scale
                if not math.isfinite(number):
                    raise ValueError
            if number <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(key) from None
        values.append(number)
    return CacheLimits(*values)


_CACHE_NAME = re.compile(
    r"(?:gif_grid_.+\.png|video_grid_.+\.(?:png|json)|"
    r"video_audio_.+\.(?:wav|silent))(?:\.[0-9a-f]{32}\.tmp)?$"
)


def _is_link(info) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _path_key(path) -> str:
    return os.path.normcase(os.path.abspath(path))


def format_size(size: int) -> str:
    return f"{size / 1024:.2f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.2f} MB"


@dataclass
class CleanupResult:
    removed: int = 0
    removed_bytes: int = 0
    skipped: int = 0
    remaining_count: int = 0
    remaining_bytes: int = 0
    next_check: float | None = None
    deferred: bool = False
    failures: set[str] = field(default_factory=set)
    remaining_paths: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)

    def wake_at(self, when: float) -> None:
        self.next_check = when if self.next_check is None else min(self.next_check, when)


def clean_cache(
    directory: str, limits: CacheLimits, *, all_files: bool = False,
    protected=(), busy_prefixes=(), retry_until=None, now: float | None = None,
) -> CleanupResult:
    """同步扫描/删除，不让出事件循环；只处理直属的已知缓存，不跟随链接。

    在用文件仍计入额度，但不以其过期时间反复唤醒。工作线程整类保护；
    被取消请求的线程可能仍在收尾，最多五分钟后兜底检查一次。
    """
    result = CleanupResult()
    if not all_files and not limits.enabled:
        return result
    root = Path(directory).absolute()
    # 也拒绝父目录 junction，避免越过预期数据目录操作其它位置。
    for parent in (root, *root.parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if _is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise OSError("缓存路径不是普通目录，已跳过清理")
    files = []
    try:
        entries = list(root.iterdir())
    except FileNotFoundError:
        return result
    for path in entries:
        if not _CACHE_NAME.fullmatch(path.name):
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode) and not _is_link(info):
            files.append((path, info))
    files.sort(key=lambda item: (item[1].st_mtime, item[0].name))
    result.remaining_count = len(files)
    result.remaining_bytes = sum(info.st_size for _, info in files)
    result.remaining_paths = {_path_key(path) for path, _ in files}
    protected = {_path_key(path) for path in protected}
    retry_until = retry_until or {}
    now = time.time() if now is None else now
    if busy_prefixes:
        result.wake_at(now + 300)  # 不是固定轮询，仅在工作线程仍活动时兜底。
    if all_files:
        result.reasons.append("手动命令")
    else:
        if limits.max_age_seconds is not None and any(now - i.st_mtime >= limits.max_age_seconds for _, i in files):
            result.reasons.append("超过保留时间")
        if limits.max_bytes is not None and result.remaining_bytes > limits.max_bytes:
            result.reasons.append("容量超限")
        if limits.max_files is not None and result.remaining_count > limits.max_files:
            result.reasons.append("数量超限")
    for path, info in files:
        key = _path_key(path)
        expiry = None if limits.max_age_seconds is None else info.st_mtime + limits.max_age_seconds
        expired = expiry is not None and now >= expiry
        over_capacity = (
            (limits.max_bytes is not None and result.remaining_bytes > limits.max_bytes)
            or (limits.max_files is not None and result.remaining_count > limits.max_files)
        )
        needed = all_files or expired or over_capacity
        if key in protected or path.name.startswith(tuple(busy_prefixes)):
            result.skipped += 1
            result.deferred |= needed
            continue
        if not needed:
            if expiry is not None:
                result.wake_at(expiry)
            continue
        if not all_files and retry_until.get(key, 0) > now:
            result.wake_at(retry_until[key])
            result.deferred = True
            continue
        try:
            current = path.lstat()
            if (_is_link(current) or not stat.S_ISREG(current.st_mode)
                    or (current.st_ino, current.st_size, current.st_mtime_ns)
                    != (info.st_ino, info.st_size, info.st_mtime_ns)):
                # 快照已失效，本轮不继续按旧总量删其它文件。
                result.failures.add(key)
                result.deferred = True
                break
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            result.failures.add(key)
            result.deferred = True
            continue
        else:
            result.removed += 1
            result.removed_bytes += info.st_size
        result.remaining_count -= 1
        result.remaining_bytes -= info.st_size
        result.remaining_paths.discard(key)
    return result


class CacheManager:
    """无固定轮询：合并事件检查，用一个定时器等待最早到期/失败重试。"""
    RETRY_MIN = 300
    RETRY_MAX = 3600

    def __init__(self, directory: str | Path, config: dict, logger, protection=None) -> None:
        self.directory = str(directory)
        self.logger = logger
        self.protection = protection or (lambda: ((), ()))
        self.limits = CacheLimits()
        self.valid = True
        self._loop = None
        self._timer = None
        self._pending = None
        self._closed = False
        self._retries = {}  # path -> (最早重试时间, 上次退避秒数)
        self._scan_retry_at = 0.0
        self._scan_retry_delay = 0
        self._warned_at = {}
        self._last_deferred_log = -math.inf
        try:
            self.limits = parse_cache_limits(config)
        except ValueError as exc:
            self.valid = False
            self._warn(f"缓存配置 {exc} 无效，自动清理暂停；请留空或填写正数，数量须为正整数。")

    @property
    def enabled(self) -> bool:
        return self.valid and self.limits.enabled

    def _warn(self, message):
        now = time.monotonic()
        if now - self._warned_at.get(message, -math.inf) >= 300:
            self.logger.warning("[astrbot_plugin_read_gif/缓存] " + message)
            self._warned_at[message] = now

    def start(self) -> None:
        if self._closed or self._loop is not None:
            return
        self._loop = asyncio.get_running_loop()
        if self.enabled:
            self.check("启动")

    def request_check(self, trigger: str = "缓存生成完成") -> None:
        if self.enabled and not self._closed and self._loop is not None and self._pending is None:
            self._pending = self._loop.call_soon(self._run_pending, trigger)

    def _run_pending(self, trigger):
        self._pending = None
        self.check(trigger)

    def _wake(self):
        self._timer = None
        self.check("到期或重试")

    def _schedule(self, when):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.enabled and not self._closed and self._loop is not None and when is not None:
            # 至少间隔一秒，避免极短保留期或时钟精度造成空转。
            self._timer = self._loop.call_later(max(1.0, when - time.time()), self._wake)

    def check(self, trigger: str = "检查", *, all_files: bool = False) -> CleanupResult:
        if self._closed or (not all_files and not self.enabled):
            return CleanupResult()
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        now = time.time()
        if not all_files and now < self._scan_retry_at:
            self._schedule(self._scan_retry_at)
            return CleanupResult(deferred=True)
        try:
            protected, busy = self.protection()
            result = clean_cache(
                self.directory, self.limits, all_files=all_files,
                protected=protected, busy_prefixes=busy,
                retry_until={key: value[0] for key, value in self._retries.items()}, now=now,
            )
        except OSError:
            self._warn("无法安全扫描缓存目录，本轮清理已跳过，稍后重试。")
            self._scan_retry_delay = min(self.RETRY_MAX, max(self.RETRY_MIN, self._scan_retry_delay * 2))
            self._scan_retry_at = now + self._scan_retry_delay
            self._schedule(self._scan_retry_at)
            return CleanupResult(deferred=True)
        self._scan_retry_delay = 0
        self._scan_retry_at = 0
        for key in list(self._retries):
            if key not in result.remaining_paths:
                del self._retries[key]
        for key in result.failures:
            delay = min(self.RETRY_MAX, max(self.RETRY_MIN, self._retries.get(key, (0, 0))[1] * 2))
            self._retries[key] = (now + delay, delay)
            result.wake_at(now + delay)
        if result.failures:
            self._warn("部分缓存无法删除或已被修改，延迟重试；不会删除正在使用的文件。")
        self._schedule(result.next_check)
        if result.reasons and (result.removed or all_files or (
            result.deferred and time.monotonic() - self._last_deferred_log >= 300
        )):
            status = "部分暂缓" if result.deferred else "清理完成"
            self.logger.info(
                f"[astrbot_plugin_read_gif/缓存] {status} | 检查={trigger} | "
                f"原因={'、'.join(result.reasons)} | 删除={result.removed}个/{format_size(result.removed_bytes)} | "
                f"在用跳过={result.skipped}个"
            )
            self._last_deferred_log = time.monotonic()
        return result

    def close(self) -> None:
        self._closed = True
        for handle in (self._timer, self._pending):
            if handle is not None:
                handle.cancel()
        self._timer = self._pending = None
