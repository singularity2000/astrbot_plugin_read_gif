"""只读原视频，通过已有 FFmpeg/ffprobe 抽帧和导出音轨。

无原视频副本、无帧文件；子进程不经过 shell，超时会杀死并回收。
只持久化宫格、音轨和小型缓存索引，写入使用原子替换。
"""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave

from PIL import Image, ImageDraw, ImageFont

from .gif_processor import GifProcessor
from .media_config import VIDEO_THRESHOLDS


class VideoProcessingError(Exception):
    """可向用户解释、不会阻止 GIF 处理的视频错误。"""


class VideoProcessor:
    CACHE_VERSION = 1
    TIMEOUT = 120
    MAX_SOURCE_BYTES = 512 * 1024 * 1024
    MAX_AUDIO_BYTES = 32 * 1024 * 1024
    SAMPLE_RATE = 24000
    FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,flv,mpeg,mpegts,ogg,asf,rm"
    # 视频处理串行，限制多个会话同时解码；锁在工作线程内持有，取消不会提前释放。
    def __init__(self):
        self._lock = threading.Lock()
        self._pending_jobs = 0
        self._cancel = threading.Event()

    @property
    def busy(self) -> bool:
        return self._pending_jobs > 0 or self._lock.locked()

    @staticmethod
    def tools() -> tuple[str, str]:
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise VideoProcessingError("未找到 FFmpeg/ffprobe；请将二者加入 AstrBot 进程的 PATH，GIF 功能不受影响")
        return ffmpeg, ffprobe

    def _run(self, args, deadline):
        if deadline <= time.monotonic():
            raise VideoProcessingError("视频处理超时，已跳过")
        if self._cancel.is_set():
            raise VideoProcessingError("视频处理已取消")
        try:
            with subprocess.Popen(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ) as process:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or self._cancel.is_set():
                        process.kill()
                        process.communicate()
                        reason = "视频处理已取消" if self._cancel.is_set() else "视频处理超时，已停止解码"
                        raise VideoProcessingError(reason)
                    try:
                        stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if process.returncode:
                    # 不把路径、签名 URL 或不受信任的元数据写进提示词。
                    raise VideoProcessingError("视频解码失败或格式不受支持")
                return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        except OSError as exc:
            raise VideoProcessingError("无法启动 FFmpeg/ffprobe") from exc

    @staticmethod
    def _number(value, default=0.0):
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _atomic_json(path, value):
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    async def process_video(
        self, source: str, grid_preset: str, cache_dir: str,
        max_output_size: int = 1600,
        thresholds: tuple[float, float, float] = VIDEO_THRESHOLDS,
        duration_limit: float | None = 60.0, include_audio: bool = True,
    ) -> tuple[str, dict]:
        cancel = threading.Event()
        self._pending_jobs += 1
        try:
            return await asyncio.to_thread(
                self._process, source, grid_preset, cache_dir, max_output_size,
                thresholds, duration_limit, include_audio, cancel,
            )
        except asyncio.CancelledError:
            cancel.set()
            raise
        finally:
            self._pending_jobs -= 1

    def _process(self, source, preset, cache_dir, max_size, thresholds, limit, audio, cancel):
        with self._lock:
            self._cancel = cancel
            if cancel.is_set():
                raise VideoProcessingError("视频处理已取消")
            deadline = time.monotonic() + self.TIMEOUT
            ffmpeg, ffprobe = self.tools()
            source = str(Path(source).resolve())
            if not Path(source).is_file():
                raise VideoProcessingError("视频文件不存在或不可读")
            if Path(source).stat().st_size > self.MAX_SOURCE_BYTES:
                raise VideoProcessingError("视频文件超过 512 MiB 安全限制，已跳过")
            # 防止本地播放列表/媒体引用让解码器再次访问网络。
            probe = self._run([
                ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", self.FORMATS,
                "-show_entries",
                "format=duration,start_time:stream=index,codec_type,width,height,duration,start_time,nb_frames,avg_frame_rate:stream_disposition=attached_pic:stream_tags=DURATION",
                "-of", "json", source,
            ], deadline)
            try:
                meta = json.loads(probe.stdout)
                streams = meta.get("streams", [])
                video = next(s for s in streams if s.get("codec_type") == "video"
                             and not s.get("disposition", {}).get("attached_pic"))
            except (ValueError, StopIteration, TypeError) as exc:
                raise VideoProcessingError("文件中没有可读取的视频画面") from exc
            duration = self._number(video.get("duration"))
            if duration <= 0:
                # Matroska 常把单轨时长放在标签中，不能让更长的音轨冒充画面时长。
                tag = video.get("tags", {}).get("DURATION", "")
                try:
                    hours, minutes, seconds = (float(part) for part in tag.split(":"))
                    tagged = hours * 3600 + minutes * 60 + seconds
                    duration = tagged if math.isfinite(tagged) and tagged > 0 else 0
                except (TypeError, ValueError):
                    duration = 0
            duration = duration or self._number(meta.get("format", {}).get("duration"))
            if duration <= 0:
                raise VideoProcessingError("无法确定视频时长，已跳过")
            if limit is not None and duration > limit:
                raise VideoProcessingError(f"视频时长 {duration:.3f} 秒，超过 {limit:g} 秒上限，未分析该视频")
            width, height = int(video.get("width", 0)), int(video.get("height", 0))
            if width <= 0 or height <= 0 or width * height > 3840 * 2160 * 2:
                raise VideoProcessingError("视频分辨率无效或超过安全解码范围")
            frame_count = int(self._number(video.get("nb_frames"))) or 1000000
            grid = GifProcessor.parse_grid_preset(preset, duration, frame_count, thresholds)
            # 视频固定档也不重复填帧；少于目标帧数时缩小为完全平方数。
            grid = min(grid, max(1, math.isqrt(frame_count) ** 2))
            max_size = int(max_size or 1600)
            max_size = min(4096, max(128, max_size if max_size > 0 else 1600))
            cache = Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with open(source, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    if time.monotonic() >= deadline or cancel.is_set():
                        raise VideoProcessingError("视频读取超时或已取消")
                    digest.update(chunk)
            digest = digest.hexdigest()[:16]
            media_id = "V-" + digest[:12]
            key = f"video_grid_{digest}_{grid}_{max_size}_v{self.CACHE_VERSION}"
            index = cache / (key + ".json")
            output = cache / (key + ".png")
            info = None
            if output.is_file() and index.is_file():
                try:
                    info = json.loads(index.read_text(encoding="utf-8"))
                    if not isinstance(info.get("times"), list) or not info["times"]:
                        info = None
                except (OSError, ValueError, TypeError, AttributeError):
                    info = None
            if info is None:
                times, actual_grid = self._render(
                    ffmpeg, source, video, duration, grid, max_size,
                    media_id, output, deadline,
                )
                info = {"media_id": media_id, "duration_s": duration,
                        "times": times, "grid_size": actual_grid}
                self._atomic_json(index, info)
            info = dict(info)
            info["output_path"] = str(output)
            info["audio_path"] = None
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
            if audio_stream is None:
                info["audio_status"] = "no_track"
            elif not audio:
                info["audio_status"] = "disabled"
            else:
                try:
                    info["audio_path"], info["audio_status"] = self._audio(
                        ffmpeg, source, video, audio_stream, duration, cache,
                        digest, deadline,
                    )
                except VideoProcessingError as exc:
                    info["audio_status"] = "failed"
                    info["audio_error"] = str(exc)
            # 按最后一次使用时间计缓存保留期。
            for path in (output, index, Path(info["audio_path"]) if info["audio_path"] else None):
                if path is not None and path.exists():
                    path.touch()
            return str(output), info

    def _render(self, ffmpeg, source, video, duration, grid, size, media_id, output, deadline):
        side = math.isqrt(grid)
        cell = max(1, (size - 32) // side)
        font_size = max(10, min(24, cell // 12))
        label_height = min(cell - 1, font_size + 8)
        image_height = cell - label_height
        # 覆盖首帧至末尾附近，不请求恰好 EOF；以帧时间戳采样而非帧序号。
        try:
            numerator, denominator = video.get("avg_frame_rate", "0/1").split("/")
            fps = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            fps = 0
        end = max(0.0, duration - (1 / fps if fps > 0 else min(0.1, duration / 100)))
        targets = [end * i / (grid - 1) for i in range(grid)] if grid > 1 else [0.0]
        select = "+".join(f"gte(t,{t:.9f})*eq(selected_n,{i})" for i, t in enumerate(targets))
        filters = (
            f"setpts=PTS-STARTPTS,select='{select}',showinfo,"
            f"scale=w='max(1,if(gte(dar,{cell/image_height}),{cell},trunc({image_height}*dar)))':"
            f"h='max(1,if(gte(dar,{cell/image_height}),trunc({cell}/dar),{image_height}))',"
            f"setsar=1,pad={cell}:{image_height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        result = self._run([
            ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-threads", "2",
            "-protocol_whitelist", "file,pipe", "-format_whitelist", self.FORMATS, "-i", source, "-map", f"0:{video['index']}",
            "-an", "-sn", "-dn", "-vf", filters, "-frames:v", str(grid),
            "-vsync", "0", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ], deadline)
        times = [float(t) for t in re.findall(
            r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:\s*([-+\d.eE]+)",
            result.stderr.decode("utf-8", errors="replace"),
        )]
        frame_bytes = cell * image_height * 3
        count = min(len(times), len(result.stdout) // frame_bytes, grid)
        if not count:
            raise VideoProcessingError("未能抽取视频画面")
        actual = math.isqrt(count) ** 2
        chosen = [round(i * (count - 1) / (actual - 1)) for i in range(actual)] if actual > 1 else [0]
        side = math.isqrt(actual)
        canvas = Image.new("RGB", (side * cell, side * cell + 32), "black")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), f"{media_id} | {duration:.3f}s", fill="white", font=ImageFont.load_default(size=18))
        font = ImageFont.load_default(size=font_size)
        for position, i in enumerate(chosen):
            frame = Image.frombytes("RGB", (cell, image_height), result.stdout[i*frame_bytes:(i+1)*frame_bytes])
            x, y = (position % side) * cell, 32 + (position // side) * cell
            canvas.paste(frame, (x, y + label_height))
            draw.text((x + 3, y + 2), f"{times[i]:.3f}s", fill="white", font=font)
        temp = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            canvas.save(temp, "PNG", optimize=True)
            os.replace(temp, output)
        finally:
            temp.unlink(missing_ok=True)
        return [times[i] for i in chosen], actual

    def _audio(self, ffmpeg, source, video, audio, duration, cache, digest, deadline):
        stem = f"video_audio_{digest}_v{self.CACHE_VERSION}"
        output, silent = cache / (stem + ".wav"), cache / (stem + ".silent")
        if silent.is_file():
            silent.touch()
            return None, "silent"
        if output.is_file():
            return str(output), "ready"
        # WAV: 24 kHz、双声道、16-bit。保留左右声道，避免相位相反的声音被混成静音。
        if duration * self.SAMPLE_RATE * 2 * 2 + 4096 > self.MAX_AUDIO_BYTES:
            raise VideoProcessingError("音轨预计超过 32 MiB 安全限制，仅分析画面")
        offset = self._number(audio.get("start_time")) - self._number(video.get("start_time"))
        filters = ["asetpts=PTS-STARTPTS"]
        if offset < 0:
            filters += [f"atrim=start={-offset:.9f}", "asetpts=PTS-STARTPTS"]
        elif offset > 0:
            filters += [f"adelay={offset * 1000:.6f}:all=1"]
        filters += ["apad", f"atrim=duration={duration:.9f}", "astats=metadata=0:reset=0"]
        temp = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            result = self._run([
                ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-threads", "2",
                "-protocol_whitelist", "file,pipe", "-format_whitelist", self.FORMATS, "-i", source, "-map", f"0:{audio['index']}",
                "-vn", "-sn", "-dn", "-af", ",".join(filters), "-t", f"{duration:.9f}",
                "-ar", str(self.SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le",
                "-fs", str(self.MAX_AUDIO_BYTES), "-f", "wav", str(temp),
            ], deadline)
            peaks = re.findall(r"Peak level dB:\s*(\S+)", result.stderr.decode("utf-8", errors="replace"))
            # astats 在重采样/量化前检查整段全部声道；极轻的非零声不被当作完全静音。
            if not peaks:
                raise VideoProcessingError("无法确认音轨状态，仅分析画面")
            if peaks[-1].lower() == "-inf":
                self._atomic_json(silent, {"silent": True})
                return None, "silent"
            if not math.isfinite(self._number(peaks[-1], math.nan)):
                raise VideoProcessingError("音轨包含无效采样值，仅分析画面")
            with wave.open(str(temp), "rb") as wav:
                actual_duration = wav.getnframes() / wav.getframerate()
            if temp.stat().st_size >= self.MAX_AUDIO_BYTES or actual_duration + 0.1 < duration:
                raise VideoProcessingError("音轨未完整导出，仅分析画面")
            os.replace(temp, output)
            return str(output), "ready"
        except (OSError, wave.Error) as exc:
            raise VideoProcessingError("音轨导出失败，仅分析画面") from exc
        finally:
            temp.unlink(missing_ok=True)
