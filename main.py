import os
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.core.star.register.star_handler import (
    register_on_waiting_llm_request,
    register_on_llm_request,
    register_command,
)
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from astrbot.core.agent.message import TextPart

from .gif_processor import GifProcessor


@register("astrbot_plugin_read_gif", "Singularity", "让多模态 LLM 真正读懂 GIF", "1.0.0")
class ReadGifPlugin(Star):
    """GIF 动图预处理插件。

    在 LLM 请求准备阶段（OnWaitingLLMRequestEvent）拦截消息链中的 GIF，
    将其均匀取帧拼成宫格静态图，替换 Image 组件。
    由于此阶段早于 build_main_agent，后续流程会把宫格图当成普通图片处理，
    自然进入 req.image_urls，不会被 AstrBot 的图片压缩破坏成单帧 JPEG。
    """

    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config
        self.processor = GifProcessor()
        self._ensure_cache_dir()
        self._last_cleanup = time.time()

    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在。"""
        cache_dir = self._get_cache_dir()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _get_cache_dir(self) -> str:
        """获取插件数据缓存目录路径。

        遵循 AstrBot 规范，使用 data/plugin_data/{plugin_name}/ 作为插件专属数据目录，
        确保跨平台兼容性（Windows、Linux、Docker 等）。
        """
        base = Path(get_astrbot_data_path()) / "plugin_data" / "astrbot_plugin_read_gif"
        return str(base)

    def _get_config(self, key: str, default: Any = None) -> Any:
        """安全读取配置，兼容 AstrBotConfig 和普通 dict。"""
        if self.config is None:
            return default
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return default

    def _should_cleanup(self) -> bool:
        """判断是否应该执行自动缓存清理。"""
        interval_min = self._get_config("auto_cleanup_interval_min", 60)
        if interval_min <= 0:
            return False
        elapsed = time.time() - self._last_cleanup
        return elapsed >= interval_min * 60

    async def _do_cleanup(self, all_files: bool = False) -> int:
        """执行缓存清理，返回删除的文件数。

        - all_files=False（默认，自动清理用）：只删除超过 cache_max_age_hours 的过期文件
        - all_files=True（手动命令用）：清空全部缓存文件，无论新旧
        """
        cache_dir = self._get_cache_dir()
        if not os.path.isdir(cache_dir):
            return 0
        cutoff = time.time() - self._get_config("cache_max_age_hours", 24) * 3600
        removed = 0
        for entry in os.listdir(cache_dir):
            path = os.path.join(cache_dir, entry)
            if not os.path.isfile(path):
                continue
            if all_files or os.path.getmtime(path) < cutoff:
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        self._last_cleanup = time.time()
        return removed

    async def _maybe_cleanup(self) -> None:
        """按需触发自动缓存清理。"""
        if self._should_cleanup():
            removed = await self._do_cleanup()
            if removed > 0:
                logger.info(f"[astrbot_plugin_read_gif] 自动清理缓存完成，删除 {removed} 个文件")

    async def _try_replace_image(self, comp: Image) -> Image | None:
        """尝试把一个 Image 组件替换为宫格图。

        返回新的 Image 组件，或 None（表示不是 GIF 或处理失败，保持原样）。
        """
        try:
            image_path = await comp.convert_to_file_path()
        except Exception as exc:
            logger.debug(f"[astrbot_plugin_read_gif] 获取图片路径失败: {exc}")
            return None

        if not self.processor.is_gif(image_path):
            return None

        try:
            grid_path, info = await self.processor.process_gif(
                image_path,
                grid_preset=self._get_config("grid_preset", "auto"),
                cache_dir=self._get_cache_dir(),
                max_output_size=self._get_config("max_output_size", 1536),
            )
        except Exception as exc:
            logger.warning(f"[astrbot_plugin_read_gif] GIF 处理失败: {exc}")
            return None

        if not grid_path or not os.path.exists(grid_path):
            return None

        preset = self._get_config("grid_preset", "auto")
        logger.info(
            f"[astrbot_plugin_read_gif] GIF帧数{info['frame_count']}，"
            f"秒数{info['duration_s']:.2f}s，"
            f"已选[{preset}]，"
            f"{'智能转为' if preset == 'auto' else '转为'}{info['grid_size']}宫格"
        )
        logger.debug(f"[astrbot_plugin_read_gif] 已替换 GIF: {image_path} -> {grid_path}")
        return Image.fromFileSystem(grid_path)

    @register_on_waiting_llm_request()
    async def on_waiting_llm_request(self, event: AstrMessageEvent) -> None:
        """LLM 请求准备阶段：在 build_main_agent 之前替换 GIF。

        这是替换 GIF 的最佳时机：
        - 早于 build_main_agent，避免 GIF 被 AstrBot 图片压缩破坏成单帧 JPEG
        - 只在触发 LLM 时执行，不影响纯图片无文字消息
        - 修改 event.message_obj.message 后，build_main_agent 会把宫格图当普通图片处理
        """
        if not self._get_config("enabled", True):
            return

        await self._maybe_cleanup()

        new_message = []
        modified = False

        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                new_image = await self._try_replace_image(comp)
                if new_image is not None:
                    new_message.append(new_image)
                    modified = True
                else:
                    new_message.append(comp)

            elif isinstance(comp, Reply):
                # 递归扫描引用消息 chain 中的 Image
                if comp.chain:
                    new_chain = []
                    chain_modified = False
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            new_image = await self._try_replace_image(reply_comp)
                            if new_image is not None:
                                new_chain.append(new_image)
                                chain_modified = True
                            else:
                                new_chain.append(reply_comp)
                        else:
                            new_chain.append(reply_comp)

                    if chain_modified:
                        comp.chain = new_chain
                        modified = True

                new_message.append(comp)

            else:
                new_message.append(comp)

        if modified:
            event.message_obj.message = new_message
            # 标记本次请求处理过 GIF，供 on_llm_request 注入提示词
            event.set_extra("gif_processed", True)

    @register_on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """兜底：处理经过第三方 Agent Runner 或其他路径进入 req.image_urls 的 GIF。

        正常情况下，on_waiting_llm_request 已经把 GIF 替换成了宫格图，
        build_main_agent 处理的是宫格图，req.image_urls 中不会出现 GIF。
        但某些路径（如 ThirdPartyAgentSubStage）会直接从消息链提取 image_urls，
        这里作为最后兜底，扫描并替换任何漏网的 GIF。

        同时，当检测到 GIF 被处理时，通过 extra_user_content_parts 注入提示词，
        让 LLM 知道这是动图，但不暴露帧序列等技术细节。
        """
        if not self._get_config("enabled", True):
            return

        gif_processed_count = 0

        if req.image_urls:
            new_urls = []
            for url in req.image_urls:
                if not self.processor.is_gif(url):
                    new_urls.append(url)
                    continue
                try:
                    grid_path, info = await self.processor.process_gif(
                        url,
                        grid_preset=self._get_config("grid_preset", "auto"),
                        cache_dir=self._get_cache_dir(),
                        max_output_size=self._get_config("max_output_size", 1536),
                    )
                    if grid_path and os.path.exists(grid_path):
                        new_urls.append(grid_path)
                        gif_processed_count += 1
                        preset = self._get_config("grid_preset", "auto")
                        logger.info(
                            f"[astrbot_plugin_read_gif] GIF帧数{info['frame_count']}，"
                            f"秒数{info['duration_s']:.2f}s，"
                            f"已选[{preset}]，"
                            f"{'智能转为' if preset == 'auto' else '转为'}{info['grid_size']}宫格"
                        )
                    else:
                        new_urls.append(url)
                except Exception as exc:
                    logger.warning(f"[astrbot_plugin_read_gif] GIF 处理失败(image_urls): {exc}")
                    new_urls.append(url)
            req.image_urls = new_urls

        # 检查 on_waiting_llm_request 是否标记了 GIF 处理
        gif_flag = event.get_extra("gif_processed", False)

        if gif_processed_count > 0 or gif_flag:
            # 注入提示词，让 LLM 知道这是 GIF 动图
            # 默认值由 _conf_schema.json 提供；留空则不注入
            hint_text = self._get_config("gif_hint_text", "")
            if hint_text:
                hint_part = TextPart(text=hint_text)
                hint_part.mark_as_temp()
                req.extra_user_content_parts.append(hint_part)
                logger.debug("[astrbot_plugin_read_gif] 已注入 GIF 提示词到 extra_user_content_parts")

    @register_command("gifcache")
    async def gif_cache_cmd(self, event: AstrMessageEvent) -> None:
        """查看当前缓存目录中的宫格图列表，或发送指定缓存图。"""
        args = event.message_str.strip().split()
        cache_dir = self._get_cache_dir()

        if not os.path.isdir(cache_dir):
            yield event.plain_result("缓存目录不存在。")
            return

        files = [f for f in os.listdir(cache_dir) if f.lower().endswith(".png")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)

        if len(args) <= 1:
            # 无参数：列出缓存
            if not files:
                yield event.plain_result("当前缓存为空。")
                return
            lines = [f"缓存文件 ({len(files)} 个)："]
            for i, fname in enumerate(files[:20], 1):
                fpath = os.path.join(cache_dir, fname)
                size_kb = os.path.getsize(fpath) / 1024
                mtime = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(fpath)))
                lines.append(f"{i}. {fname} ({size_kb:.1f}KB, {mtime})")
            if len(files) > 20:
                lines.append(f"... 还有 {len(files) - 20} 个")
            yield event.plain_result("\n".join(lines))
            return

        # 有参数：尝试按索引或文件名发送
        arg = args[1]
        target = None
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(files):
                target = os.path.join(cache_dir, files[idx])
        else:
            for f in files:
                if f.startswith(arg) or arg in f:
                    target = os.path.join(cache_dir, f)
                    break

        if target and os.path.exists(target):
            # chain_result 期望 list[BaseMessageComponent]，不是 MessageChain 对象
            yield event.chain_result([
                Plain(f"缓存图：{os.path.basename(target)}"),
                Image.fromFileSystem(target),
            ])
        else:
            yield event.plain_result("未找到指定的缓存文件。")

    @register_command("gifclean")
    async def gif_clean_cmd(self, event: AstrMessageEvent) -> None:
        """手动清空缓存目录中的所有宫格图。"""
        removed = await self._do_cleanup(all_files=True)
        yield event.plain_result(f"已清理 {removed} 个缓存文件。")
