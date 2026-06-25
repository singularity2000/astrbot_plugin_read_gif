import os
import re
import time
import base64
import importlib
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

from .gif_processor import GifProcessor


@register("astrbot_plugin_read_gif", "Singularity", "让多模态 LLM 真正读懂 GIF", "1.2.0")
class ReadGifPlugin(Star):
    """GIF 动图预处理插件。

    在 LLM 请求准备阶段（OnWaitingLLMRequestEvent）拦截消息链中的 GIF，
    将其均匀取帧拼成宫格静态图，替换 Image 组件。
    由于此阶段早于 build_main_agent，后续流程会把宫格图当成普通图片处理，
    自然进入 req.image_urls，不会被 AstrBot 的图片压缩破坏成单帧 JPEG。

    GIF 提示词注入策略（按主 LLM 是否支持图片模态分流）：

    1) 主 LLM 支持图片模态（直接看宫格图）
       - 在 on_llm_request 阶段把提示词注入 req.system_prompt
       - 用标记对 GIF_HINT_MARK 包裹，注入前先清理旧痕迹（防御性）

    2) 主 LLM 不支持图片模态 + 配了图片转述模型（转述模型看图，主 LLM 看文字）
       - 框架转述图片时调用转述模型实例的 text_chat，但对"引用消息中的图片"
         走 _process_quote_message，prompt 是硬编码的，不读 image_caption_prompt；
         对"直接发送的图片"走 _request_img_caption，读 image_caption_prompt。
       - 为同时覆盖这两条路径且不依赖框架的 prompt 字面量，本插件在命中转述路径
         时临时包装转述模型实例的 text_chat：在框架真正发请求前，把 GIF 提示词
         追加到 prompt 结尾，让转述模型按"动态序列"而非"宫格"描述。
       - 匹配方式不认 prompt 文本、不认图片路径（压缩后路径为随机名），只认
         "本轮包装存活" + "本次调用带 image_urls"。同一条消息若同时含直接图与
         引用图，框架会发两次带图调用，故包装存活期内对【每次】带图调用都追加，
         本轮边界由 on_waiting 入口 / on_llm_request / terminate 的卸载界定。
       - 局限（如实标注）：实例级覆盖在并发时理论上可能影响同一转述模型的其它会话；
         以"带图判定 + 本轮即卸载"将窗口压到最小，并在多处兜底卸载。
    """

    # 提示词标记对：用于在 system_prompt 中精准定位并清理旧提示词
    _GIF_HINT_START = "[GIF_HINT_START]"
    _GIF_HINT_END = "[GIF_HINT_END]"

    # 转述模型实例被本插件包装的标记属性名
    _WRAP_FLAG_ATTR = "_read_gif_text_chat_wrapped"

    # 框架 PreProcessStage 模块路径（v4.26+ 在此对图片调用 ensure_jpeg）。
    # 该阶段早于本插件任何钩子，会把 GIF 转成单帧 JPEG，使后续无从识别 GIF。
    _PREPROCESS_MODULE = "astrbot.core.pipeline.preprocess_stage.stage"
    # ensure_jpeg 已被本插件包装的标记属性名（幂等 + 卸载用）
    _ENSURE_JPEG_FLAG = "_read_gif_ensure_jpeg_patched"

    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config
        self.processor = GifProcessor()
        self._ensure_cache_dir()
        self._last_cleanup = time.time()
        # 转述模型 text_chat 包装状态
        self._wrapped_provider = None        # 被包装的转述模型实例
        self._caption_hint_active = False    # 本轮包装是否生效（存活期内每次带图调用都追加）
        # 中和框架 PreProcessStage 对 GIF 的 JPEG 转换（仅 v4.26+ 需要，旧版本自动跳过）
        self._install_ensure_jpeg_guard()

    def _install_ensure_jpeg_guard(self) -> None:
        """中和框架 PreProcessStage 对 GIF 的 JPEG 转换。

        背景：AstrBot v4.26+ 在 PreProcessStage（早于本插件任何钩子的管道阶段）
        对消息链中每个 Image 调用 ensure_jpeg，把非 JPEG 图片（含 GIF）转成单帧
        JPEG 并改写组件的 file/path/url。GIF 因此在抵达本插件的 on_waiting_llm_request
        钩子前就被破坏成单帧，插件无从识别，表现为"完全静默失效"。

        本方法在插件加载时，把 PreProcessStage 模块命名空间里的 ensure_jpeg 名字
        替换为一个包装：遇到 GIF（magic bytes 判定，与 GifProcessor.is_gif 一致）
        原样返回，跳过转换；非 GIF 完全透传原函数。GIF 由此原封不动抵达本插件钩子，
        等价于把框架对 GIF 的行为退回到 v4.25.6。

        安全性：
        - 只替换 stage 模块的一个全局名（调用处 `await ensure_jpeg(...)` 在调用时
          按该名查找），不修改框架任何源码文件，不触碰 media_utils 源定义。
        - 旧版本（无该破坏逻辑）探测不到属性，直接跳过，不 patch。
        - 幂等：已被本插件包装过则不重复包装（覆盖热重载）。
        - 整体包 try/except：框架结构若再变动，最坏只是静默不 patch，绝不影响插件其余功能。
        """
        try:
            mod = importlib.import_module(self._PREPROCESS_MODULE)
        except Exception as exc:
            # 模块不存在/结构变动：不影响插件其余功能
            self._dlog(f"[astrbot_plugin_read_gif] 未能定位 PreProcessStage 模块，跳过 GIF 预处理中和: {exc}")
            return

        original = getattr(mod, "ensure_jpeg", None)
        if original is None:
            # 旧版本（如 v4.25.6）没有此破坏逻辑，GIF 本就正常，无需 patch
            self._dlog("[astrbot_plugin_read_gif] 当前框架无 ensure_jpeg 预处理，无需中和")
            return
        if getattr(original, self._ENSURE_JPEG_FLAG, False):
            # 已被本插件包装（热重载场景），保持幂等
            return

        async def _guarded_ensure_jpeg(*args, **kwargs):
            # 安全取首个图片路径参数（兼容位置/关键字两种传法，且不假设后续参数）
            image_path = args[0] if args else kwargs.get("image_path")
            # 仅对 GIF 跳过：magic bytes 判定，与 GifProcessor.is_gif 完全一致
            try:
                if image_path and os.path.isfile(image_path):
                    with open(image_path, "rb") as f:
                        if f.read(6) in (b"GIF87a", b"GIF89a"):
                            return image_path  # 原样返回，GIF 不被破坏
            except OSError:
                pass
            # 非 GIF：原参数 100% 透传框架原始行为（签名变动也无损转发）
            return await original(*args, **kwargs)

        setattr(_guarded_ensure_jpeg, self._ENSURE_JPEG_FLAG, True)
        # 留存原函数，供卸载时恢复
        _guarded_ensure_jpeg._read_gif_original = original
        mod.ensure_jpeg = _guarded_ensure_jpeg
        logger.info(
            "[astrbot_plugin_read_gif] 已中和框架 PreProcessStage 的 GIF→JPEG 预处理，"
            "GIF 将原样抵达本插件处理"
        )

    def _uninstall_ensure_jpeg_guard(self) -> None:
        """卸载 ensure_jpeg 包装，恢复框架原始函数。幂等，无副作用。"""
        try:
            mod = importlib.import_module(self._PREPROCESS_MODULE)
        except Exception:
            return
        current = getattr(mod, "ensure_jpeg", None)
        if current is not None and getattr(current, self._ENSURE_JPEG_FLAG, False):
            original = getattr(current, "_read_gif_original", None)
            if original is not None:
                mod.ensure_jpeg = original
                logger.info("[astrbot_plugin_read_gif] 已恢复框架原始 ensure_jpeg 预处理")

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

    def _dlog(self, msg: str) -> None:
        """诊断日志统一出口。

        默认走 logger.debug；当配置 debug_to_info 开启时，升级为 logger.info，
        便于在不调整全局日志等级的情况下观察插件内部行为。
        """
        if self._get_config("debug_to_info", False):
            logger.info(msg)
        else:
            logger.debug(msg)

    def _should_cleanup(self) -> bool:
        """判断是否应该执行自动缓存清理。"""
        interval_min = self._get_config("auto_cleanup_interval_min", 60)
        if interval_min <= 0:
            return False
        elapsed = time.time() - self._last_cleanup
        return elapsed >= interval_min * 60

    def _get_provider_settings(self, event: AstrMessageEvent) -> dict:
        """获取当前会话的 provider_settings dict。

        返回的对象与框架 _decorate_llm_request 中读取的 cfg 是同一个引用，
        修改它即会影响本轮 build_main_agent 的图片转述行为。
        """
        cfg = self.context.get_config(event.unified_msg_origin)
        return cfg.get("provider_settings", {}) or {}

    def _is_caption_path(self, event: AstrMessageEvent) -> bool:
        """判断本轮是否命中"图片转述"路径。

        命中条件：主 LLM 不支持图片模态 + 配了默认图片转述模型。
        与框架 astr_main_agent._ensure_img_caption 的触发条件一致。
        """
        settings = self._get_provider_settings(event)
        if not settings.get("default_image_caption_provider_id"):
            return False
        provider = self.context.get_using_provider(event.unified_msg_origin)
        if provider is None:
            return False
        modalities = provider.provider_config.get("modalities")
        # 与框架 _provider_supports_modality 一致：空列表视为未配置=支持
        if modalities == []:
            return False
        return "image" not in modalities

    def _get_caption_provider(self, event: AstrMessageEvent):
        """获取本轮图片转述模型实例（与框架转述时取到的是同一个实例）。

        框架 _request_img_caption / _process_quote_message 都用
        context.get_provider_by_id(default_image_caption_provider_id) 取实例，
        二者从全局 inst_map 取同一个对象，因此这里取到的就是框架会调用的那个。
        """
        prov_id = self._get_provider_settings(event).get(
            "default_image_caption_provider_id"
        )
        if not prov_id:
            return None
        try:
            return self.context.get_provider_by_id(prov_id)
        except Exception as exc:
            self._dlog(f"[astrbot_plugin_read_gif] 获取转述模型实例失败: {exc}")
            return None

    def _install_caption_wrapper(self, event: AstrMessageEvent) -> None:
        """在转述模型实例上临时包装 text_chat，向 prompt 结尾追加 GIF 提示词。

        通用性：对任意 Provider 子类（OpenAI / Anthropic / Gemini / 自建等）都适用，
        因为包装的是实例方法、且用 *args/**kwargs 原样透传，不依赖具体模型与签名。

        覆盖两条转述路径：直接发送的 GIF（_request_img_caption，一次调用传全部图）
        和引用消息里的 GIF（_process_quote_message，单独一次调用）最终都调用本实例
        的 text_chat。同一条消息若同时含"直接图"和"引用图"，框架会发两次带图调用，
        因此提示词在包装存活期内对【每次】带图调用都追加，而非一次性消费。

        匹配方式：不认 prompt 文本、不认图片路径（压缩后为随机名），只认
        "本轮包装存活（_caption_hint_active）"+ "本次调用带 image_urls"。
        本轮边界由 on_waiting 入口 / on_llm_request / terminate 的卸载来界定。
        """
        hint_text = self._get_config("gif_hint_text", "")
        if not hint_text:
            # 提示词为空：不安装包装。日志后缀据此保持一致（不声称已附加）。
            return
        provider = self._get_caption_provider(event)
        if provider is None:
            return

        # 防重复/跨轮残留：若已被本插件包装，先卸载再重装，确保闭包捕获本轮 hint
        if getattr(provider, self._WRAP_FLAG_ATTR, False):
            self._uninstall_caption_wrapper()
            provider = self._get_caption_provider(event)
            if provider is None:
                return

        original_text_chat = provider.text_chat
        plugin = self

        async def _wrapped_text_chat(*args, **kwargs):
            # 包装存活期内，对每次带图调用都追加（覆盖直接图、引用图两次调用）
            if plugin._caption_hint_active and kwargs.get("image_urls"):
                if "prompt" in kwargs:
                    base = kwargs.get("prompt") or ""
                    kwargs["prompt"] = f"{base}\n\n{hint_text}" if base else hint_text
                elif args:
                    base = args[0] or ""
                    new_prompt = f"{base}\n\n{hint_text}" if base else hint_text
                    args = (new_prompt,) + args[1:]
                else:
                    kwargs["prompt"] = hint_text
                plugin._dlog(
                    "[astrbot_plugin_read_gif] 已向转述模型 prompt 结尾追加 GIF 提示词"
                )
            return await original_text_chat(*args, **kwargs)

        # 实例属性遮蔽类方法；卸载时 del 即恢复，不污染类、不影响其它 provider
        provider.text_chat = _wrapped_text_chat
        setattr(provider, self._WRAP_FLAG_ATTR, True)
        self._wrapped_provider = provider
        self._caption_hint_active = True

    def _uninstall_caption_wrapper(self) -> None:
        """卸载转述模型 text_chat 包装，恢复到原始实例方法。

        删除实例属性即可让属性查找回退到类方法，无副作用。多处调用以兜底：
        on_waiting 入口、on_llm_request 入口、terminate。幂等。
        """
        provider = self._wrapped_provider
        self._caption_hint_active = False
        if provider is None:
            return
        try:
            if "text_chat" in provider.__dict__:
                del provider.text_chat
            if hasattr(provider, self._WRAP_FLAG_ATTR):
                try:
                    delattr(provider, self._WRAP_FLAG_ATTR)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"[astrbot_plugin_read_gif] 卸载转述包装失败: {exc}")
        finally:
            self._wrapped_provider = None

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

    @register_on_waiting_llm_request()
    async def on_waiting_llm_request(self, event: AstrMessageEvent) -> None:
        """LLM 请求准备阶段：在 build_main_agent 之前替换 GIF。

        这是替换 GIF 的最佳时机：
        - 早于 build_main_agent，避免 GIF 被 AstrBot 图片压缩破坏成单帧 JPEG
        - 只在触发 LLM 时执行，不影响纯图片无文字消息
        - 修改 event.message_obj.message 后，build_main_agent 会把宫格图当普通图片处理

        按路径分流（仅在本轮确实替换了 GIF 时才判断和影响）：
        - 主 LLM 支持图片：走 on_llm_request 的 system_prompt 注入路径
        - 主 LLM 不支持图片 + 配了图片转述模型：包装转述模型 text_chat，
          让转述模型按"GIF 关键帧序列"描述（覆盖直接发送与引用两种情境）；
          on_llm_request 阶段跳过 system_prompt 注入，避免双重提示

        插件开闭由 AstrBot 主框架统一管理，此处不再自带 enabled 门控。
        """
        # 入口兜底：若上一轮因异常未卸载转述包装，先卸载，避免污染本轮
        self._uninstall_caption_wrapper()

        await self._maybe_cleanup()

        new_message = []
        modified = False
        last_info = None
        gif_count = 0

        async def _handle_image(comp: Image):
            """替换单个 Image，返回 (新组件或None, info或None)。"""
            try:
                image_path = await comp.convert_to_file_path()
            except Exception as exc:
                self._dlog(f"[astrbot_plugin_read_gif] 获取图片路径失败: {exc}")
                return None, None
            if not self.processor.is_gif(image_path):
                return None, None
            try:
                grid_path, info = await self.processor.process_gif(
                    image_path,
                    grid_preset=self._get_config("grid_preset", "auto"),
                    cache_dir=self._get_cache_dir(),
                    max_output_size=self._get_config("max_output_size", 1600),
                )
            except Exception as exc:
                logger.warning(f"[astrbot_plugin_read_gif] GIF 处理失败: {exc}")
                return None, None
            if not grid_path or not os.path.exists(grid_path):
                return None, None
            return Image.fromFileSystem(grid_path), info

        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                new_image, info = await _handle_image(comp)
                if new_image is not None:
                    new_message.append(new_image)
                    modified = True
                    gif_count += 1
                    if info:
                        last_info = info
                else:
                    new_message.append(comp)

            elif isinstance(comp, Reply):
                # 递归扫描引用消息 chain 中的 Image
                if comp.chain:
                    new_chain = []
                    chain_modified = False
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            new_image, info = await _handle_image(reply_comp)
                            if new_image is not None:
                                new_chain.append(new_image)
                                chain_modified = True
                                gif_count += 1
                                if info:
                                    last_info = info
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
            event.set_extra("gif_processed", True)

            # 判断本轮路径（仅在本轮确实有 GIF 时才判断，无 GIF 不碰任何配置）
            caption_path = self._is_caption_path(event)
            event.set_extra("gif_caption_path", caption_path)

            if caption_path:
                # 转述路径：包装转述模型 text_chat，让转述模型按 GIF 序列描述
                self._install_caption_wrapper(event)

            # 统一日志（仅当包装真正安装成功时才显示"已附加提示词"后缀，
            # 与实际行为一致：gif_hint_text 为空或取不到转述模型时不会安装包装）
            if last_info:
                preset = self._get_config("grid_preset", "auto")
                if caption_path:
                    if self._caption_hint_active:
                        suffix = "（当前主LLM不支持图片模态，已对识图模型附加提示词）"
                    else:
                        suffix = "（当前主LLM不支持图片模态，未配置提示词，仅做宫格转换）"
                else:
                    suffix = ""
                count_text = f"，本轮共{gif_count}张GIF" if gif_count > 1 else ""
                logger.info(
                    f"[astrbot_plugin_read_gif] GIF帧数{last_info['frame_count']}，"
                    f"秒数{last_info['duration_s']:.2f}s，"
                    f"已选[{preset}]，"
                    f"{'智能转为' if preset == 'auto' else '转为'}{last_info['grid_size']}宫格"
                    f"{count_text}{suffix}"
                )

    @register_on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """提示词注入 + 第三方 Agent Runner 路径检测 + 转述包装卸载。

        正常情况下，on_waiting_llm_request 已经把 GIF 替换成了宫格图，
        build_main_agent 处理的是宫格图，req.image_urls 中不会出现 GIF。

        第三方 Agent Runner（Dify/Coze/Dashscope/DeerFlow）路径不触发
        OnWaitingLLMRequestEvent，GIF 会以 base64 原文进入 req.image_urls，
        本插件无法有效替换（外部平台通常只取第一帧）。此处检测到该情况时
        打一条 info 日志提示用户，不静默失效。

        路径分流（on_waiting_llm_request 已标记 gif_caption_path）：
        - True（转述路径）：转述调用在 build_main_agent 内已发生、包装已消费，
          此处卸载包装兜底；跳过 system_prompt 注入，避免双重提示。
        - False（主 LLM 直看图路径）：注入提示词到 system_prompt，让 LLM
          知道这是动图。用标记对包裹，注入前先清理旧痕迹。
        """
        # 转述调用发生在 build_main_agent 内（早于本钩子），此处卸载兜底，
        # 确保包装不残留到下一轮或其它会话（主图直看路径未装包装，卸载为 no-op）
        self._uninstall_caption_wrapper()

        # 检测第三方 Agent Runner 路径：req.image_urls 中出现 base64 GIF
        third_party_gif_count = self._count_base64_gif_in_urls(req.image_urls)
        if third_party_gif_count > 0:
            logger.info(
                "[astrbot_plugin_read_gif] 检测到第三方 Agent Runner 路径下传入了 "
                f"{third_party_gif_count} 张 GIF，本插件仅对内置 Agent 生效，"
                "GIF 替换将不生效（外部平台通常只取第一帧）。"
            )

        # 检查 on_waiting_llm_request 是否标记了 GIF 处理
        gif_flag = event.get_extra("gif_processed", False)
        if not gif_flag:
            return

        caption_path = event.get_extra("gif_caption_path", False)
        if caption_path:
            # 转述路径：提示词已在 build_main_agent 内由包装追加到转述 prompt，
            # 跳过 system_prompt 注入，避免双重提示
            self._dlog("[astrbot_plugin_read_gif] 转述路径，跳过 system_prompt 注入")
            return

        # 主 LLM 直看图路径：注入提示词到 system_prompt
        hint_text = self._get_config("gif_hint_text", "")
        if hint_text:
            # 防御性清理：移除可能存在的旧提示词标记段
            # （system_prompt 每轮由 build_main_agent 重建，正常情况无残留；
            #   此清理防止同轮多次触发或框架行为变化的边缘情况）
            pattern = re.compile(
                re.escape(self._GIF_HINT_START)
                + r".*?"
                + re.escape(self._GIF_HINT_END)
                + r"\n*",
                re.DOTALL,
            )
            req.system_prompt = pattern.sub("", req.system_prompt or "")
            # 追加新提示词，用标记对包裹
            req.system_prompt = (
                f"{req.system_prompt or ''}\n"
                f"{self._GIF_HINT_START}\n{hint_text}\n{self._GIF_HINT_END}\n"
            )
            self._dlog("[astrbot_plugin_read_gif] 已注入 GIF 提示词到 system_prompt")

    @staticmethod
    def _count_base64_gif_in_urls(urls) -> int:
        """统计 req.image_urls 中以 base64 编码的 GIF 数量。

        第三方 Agent Runner 路径会把图片转为 base64 塞入 req.image_urls，
        其中 GIF 的 base64 解码后前 6 字节为 GIF87a/GIF89a。
        本地路径/url 路径不是第三方路径特征，跳过。
        """
        if not urls:
            return 0
        count = 0
        for item in urls:
            if not isinstance(item, str):
                continue
            raw = None
            if item.startswith("data:image/gif;base64,"):
                raw = item.split(",", 1)[1] if "," in item else ""
            elif item.startswith("base64://"):
                raw = item[len("base64://"):]
            elif not item.startswith(("http", "file://", "/")) and not os.path.isfile(item):
                # 纯 base64 字符串（第三方路径 convert_to_base64 的产出）
                raw = item
            if raw:
                try:
                    header = base64.b64decode(raw[:12])[:6]
                    if header in (b"GIF87a", b"GIF89a"):
                        count += 1
                except Exception:
                    pass
        return count

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

    async def terminate(self) -> None:
        """插件禁用/重载时调用：兜底卸载本插件安装的所有运行时包装。

        两类包装都需在此恢复，互不影响：
        1) 转述模型实例的 text_chat 包装（若本轮残留）；
        2) 框架 PreProcessStage 的 ensure_jpeg 包装（GIF 防破坏 guard）。
        若插件在已安装包装的状态下被卸载或热重载，不恢复会留下残留覆盖。
        """
        if self._wrapped_provider is not None:
            self._uninstall_caption_wrapper()
            logger.info("[astrbot_plugin_read_gif] terminate 时已兜底卸载转述包装")
        # 恢复框架原始 ensure_jpeg，避免插件卸载后仍残留 GIF guard 包装
        self._uninstall_ensure_jpeg_guard()
