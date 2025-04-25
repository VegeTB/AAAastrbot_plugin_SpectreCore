from astrbot.api.all import *
from astrbot.api.event import filter
from .utils import *
import aiohttp
import json
import asyncio


@register(
    "spectrecore_alpha",
    "vege_testing",
    "使大模型更好的主动回复群聊中的消息，带来生动和沉浸的群聊对话体验",
    "2.1.0",
    "https://github.com/Vege_TB/AAAastrbot_plugin_SpectreCore"
)

class SpectreCore(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.model_handler = DualModelHandler(config)

        # 初始化原有模块
        HistoryStorage.init(config)
        ImageCaptionUtils.init(context, config)

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群消息喵"""
        try:
            async for result in self._process_message(event):
                yield result
        except Exception as e:
            logger.error(f"处理群消息时发生错误: {e}")

    @event_message_type(EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent):
        """处理私聊消息喵"""
        try:
            async for result in self._process_message(event):
                yield result
        except Exception as e:
            logger.error(f"处理私聊消息时发生错误: {e}")

    async def _process_message(self, event: AstrMessageEvent):
        """处理消息的通用逻辑：保存历史记录并尝试回复"""
        # 保存用户消息到历史记录
        HistoryStorage.process_and_save_user_message(event)

        # 尝试自动回复（传入当前配置）
        if ReplyDecision.should_reply(event, self.config):
            async for result in self._dual_model_process(event):
                yield result

    async def _dual_model_process(self, event):
        """双模型协同处理流程"""
        try:
            raw_input = MessageUtils.clean_message(event.message_str)

            # 并行调用双模型
            v3_task = self.model_handler.call_v3(raw_input)
            r1_task = self.model_handler.call_r1(f"分析对话上下文：{raw_input}")
            v3_response, r1_analysis = await asyncio.gather(v3_task, r1_task)

            # 生成增强型提示
            enhanced_prompt = self._build_dual_prompt(
                raw_input,
                r1_analysis,
                HistoryStorage.get_current_context(event)
            )

            # 获取最终响应
            final_response = await self.model_handler.call_v3(enhanced_prompt)

            # 保存上下文并返回结果
            HistoryStorage.process_and_save_bot_message(final_response, event)
            yield event.plain_result(final_response)

        except Exception as e:
            logger.error(f"双模型处理失败: {str(e)}")
            yield event.plain_result("思考出现了一点小问题...")

    def _build_dual_prompt(self, raw, analysis, context):
        """构建双模型协作提示模板"""
        return f"""
        [对话上下文]
        {context}

        [当前消息] {raw}
        [分析建议] {analysis}

        请根据以上分析生成自然回复：
        """

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """处理bot发送的消息喵"""
        try:
            if event._result and hasattr(event._result, "chain"):
                HistoryStorage.save_bot_message_from_chain(event._result.chain, event)
                logger.debug(f"已保存bot回复消息到历史记录")
        except Exception as e:
            logger.error(f"处理bot发送的消息时发生错误: {e}")

    from astrbot.api.provider import LLMResponse
    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse, priority=114514):
        """处理大模型回复喵"""
        logger.debug(f"收到大模型回复喵: {resp}")
        try:
            if resp.role != "assistant":
                return
            resp.completion_text = TextFilter.process_model_text(resp.completion_text, self.config)
            if resp.completion_text == "<NO_RESPONSE>":
                event.stop_event()
        except Exception as e:
            logger.error(f"处理大模型回复时发生错误: {e}")

    @filter.command_group("spectrecore", alias={'sc'})
    def spectrecore(self):
        """插件的前缀喵 可以用sc代替喵"""
        pass

    @spectrecore.command("help", alias=['帮助', 'helpme'])
    async def help(self, event: AstrMessageEvent):
        """查看插件的帮助喵"""
        yield event.plain_result(
            "SpectreCore插件帮助文档\n"
            "使用spectrecore或sc作为指令前缀 如/sc help\n"
            "使用reset指令重置当前聊天记录 如/sc reset\n"
            "   你也可以重置指定群聊天记录 如/sc reset 群号\n"
            "使用history指令可以查看最近聊天记录 如/sc history\n"
            "↓强烈建议您阅读Github中的README文档\n↓"
            "https://github.com/23q3/astrbot_plugin_SpectreCore"
        )

    @spectrecore.command("history")
    async def history(self, event: AstrMessageEvent, count: int = 10):
        """查看最近的聊天记录喵，默认10条喵，示例/sc history 5"""
        try:
            platform_name = event.get_platform_name()
            is_private = event.is_private_chat()
            chat_id = event.get_group_id() if not is_private else event.get_sender_id()

            if not chat_id:
                yield event.plain_result("获取聊天ID失败喵，无法显示历史记录")
                return

            history = HistoryStorage.get_history(platform_name, is_private, chat_id)
            if not history:
                yield event.plain_result("暂无聊天记录喵")
                return

            if count > 20:
                count = 20

            recent_history = history[-count:] if len(history) > count else history
            formatted_history = await MessageUtils.format_history_for_llm(recent_history)

            chat_type = "私聊" if is_private else f"群聊({chat_id})"
            title = f"最近{len(recent_history)}条{chat_type}聊天记录喵：\n\n"
            full_content = title + formatted_history

            if len(full_content) > 3000:
                image_url = await self.text_to_image(full_content)
                yield event.image_result(image_url)
            else:
                yield event.plain_result(full_content)

        except Exception as e:
            logger.error(f"获取历史记录时发生错误: {e}")
            yield event.plain_result(f"获取历史记录失败喵：{str(e)}")

    @spectrecore.command("reset")
    async def reset(self, event: AstrMessageEvent, group_id: str = None):
        """重置历史记录喵"""
        try:
            platform_name = event.get_platform_name()

            if group_id:
                is_private = False
                chat_id = group_id
                chat_type = f"群聊({group_id})"
            else:
                is_private = event.is_private_chat()
                chat_id = event.get_group_id() if not is_private else event.get_sender_id()
                chat_type = "私聊" if is_private else f"群聊({chat_id})"

                if not chat_id:
                    yield event.plain_result("获取聊天ID失败喵，无法重置历史记录")
                    return

            history = HistoryStorage.get_history(platform_name, is_private, chat_id)
            if not history:
                yield event.plain_result(f"{chat_type}没有历史记录喵，无需重置")
                return

            success = HistoryStorage.clear_history(platform_name, is_private, chat_id)

            if success:
                yield event.plain_result(f"已成功重置{chat_type}的历史记录喵~")
            else:
                yield event.plain_result(f"重置{chat_type}的历史记录失败喵，可能发生错误")

        except Exception as e:
            logger.error(f"重置历史记录时发生错误: {e}")
            yield event.plain_result(f"重置历史记录失败喵：{str(e)}")

    @spectrecore.command("callllm")
    async def callllm(self, event: AstrMessageEvent):
        """触发大模型回复"""
        try:
            yield await LLMUtils.call_llm(event, self.config, self.context)
        except Exception as e:
            logger.error(f"调用大模型时发生错误: {e}")
            yield event.plain_result(f"触发大模型回复失败喵：{str(e)}")


class DualModelHandler:
    """双模型处理器（保持不变）"""

    def __init__(self, config):
        self.v3_endpoint = config.get("deepseek.v3_endpoint")
        self.r1_endpoint = config.get("deepseek.r1_endpoint")
        self.api_key = config.get("deepseek.api_key")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def call_v3(self, prompt, context=None):
        payload = self._build_payload("v3", prompt, context)
        return await self._call_api(self.v3_endpoint, payload)

    async def call_r1(self, prompt):
        payload = self._build_payload("r1", prompt)
        return await self._call_api(self.r1_endpoint, payload)

    def _build_payload(self, model_type, prompt, context=None):
        base = {
            "v3": {"temperature": 0.7, "max_tokens": 1024},
            "r1": {"temperature": 0.5, "max_tokens": 256}
        }[model_type]

        messages = [{"role": "user", "content": prompt}]
        if context:
            messages = context + messages

        return {"messages": messages, **base}

    async def _call_api(self, endpoint, payload):
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=self.headers, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data['choices'][0]['message']['content']


class ReplyDecision:
    """修正后的决策模块（无需初始化）"""

    @classmethod
    def should_reply(cls, event, config):
        if not config.get("auto_reply"):
            return False

        # 强制回复条件
        if event.is_at_bot():
            return True

        # 概率性回复
        return random.random() < config.get("reply_probability", 0.3)