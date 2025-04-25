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
        ReplyDecision.init(config)  # 扩展决策模块

    # 原有事件处理保持不变...

    async def _process_message(self, event: AstrMessageEvent):
        """修改后的消息处理流程"""
        # 保存用户消息到历史记录
        HistoryStorage.process_and_save_user_message(event)

        # 判断是否需要回复（读空气功能）
        if not ReplyDecision.should_reply(event, self.config):
            return

        # 双模型协同处理
        try:
            raw_input = MessageUtils.clean_message(event.message_str)

            # 并行调用双模型
            v3_task = self.model_handler.call_v3(raw_input)
            r1_task = self.model_handler.call_r1(f"将要并快速地思考对话上下文：{raw_input}")
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

        请根据以上分析思考，根据你的设定和规则生成自然回复：
        """

    # 其他原有方法保持不变...


class DualModelHandler:
    """新增的双模型处理器"""

    def __init__(self, config):
        self.v3_endpoint = config.get("deepseek.v3_endpoint")
        self.r1_endpoint = config.get("deepseek.r1_endpoint")
        self.api_key = config.get("deepseek.api_key")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def call_v3(self, prompt, context=None):
        """主对话模型调用"""
        payload = self._build_payload("v3", prompt, context)
        return await self._call_api(self.v3_endpoint, payload)

    async def call_r1(self, prompt):
        """思考模型调用"""
        payload = self._build_payload("r1", prompt)
        return await self._call_api(self.r1_endpoint, payload)

    def _build_payload(self, model_type, prompt, context=None):
        base = {
            "v3": {
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "r1": {
                "temperature": 0.5,
                "max_tokens": 256
            }
        }[model_type]

        return {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **base
        }

    async def _call_api(self, endpoint, payload):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    endpoint,
                    headers=self.headers,
                    json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data['choices'][0]['message']['content']


class ReplyDecision:
    """扩展的决策模块"""

    @classmethod
    def should_reply(cls, event, config):
        # 原有读空气逻辑
        if not config.get("auto_reply"):
            return False

        # 新增：检查@消息
        if event.is_at_bot():
            return True

        # 原有概率性回复逻辑
        return random.random() < config.get("reply_probability", 0.3)

    # 其他原有方法保持不变...


# 在config.py中添加配置项
class SpectreCoreConfig(AstrBotConfig):
    def __init__(self):
        super().__init__()
        self.add_field(
            "deepseek.v3_endpoint",
            "DeepSeek V3 API地址",
            field_type=str,
            required=True
        )
        self.add_field(
            "deepseek.r1_endpoint",
            "DeepSeek R1 API地址",
            field_type=str,
            required=True
        )
        self.add_field(
            "deepseek.api_key",
            "API密钥",
            field_type=str,
            required=True
        )