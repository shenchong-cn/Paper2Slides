"""
自定义 API 客户端，使用 requests 而不是 OpenAI 客户端
"""

import os
import requests
import json
import time
from typing import List, Dict, Any, Optional
from openai.types.chat import ChatCompletion


class CustomAPIClient:
    """自定义 API 客户端，兼容 OpenAI 客户端接口"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        })

    class chat:
        """嵌套的 chat 类，兼容 OpenAI 客户端接口"""

        def __init__(self, parent_client):
            self.client = parent_client

        class completions:
            """嵌套的 completions 类"""

            def __init__(self, chat_client):
                self.chat_client = chat_client

            def create(self, model: str, messages: List[Dict[str, str]],
                      max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> ChatCompletion:
                """创建聊天完成请求"""

                url = f"{self.chat_client.client.base_url}/chat/completions"

                data = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }

                try:
                    response = self.chat_client.client.session.post(url, json=data, timeout=60)
                    response.raise_for_status()

                    result = response.json()

                    # 创建一个兼容的 ChatCompletion 对象
                    return ChatCompletion(**result)

                except requests.exceptions.RequestException as e:
                    raise Exception(f"API 请求失败: {e}")

    def __init_subclass__(cls):
        """初始化嵌套类"""
        super().__init_subclass__()
        cls.chat = type('chat', (), {})
        cls.chat.completions = type('completions', (), {})
        cls.chat.completions.create = lambda self, **kwargs: self._create_completion(**kwargs)


def create_custom_client(api_key: str, base_url: Optional[str] = None) -> CustomAPIClient:
    """创建自定义 API 客户端"""
    # 检测是否是Gemini API
    is_gemini = base_url and "generativelanguage.googleapis.com" in base_url

    if is_gemini:
        return GeminiAPIClient(api_key, base_url)
    else:
        client = CustomAPIClient(api_key, base_url)

        # 手动设置嵌套类结构
        chat_completions = type('completions', (), {})

        def create_completion(model: str, messages: List[Dict[str, str]],
                             max_tokens: int = 1000, temperature: float = 0.7, **kwargs):
            url = f"{client.base_url}/chat/completions"

            data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }

            try:
                response = client.session.post(url, json=data, timeout=60)
                response.raise_for_status()

                result = response.json()
                return ChatCompletion(**result)

            except requests.exceptions.RequestException as e:
                raise Exception(f"API 请求失败: {e}")

        chat_completions.create = create_completion

        chat_class = type('chat', (), {})
        chat_class.completions = chat_completions

        client.chat = chat_class

        return client


def convert_to_openai_format(gemini_response: dict, model: str) -> dict:
    """将Gemini API响应转换为OpenAI格式"""
    candidates = gemini_response.get("candidates", [])

    if not candidates:
        # 如果没有候选响应，返回空响应
        return {
            "id": f"gemini-{hash(str(gemini_response))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": "stop"
            }]
        }

    # 取第一个候选响应
    candidate = candidates[0]
    content_parts = candidate.get("content", {}).get("parts", [])

    # 提取文本内容
    content = ""
    for part in content_parts:
        if "text" in part:
            content += part["text"]

    return {
        "id": gemini_response.get("id", f"gemini-{hash(str(gemini_response))}"),
        "object": "chat.completion",
        "created": gemini_response.get("created", int(time.time())),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": candidate.get("finishReason", "stop").lower()
        }],
        "usage": {
            "prompt_tokens": gemini_response.get("usageMetadata", {}).get("promptTokenCount", 0),
            "completion_tokens": gemini_response.get("usageMetadata", {}).get("candidatesTokenCount", 0),
            "total_tokens": gemini_response.get("usageMetadata", {}).get("totalTokenCount", 0)
        }
    }


class GeminiAPIClient:
    """Gemini API 客户端，兼容 OpenAI 客户端接口"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        })

        # 设置嵌套类结构
        chat_completions = type('completions', (), {})

        def create_completion(model: str, messages: List[Dict[str, str]],
                             max_tokens: int = 1000, temperature: float = 0.7, **kwargs):
            """创建聊天完成请求"""

            # 将OpenAI格式的messages转换为Gemini格式
            gemini_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Gemini使用"user"和"model"角色
                gemini_role = "user" if role == "user" else "model"

                gemini_messages.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })

            # 构建Gemini API请求
            url = f"{self.base_url}/models/{model}:generateContent"

            data = {
                "contents": gemini_messages,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "candidateCount": 1,
                    "stopSequences": [],
                    **kwargs
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }

            try:
                response = self.session.post(url, json=data, timeout=60)
                response.raise_for_status()

                result = response.json()

                # 将Gemini响应转换为OpenAI兼容格式
                openai_response = convert_to_openai_format(result, model)
                return ChatCompletion(**openai_response)

            except requests.exceptions.RequestException as e:
                raise Exception(f"Gemini API 请求失败: {e}")

        chat_completions.create = create_completion

        chat_class = type('chat', (), {})
        chat_class.completions = chat_completions

        self.chat = chat_class