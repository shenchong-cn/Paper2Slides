"""
自定义 API 客户端，使用 requests 而不是 OpenAI 客户端
"""

import os
import requests
import json
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