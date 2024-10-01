from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Optional
from openai import OpenAI
import requests

@dataclass
class Message:
    role: str
    content: Optional[str]
    json_data: Optional[any]

class Model(ABC):
    @abstractmethod
    def run(messages: List[Message], **options) -> Message:
        pass

class OpenAIModel(Model):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def run(self, messages: List[Message], json=True, **options) -> Message:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"} if json else None,
            **options
        ).choices[0].message
        if json:
            return Message(response.role, None, response.json_data)
        return Message(response.role, response.content)
    
class LocalHostedModel(Model):
    def __init__(self, url: str):
        self.url = url

    def run(self, messages: List[Message], **options) -> Message:
        if len(messages) != 1:
            raise Exception("LocalHostedModel only supports one message at a time")
        response = requests.post(self.url, json={'messages': messages, **options}).json()
        return Message(response.role, response.content)