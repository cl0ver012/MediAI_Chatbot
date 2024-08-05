from concurrent.futures.thread import ThreadPoolExecutor
from logging import getLogger
from time import time
from typing import Type, List, Union

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel


from config import PromptTemplate, get_prompt_template, ModelType

logger = getLogger('freedo')


class LLMService:

    def __del__(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def __init__(self):
        logger.debug("Initializing LLMService")
        self.pool = ThreadPoolExecutor()
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.cache = {}

    def embedding(self, input: str, model=ModelType.embedding) -> List[float]:
        return self.client.embeddings.create(input=[input], model=model).data[0].embedding

    def embeddings(self, inputs: List[str], model=ModelType.embedding) -> Union[List[float], List[List[float]]]:
        return [d.embedding for d in self.client.embeddings.create(input=inputs, model=model).data]

    def _do(self, model_type: ModelType, query_type, messages, user: str, json_response: bool = False) -> str:
        logger.debug(f"{user}: {query_type} model={model_type}")
        # messages = [
        #     {"role": "user", "content": prompt},
        # ]
        if json_response:
            completion = self.client.chat.completions.create(
                model=model_type,
                messages=messages,
                response_format={"type": "json_object"},
                user=user
            )
        else:
            completion = self.client.chat.completions.create(
                model=model_type,
                messages=messages,
                user=user)
        response_obj = completion.model_dump()
        response_obj['user'] = user
        text = completion.choices[0].message.content
        return text

    def do(
            self,
            query_type: str,
            model_type: ModelType,
            messages,
            shadow_models: List[ModelType] = None,
            user=None,
            json_response: bool = False) -> str:
        logger.debug(f"{user}: {query_type} {messages}")
        return self.shadow_wrapper(model_type, shadow_models, self._do, query_type, messages, user, json_response)

    def shadow_wrapper(self, model_type: ModelType, shadow_models: List[ModelType], fn, *args):
        f = self.pool.submit(fn, model_type, *args)
        if shadow_models:
            for model in shadow_models:
                self.pool.submit(fn, model, *args)
        return f.result()