from concurrent.futures.thread import ThreadPoolExecutor
from logging import getLogger
from time import time
from typing import Type, List, Union
import json

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel


from config import PromptTemplate, get_prompt_template, ModelType, get_function_template, FunctionTemplate, Fixtures, get_fixture

logger = getLogger('mediAI')


class LLMService:

    def __del__(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def __init__(self):
        logger.debug("Initializing LLMService")
        self.pool = ThreadPoolExecutor()
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.cache = {}
        self.system_prompt = get_prompt_template(PromptTemplate.SYSTEM_PROMPT)
        self.doctors = get_fixture(Fixtures.DOCTOR_TEXT)
        self.example_conversation = get_fixture(Fixtures.EXAMPLE_CONVERSATION)
        self.doctor_prompt = get_prompt_template(PromptTemplate.DOCTOR_PROMPT)
        self.final = self.doctor_prompt.format(system_prompt=self.system_prompt, doctors=self.doctors, example_conversation=self.example_conversation)
        print("---")
        print(self.final)

    def embedding(self, input: str, model=ModelType.embedding) -> List[float]:
        return self.client.embeddings.create(input=[input], model=model).data[0].embedding

    def embeddings(self, inputs: List[str], model=ModelType.embedding) -> Union[List[float], List[List[float]]]:
        return [d.embedding for d in self.client.embeddings.create(input=inputs, model=model).data]

    def _do(self, model_type: ModelType, query_type, messages, user: str, json_response: bool = False) -> str:
        logger.debug(f"{user}: {query_type} model={model_type}")
        
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

    def determine_params(self, model_type: ModelType, messages):
        response = self.client.chat.completions.create(
            model=model_type, 
            messages=messages,
            tools=get_function_template(FunctionTemplate.DETERMINE_PARAMS),
            tool_choice={"type": "function", "function": {"name": "determine_params"}}
        )
        try:
            params = json.loads(json.loads(response.json())['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
        except:
            params = {}
        return params
    
    
    def ask(
            self,
            query_type: str,
            model_type: ModelType,
            messages,
            shadow_models: List[ModelType] = None,
            user=None,
            json_response: bool = False) -> str:
        logger.debug(f"{user}: {query_type} {messages}")
        messages.insert(0, {"role":"system", "content":self.final})
        params = self.shadow_wrapper(model_type, shadow_models, self.determine_params, messages)

        print(params)

        return self.shadow_wrapper(model_type, shadow_models, self._do, query_type, messages, user, json_response)

    def shadow_wrapper(self, model_type: ModelType, shadow_models: List[ModelType], fn, *args):
        f = self.pool.submit(fn, model_type, *args)
        if shadow_models:
            for model in shadow_models:
                self.pool.submit(fn, model, *args)
        return f.result()