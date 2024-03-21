from openai import OpenAI
import re
import json
import logging
import tiktoken
import time

import asyncio 
import aiohttp
import ssl
import certifi

from pydantic import BaseModel

logger = logging.getLogger(__name__)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

async def call_chatgpt_async(session, messages, model_name, key, endpoint, max_tokens): # https://medium.com/@nitin_l/parallel-chatgpt-requests-from-python-6ab48cc2a610
    payload = {
        'model': model_name,
        'messages': messages,
        'max_tokens': max_tokens
    }
    try:
        start_time = time.time()
        async with session.post(
            url=f'{endpoint}/chat/completions',
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json=payload,
            ssl=ssl.create_default_context(cafile=certifi.where())
        ) as response:
            response = await response.json()
        if "error" in response:
            logging.error(f"OpenAI request failed with error {response['error']}")
        end_time = time.time()
        response_content = response['choices'][0]['message']['content']
        logging.info(f"Time spent on request: {end_time - start_time}, {len(encoding.encode(response_content))} tokens generated.")
        return response_content
    except:
        print("Request failed.")

async def bulk_call(message_sets, model_name, api_key, base_url, max_tokens):
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(call_chatgpt_async(session, message_set, model_name, api_key, base_url, max_tokens)) for message_set in message_sets]
        responses = await asyncio.gather(*tasks)
    return responses

def message_fmt(message):
    if type(message) == str:
        return {"role": "user", "content": message}
    elif type(message) == tuple:
        return {"role": message[0], "content": message[1]}
    else:
        return message

def create_model(api_key, model_name, base_url="https://api.openai.com/v1", temperature=0.2, max_tokens=2048): # I don't want to pass the model name in separately
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    class Model:
        def __init__(self):
            pass

        def get_direct_client(self):
            return client
        
        def call_with_json(self, system_prompt, messages, schema: BaseModel):
            start_time = time.time()
            logger.info("Calling with json")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt}
                ] + [message_fmt(message) for message in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object", "schema": schema.model_json_schema()}
            )
            end_time = time.time()
            logger.info(f"Time taken: {end_time - start_time}")
            # logger.info(f"Model response:\n {response.choices[0].message.model_dump_json()}")
            return json.loads(response.choices[0].message.content)

        def call_with_tools(self, system_prompt, messages, tools, tool_choice=None):
            start_time = time.time()
            logger.info("Calling with tools")

            if tool_choice:
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            else:
                tool_choice = "auto"

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt}
                ] + [message_fmt(message) for message in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice
            )
            end_time = time.time()
            logger.info(f"Time taken: {end_time - start_time}")
            # logger.info(f"Model response:\n {response.choices[0].message.model_dump_json()}")
            return response.choices[0].message.model_dump()['tool_calls']

        def __call__(self, system_prompt, messages, max_tokens=max_tokens):
            logger.info(f"Sending model request")
            if type(system_prompt) == list: # That means that multiple requests should be made
                logger.info("Bulk processing")
                message_sets = []
                for message_index in range(len(system_prompt)):
                    new_messages =[
                            {"role": "system", "content": system_prompt[message_index]},
                    ] + [message_fmt(message) for message in messages[message_index]]
                    # print(json.dumps(new_messages, indent=4))
                    message_sets.append(new_messages)
                results = asyncio.run(bulk_call(message_sets, model_name, api_key, base_url, max_tokens))
                return results
            else:
                start_time = time.time()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                    ] + [message_fmt(message) for message in messages],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                response_content = response.choices[0].message.content
                end_time = time.time()
                logger.info(f"Time taken: {end_time - start_time}, {len(encoding.encode(response_content))} tokens generated.")
                return response_content

    model = Model()
    return model


def extract_code_block_data(md_text, language):
   # Regular expression pattern for matching diff code blocks
   pattern = rf'```{language}([\s\S]*?)```'
   code_blocks = re.findall(pattern, md_text, re.MULTILINE)
   return code_blocks

def extract_xml_tags(text, tag):
    pattern = r'<' + tag + '>(.*?)</' + tag + '>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches