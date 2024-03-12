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

logger = logging.getLogger(__name__)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

async def call_chatgpt_async(session, messages, model_name, key): # https://medium.com/@nitin_l/parallel-chatgpt-requests-from-python-6ab48cc2a610
    payload = {
        'model': model_name,
        'messages': messages
    }
    try:
        start_time = time.time()
        async with session.post(
            url='https://api.openai.com/v1/chat/completions',
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json=payload,
            ssl=ssl.create_default_context(cafile=certifi.where())
        ) as response:
            response = await response.json()
        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        end_time = time.time()
        print("Time spent on request: ", (end_time - start_time))
        return response['choices'][0]['message']['content']
    except:
        print("Request failed.")

async def bulk_call(message_sets, model_name, api_key):
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(call_chatgpt_async(session, message_set, model_name, api_key)) for message_set in message_sets]
        responses = await asyncio.gather(*tasks)
    return responses

def create_model(api_key, model_name, base_url="https://api.openai.com/v1", base_temperature=0.2, base_max_tokens=2048): # I don't want to pass the model name in separately
    model = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    def run_model(system_prompt, messages, temperature=base_temperature, max_tokens=base_max_tokens):
        print(len(system_prompt), len(messages))
        if type(system_prompt) == list: # That means that multiple requests should be made
            print("Bulk processing")
            message_sets = []
            for message_index in range(len(system_prompt)):
                new_messages =[
                        {"role": "system", "content": system_prompt[message_index]},
                ] + [{"role": "user", "content": message} for message in messages[message_index]]
                # print(json.dumps(new_messages, indent=4))
                message_sets.append(new_messages)
            results = asyncio.run(bulk_call(message_sets, model_name, api_key))
            return results
        else:
            response = model.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                ] + [ {"role": "user", "content": message} for message in messages],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

    return run_model


def extract_code_block_data(md_text, language):
   # Regular expression pattern for matching diff code blocks
   pattern = rf'```{language}([\s\S]*?)```'
   code_blocks = re.findall(pattern, md_text, re.MULTILINE)
   return code_blocks

def extract_xml_tags(text, tag):
    pattern = r'<' + tag + '>(.*?)</' + tag + '>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches