from openai import OpenAI
import re

def create_model(api_key, model_name, base_url="https://api.openai.com/v1", base_temperature=0.1, base_max_tokens=2048): # I don't want to pass the model name in separately
    model = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    def run_model(system_prompt, messages, temperature=base_temperature, max_tokens=base_max_tokens):
        print(len(system_prompt), len(messages))
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