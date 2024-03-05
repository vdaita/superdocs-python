from openai import OpenAI

def create_model(api_key, model_name, base_url="https://api.openai.com/v1", base_temperature=0.1, base_max_tokens=2048): # I don't want to pass the model name in separately
    model = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    def run_model(system_prompt, messages, temperature=base_temperature, max_tokens=base_max_tokens):
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