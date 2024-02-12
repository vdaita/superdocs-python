from readability import Document
from markdownify import markdownity as md
from googlesearch import search
import requests
from openai import OpenAI

class PerplexityExternalSearch:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://api.perplexity.ai",
            api_key=self.api_key
        )

    def answer_question(self, question):
        response = self.client.chat.completions.create(
            model="pplx-7b-online",
            messages=[
                {
                    "role": "system",
                    "content": "You are an information gatherer for an automated coding bot."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        return response.choices[0].message.content

class MetaphorExternalSearch:
    def __init__(self, api_key, condenser):
        self.api_key = api_key
        self.condenser = condenser

    def answer_question(self, question):
        pass

class GoogleExternalSearch:
    def __init__(self, condenser):
        self.condenser = condenser
    
    def answer_question(self, question):
        results = search(question, num_results=2, advanced=True, timeout=5)
        combined_text = ""
        for result in results:
            response = requests.get(result.url)
            doc = Document(response.text)
            html_summary = doc.summary()
            combined_text += "\n\n" + md(html_summary)

        return self.condenser(question, combined_text)