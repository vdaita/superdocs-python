from readability import Document
from markdownify import markdownity as md
from googlesearch import search
import requests

class PerplexityExternalSearch:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def answer_question():
        pass

class MetaphorExternalSearch:
    def __init__(self, api_key, condenser):
        self.api_key = api_key
        self.condenser = condenser

def get_data_from_metaphor():
    pass

def get_data_from_google(question: str, condense_function):
    results = search(question, num_results=2, advanced=True, timeout=5)
    combined_text = ""
    for result in results:
        response = requests.get(result.url)
        doc = Document(response.text)
        html_summary = doc.summary()
        combined_text += "\n\n" + md(html_summary)

    return condense_function(question, combined_text)