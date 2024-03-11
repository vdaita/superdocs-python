from flask import request, Flask, stream_with_context
from flask_cors import CORS
import logging
import time
import json 
from dotenv import load_dotenv
from openai import OpenAI
import os

import time

from retriever import CodebaseRetriever
from utils.prompts import INFORMATION_RETRIEVAL_PROMPT
from utils.model import create_model, extract_code_block_data, extract_xml_tags
from code_executor import Executor
import code_executor 

load_dotenv(".env")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
logging.getLogger('flask_cors').level = logging.DEBUG

response, most_recent_response_time = "", -1
codebase_retriever = CodebaseRetriever("")

openai_model = create_model(
    os.environ["OPENAI_API_KEY"],
    "gpt-3.5-turbo"
)

plan_model = create_model(
    os.environ["TOGETHER_API_KEY"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_url="https://api.together.xyz"
)
information_request_model = create_model(
    os.environ["TOGETHER_API_KEY"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_url="https://api.together.xyz"
) # Make this a decent information request model as well
coding_model = create_model(
    os.environ["TOGETHER_API_KEY"],
    "deepseek-ai/deepseek-coder-33b-instruct",
    base_url="https://api.together.xyz"
)

perplexity_model = create_model(
    os.environ["PERPLEXITY_API_KEY"],
    "sonar-small-online",
    base_url="https://api.perplexity.ai"
)

@app.post("/process")
def ask():
    data = request.get_json()
    directory, objective, snippets = data["directory"], data["objective"], data["snippets"]
    return app.response_class(stream_with_context(generate_response(directory, objective, snippets, user_input=False)))

@app.post("/send_response")
def send_response():
    global response, most_recent_response_time
    data = request.get_json()
    print("Send_response received input: ", data)
    response = data["message"]
    most_recent_response_time = time.time()
    return {'ok': True}

def wait_for_gui_response(request_time):
    global response, response_time
    while most_recent_response_time < request_time:
        print("     No response received: waiting for GUI response")
        time.sleep(0.5)
    return response

# if __name__ == "__main__":
#     app.run(port=8125, debug=True)
    
if __name__ == "__main__":
    directory = "/Users/vijaydaita/Files/uiuc/rxassist/rxmind-nextjs-main"
    filepath = "/app/pages/quiz/page.tsx"
    goal = "When the quiz finishes, add a modal that allows the user to refresh the webpage and displays the score."
    files = {filepath: open(directory + filepath, "r").read()}
    codebase = CodebaseRetriever(directory)
    codebase.load_all_documents()

    information_request = openai_model(INFORMATION_RETRIEVAL_PROMPT, messages=[f"Objective: {goal}", f"Files: {code_executor.stringify_files(files)}"])
    internal_requests = extract_xml_tags(information_request, "a")
    external_requests = extract_xml_tags(information_request, "b")

    print("Internal requests:", internal_requests)
    print("External requests:", external_requests)
    
    # Perform the relevant information request queries

    context = ""
    for request in internal_requests:
        retrieved = codebase.retrieve_documents(request)
        retrieved_filename = [chunk.splitlines()[0].replace("Filepath: ", "").strip() for chunk in retrieved]

        new_snippets = []

        for index, retrieved in enumerate(retrieved):
            if not(retrieved_filename[index] in files):
                new_snippets.append(retrieved[0])
        snippets = "\n".join(codebase.retrieve_documents(request))
        context += f"# Snippets for query {request} \n {snippets}"
        
    for request in external_requests:
        perplexity_response = perplexity_model("Write a concise summary that would be helpful for a software developer to implement a fix or feature.", [f"Query: {request}"])
        context += f"# Answer for request {request} \n {perplexity_response}"

    print("Context: ", context)

    executor = Executor(goal, files, context, openai_model)
    executor.chain_execute()