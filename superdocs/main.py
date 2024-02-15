from flask import request, Flask, stream_with_context
from flask_cors import CORS
import logging
import time
import json 
from dotenv import load_dotenv
from openai import OpenAI
import os

from utils.gpt_output_utils import extract_xml_tags
from internal_retriever import CodebaseRetriever
from external_retrieval import PerplexityExternalSearch
from code_executor import process_with_search_replace_blocks
from refinement import run_refinement_chain

load_dotenv(".env")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
logging.getLogger('flask_cors').level = logging.DEBUG

response, response_time = "", -1
codebase_retriever = CodebaseRetriever("")
external_retriever = PerplexityExternalSearch(os.environ["PERPLEXITY_API_KEY"])

openai_model = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

def simple_request(system_prompt, query, model_name="gpt-4-preview-0116"):
    pass

@app.post("/process")
def ask():
    data = request.get_json()
    directory, objective, snippets = data["directory"], data["objective"], data["snippets"]
    def generate_response():
        global codebase_retriever
        yield json.dumps({"information": "Started processing information!"})
    
        # Perform information retrieval
        if not(codebase_retriever.directory) == directory:
            codebase_retriever = CodebaseRetriever(directory)
            codebase_retriever.load_all_documents()
            yield json.dumps({"information": "Loading vectorstore search for your codebase."})
        else:
            yield json.dumps({"information": "Existing codebase retriever works."})

        model_request = simple_request(INFORMATION_RETRIEVAL_PROMPT, f"Objective: {objective} \n \n Existing information {snippets}")
        
        # With the queries, run the corresponding searches
        external_queries = extract_xml_tags(model_request, "e")
        internal_queries = extract_xml_tags(model_request, "i")

        new_information = snippets
        for query in external_queries:
            new_information += f"\n# External Query: {query} \n {external_retriever.answer_question(query)}"

        for query in internal_queries:
            code_snippets = codebase_retriever.retrieve_documents(query, None)
            code_snippets_string = "\n".join(code_snippets)
            new_information += f"\n# Codebase Query: {query} \n {code_snippets_string}"
        

        # Send back the documents and ask for feedback
        yield json.dumps({"Extracted information": new_information})

        information += new_information

        # (potentially) Rerun the information retrieval process

        # Generate an implementation plan
        plan = simple_request(
            PLAN_WRITING_PROMPT, 
            f"""Given the provided information and the objective, write a plan to complete it. 
            Do not write any code. Objective: {objective} \n \n Information: {information}"""
        )

        # Get feedback
        yield json.dumps({"Feedback: ", plan})

        # Generate diffs
        changes = process_with_search_replace_blocks(openai_model, directory, f"Objective: {objective} \n \n Information: {information}")

        # Self-refine everything
        refined_changes = run_refinement_chain(directory, changes, objective, information, openai_model, process_with_search_replace_blocks)

        # Return a set of changes to the user
        return refined_changes
    return app.response_class(stream_with_context(generate_response()))

@app.post("/send_response")
def send_response():
    global response, response_time
    data = request.get_json()
    response = data["message"]
    response_time = time.time()
    return {'ok': True}

def wait_for_response(request_time):
    global response, response_time
    while request_time < response_time:
        return response
    
INFORMATION_RETRIEVAL_PROMPT = """
You are a development assistant, responsible for finding and requesting information to solve the objective.
From the provided query and existing context, you are responsible for determining what kind of further information should be gathered.
To request further information, you can use the following four tags:
I queries are for searching for content within the user's current codebase, such as asking for where specific method definitions are or where are specific pieces of code that complete certain functionality: <i>query</i>
E queries use Google for retrieval external API documentation, tutorials for solving coding problems not within the database, consulting externally for errors, finding tools to use, etc.: <e>query</e>
Add as much context, such as programming language or framework when making requests.
Complete all the requests you think you need at one go.
Think step-by-step.

Your first step should be to identify all the relevant libraries that are being used by the program (such as UI libraries, networking libraries, etc.).
Your second step is to identify the queries you want.
Your third step is to identify, for each query, whether or not it will be an I or E query (state why).

Do not write any code planning or coding suggestions under any circumstances.
You can provide multiple queries at one go.

# Example conversation 1

## USER: Objective: Write a script that pulls images of buzzcuts from google images
Code: # Code written in Python or in a .py file...

## ASSISTANT: <e>Python libraries for downloading google images</e> <e>Python script for downloading images from google</e>

# Example conversation 2

## USER: Objective: Find where in the code do I make network requests to the GitHub api
## ASSISTANT: <i>network requests, GitHub</i>
"""

PLAN_WRITING_PROMPT = """
Given the following context and the user's objective, create a plan for modifying the codebase and running commands to solve the objective.
Create a step-by-step plan to accomplish these objectives without writing any code. First, write an explanation of each chunk of code that needs to be edited.
The plan executor can only: replace content in files and provide code instructions to the user. 
Under each command, write subinstructions that break down the solution so that the code executor can write the code.
Make your plan as concise as possible.

PLEASE DO NOT WRITE ANY CODE YOURSELF.

Let's think step by step.
"""