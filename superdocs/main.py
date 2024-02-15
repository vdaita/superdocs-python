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
"""

PLAN_WRITING_PROMPT = """
"""