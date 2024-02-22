from flask import request, Flask, stream_with_context
from flask_cors import CORS
import logging
import time
import json 
from dotenv import load_dotenv
from openai import OpenAI
import os

import time

from utils.gpt_output_utils import extract_xml_tags
from internal_retriever import CodebaseRetriever
from external_retrieval import PerplexityExternalSearch
from code_executor import process_with_search_replace_blocks, process_with_diffs
from refinement import run_refinement_chain
from utils.codebase import find_closest_file
from utils.prompts import INFORMATION_RETRIEVAL_PROMPT, PLAN_WRITING_PROMPT
from utils.model import create_model

load_dotenv(".env")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
logging.getLogger('flask_cors').level = logging.DEBUG

response, response_time = "", -1
codebase_retriever = CodebaseRetriever("")
external_retriever = PerplexityExternalSearch(os.environ["PERPLEXITY_API_KEY"])

openai_model = create_model(
    os.environ["OPENAI_API_KEY"],
    "gpt-4-0125-preview"
)

plan_model = create_model(
    os.environ["TOGETHER_API_KEY"],
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    base_url="https://api.together.xyz"
)
information_request_model = create_model(
    os.environ["TOGETHER_API_KEY"],
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    base_url="https://api.together.xyz"
) # Make this a decent information request model as well
coding_model = create_model(
    os.environ["TOGETHER_API_KEY"],
    "deepseek-ai/deepseek-coder-33b-instruct",
    base_url="https://api.together.xyz"
)

# plan_model, information_request_model, coding_model = openai_model, openai_model, openai_model

def generate_multifile_response(directory, objective, snippets):
    pass

def generate_response(directory, objective, snippets, verbose=True, use_vectorstore=True, user_input=False):
        global codebase_retriever
        yield json.dumps({"information": "Started processing information!"})
    
        # Perform information retrieval
        if not(codebase_retriever.directory) == directory:
            codebase_retriever = CodebaseRetriever(directory, use_vectorstore=use_vectorstore)
            codebase_retriever.load_all_documents()
            yield json.dumps({"information": "Loading vectorstore search for your codebase."})
        else:
            yield json.dumps({"information": "Existing codebase retriever works."})

        start_time = time.time()
        model_request = information_request_model(INFORMATION_RETRIEVAL_PROMPT,[f"Objective: {objective} \n \n Existing information {snippets}"])
        end_time = time.time()

        if verbose:
            print("Performed Information Retrieval Request: ", (end_time - start_time))

        information = snippets
        
        start_time = time.time()
        for _ in range(1): # Setting a limit to 3 queries at the most. TODO: find a fix to the recursive problem eh
            # With the queries, run the corresponding searches
            external_queries = extract_xml_tags(model_request, "a")
            internal_queries = extract_xml_tags(model_request, "b")
            # file_read_queries = extract_xml_tags(model_request, "f")

            if verbose:
                print("Processing information retrieval request with: ", model_request)
                print(" External queries: ", json.dumps(external_queries, indent=4))
                print(" Internal queries: ", json.dumps(internal_queries, indent=4))
                # print(" File read queries: ", json.dumps(file_read_queries, indent=4))

            for query in external_queries:
                if verbose:
                    print("Processing external query: ", query)
                external_response = external_retriever.answer_question(query)
                if verbose:
                    print("Received response: ", external_response)
                information += f"\n# External Query: {query} \n {external_response}"

            for query in internal_queries:
                code_snippets = codebase_retriever.retrieve_documents(query, None)
                code_snippets_string = "\n".join(code_snippets)
                information += f"\n# Codebase Query: {query} \n {code_snippets_string}"
            
            # for query in file_read_queries:
            #     accurate_file = find_closest_file(directory, query)
            #     contents = open(accurate_file, "r").read()
            #     information += f"\n# File Contents for: {query} \n ```\n{contents}\n```"

            if len(external_queries) == 0 and len(internal_queries) == 0: # and len(file_read_queries) == 0:
                break
        end_time = time.time()

        if verbose:
            print("Performed all information retrieval requests: ", (start_time - end_time))

        # Send back the documents and ask for feedback
        yield json.dumps({"All information": information}, indent=4)
        
        if user_input:
            new_information = wait_for_gui_response(time.time())
            information = new_information

        # (potentially) Rerun the information retrieval process

        # Generate an implementation plan
        start_time = time.time()
        plan = plan_model(
            PLAN_WRITING_PROMPT, 
            [f"""Given the provided information and the objective, write a plan to complete it. 
            Do not write any code. Objective: {objective} \n \n Information: {information}"""]
        )
        end_time = time.time()

        if verbose:
            print("Generated plan in: ", (start_time - end_time))

        # Get feedback
        yield json.dumps({"Feedback: ": plan})

        if user_input:
            plan = wait_for_gui_response(time.time())

        start_time = time.time()
        # Generate diffs
        changes = process_with_diffs(openai_model, directory, f"Objective: {objective} \n \n Information: {information}")
        end_time = time.time()

        if verbose:
            print("Processing initial diff: ", (end_time - start_time))
            print("Initial changes: ", json.dumps(changes, indent=4))

        # Self-refine everything
        start_time = time.time()
        refined_changes = run_refinement_chain(directory, changes, objective, information, openai_model, process_with_diffs)
        end_time = time.time()

        if verbose:
            print("Finishing refinement in time: ", end_time - start_time)

        # Return a set of changes to the user
        yield json.dumps({"changes": refined_changes})

@app.post("/process")
def ask():
    data = request.get_json()
    directory, objective, snippets = data["directory"], data["objective"], data["snippets"]
    return app.response_class(stream_with_context(generate_response(directory, objective, snippets)))

@app.post("/send_response")
def send_response():
    global response, response_time
    data = request.get_json()
    response = data["message"]
    response_time = time.time()
    return {'ok': True}

def wait_for_gui_response(request_time):
    global response, response_time
    while request_time < response_time:
        time.sleep(0.5)
    return response
    
if __name__ == "__main__":
    directory = "/Users/vijaydaita/Files/uiuc/rxassist/rxmind-nextjs-main"
    filepath = "/app/pages/quiz/page.tsx"
    file_snippet = open(directory + filepath).read()
    for value in generate_response(
            directory, 
            "In the main quiz page, add a modal for when the quiz is over that shows the score and allows you to retake the quiz.", 
            f"Filepath: {filepath} \n ```\n{file_snippet}\n```",
            use_vectorstore=False
        ):
        print(json.dumps(json.loads(value), indent=4))
        ret_object = json.loads(value)
        if "changes" in ret_object:
            should_apply = input("Apply to rewrite? Send Y for yes. ")
            if should_apply.strip().lower() == "y":
                for change in ret_object["changes"]:
                    file_snippet.replace(change["original"], change["new"])
        
        print(file_snippet)
    # app.run(port=8123, debug=True)
