from dotenv import load_dotenv
import os
import typer
from typing import Optional
from typing_extensions import Annotated
import re

import time

from .retriever import CodebaseRetriever, SearchRetriever
from .utils.prompts import INFORMATION_RETRIEVAL_PROMPT
from .utils.model import create_model, extract_code_block_data, extract_xml_tags
from .code_executor import Executor
from . import code_executor

load_dotenv(".env")
codebase = CodebaseRetriever("")
openai_model = None
files = {}

app = typer.Typer()

@app.command("run")
def main(directory: str, model_name: Annotated[Optional[str], typer.Argument()] = "gpt-3.5-turbo", api_key: Annotated[str, typer.Argument(envvar="OPENAI_API_KEY")] = None):
    global search_retriever, codebase, files, openai_model

    codebase = CodebaseRetriever(directory)
    def extract_text_within_single_quotes(text):
        return re.findall(r"'(.*?)'", text)

    if not(api_key):
        typer.echo("You must have an OpenAI API key loaded in your environment variables as OPENAI_API_KEY.")
    
    openai_model = create_model(api_key, model_name)
    search_retriever = SearchRetriever(openai_model)

    extracted_filenames = []

    while True:
        command = typer.prompt("What would you like to do next? ('add' to add file, 'run' to make an edit, 'exit' to exit)")
        if "add" in command.lower():
            add_files = typer.prompt("Copy filepaths (filenames that are within single-quotes will be considered, like from VSCode drag-and-drop): ")
            filenames = extract_text_within_single_quotes(add_files)
            new_filenames = []
            for filename in filenames:
                shortened_filename = filename.replace(directory, "")
                extracted_filenames.append(shortened_filename)
            extracted_filenames.extend(new_filenames)
            typer.echo("Finished adding new filenames to the list.")
        elif "run" in command.lower():
            files = {}
            for rel_filepath in extracted_filenames:
                filepath = os.path.join(directory, rel_filepath)
                try:
                    files[rel_filepath] = open(filepath, "r").read()
                except Exception:
                    typer.echo("Error loading file: ", filepath)
            
            goal = typer.prompt("What objective would you like to run?")

            information_request = openai_model(INFORMATION_RETRIEVAL_PROMPT, messages=[f"Objective: {goal}", f"Files: {code_executor.stringify_files(files)}"])
            internal_requests = extract_xml_tags(information_request, "a")
            external_requests = extract_xml_tags(information_request, "b")

            typer.echo("Internal requests:", internal_requests)
            typer.echo("External requests:", external_requests)


            context = ""
            for request in internal_requests:
                retrieved = codebase.retrieve_documents(request)
                retrieved_filename = [chunk.splitlines()[0].replace("Filepath: ", "").strip() for chunk in retrieved]

                new_snippets = []

                for index, retrieved in enumerate(retrieved):
                    if not(retrieved_filename[index] in files):
                        new_snippets.append(retrieved[0])
                snippets = "\n".join(codebase.retrieve_documents(request))
                context += f"------- \n # Snippets for query {request} \n {snippets}"
                
            for request in external_requests:
                search_response = search_retriever.search(request)
                context += f"------- \n # Answer for request {request} \n {search_response}"

            # print("Context: ", context)

            executor = Executor(goal, files, context, openai_model)
            modifications = executor.chain_execute()
            
            typer.echo("MODIFICATIONS")

            for filepath in modifications["annotated"]:
                typer.echo("Filepath: ", filepath)
                typer.echo(modifications["annotated"][filepath])

            save = typer.confirm("Do you want to save these changes?")

            if save:
                for filepath in modifications["unannotated"]:
                    file = open(os.path.join(directory, filepath), "w")
                    file.write(modifications[filepath])
                    file.close()
            else:
                typer.echo("Not saving changes")
        elif "exit" in command:
            break

    # Perform the relevant information request queries

# if __name__ == "__main__":
#     typer.run(main)