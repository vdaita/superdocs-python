import logging
logging.basicConfig(filename="superdocs.log", filemode="w", level=logging.DEBUG)
logging.info("Logging from main.py")

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

from rich import print
from rich.prompt import Prompt

load_dotenv(".env")
codebase = CodebaseRetriever("")
large_model = None
small_model = None
files = {}

app = typer.Typer()

@app.command("run")
def main(api_key: Annotated[str, typer.Argument(envvar="OPENAI_API_KEY")] = None):
    global search_retriever, codebase, files, large_model, small_model

    logging.basicConfig(filename="superdocs.log", filemode="w", level=logging.DEBUG)
    logging.info("Logging from main function")

    directory = os.getcwd()
    typer.echo(f"Current working directory: {directory}")
    codebase = CodebaseRetriever(directory)
    def extract_text_within_single_quotes(text):
        return re.findall(r"'(.*?)'", text)

    if not(api_key):
        print("[bold red]You must have an OpenAI API key loaded in your environment variables as OPENAI_API_KEY.[/bold red]")
    
    large_model = create_model(api_key, "gpt-4-turbo-preview")
    small_model = create_model(api_key, "gpt-3.5-turbo")
    search_retriever = SearchRetriever(small_model)

    extracted_filenames = []

    while True:
        command = Prompt.ask("[bold red]What would you like to do next? ('add' to add file, 'run' to make an edit, 'exit' to exit)[/bold red]")
        if "add" in command.lower():
            add_files = Prompt.ask("[bold red]Copy filepaths (filenames that are within single-quotes will be considered, like from VSCode drag-and-drop): [/bold red]")
            filenames = extract_text_within_single_quotes(add_files)
            new_filenames = []
            for filepath in filenames:
                shortened_filename = os.path.relpath(filepath, directory)
                extracted_filenames.append(shortened_filename)
            extracted_filenames.extend(new_filenames)
            print("[bold red]Finished adding new filenames to the list.[/bold red]")
        elif "run" in command.lower():
            files = {}
            for rel_filepath in extracted_filenames:
                logging.info("Current directory: " + directory)
                logging.info("Loading file at path: " + os.path.join(directory, rel_filepath))
                try:
                    files[rel_filepath] = open(os.path.join(directory, rel_filepath), "r").read()
                except Exception:
                    print(f"[bold red]Error loading file: {rel_filepath}[/bold red]")
            
            goal = Prompt.ask("[bold red]What objective would you like to run?[/bold red]")
            start_time = time.time()

            information_request = large_model(INFORMATION_RETRIEVAL_PROMPT, messages=[f"Objective: {goal}", f"Files: {code_executor.stringify_files(files)}"])
            internal_requests = extract_xml_tags(information_request, "a")
            external_requests = extract_xml_tags(information_request, "b")

            print(f"[bold red]Internal requests: {internal_requests}[/bold red]")
            print(f"[bold red]External requests: {external_requests}[/bold red]")

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

            executor = Executor(goal, files, context, large_model, small_model)
            modifications = executor.execute()
            executor.files = modifications
            end_time = time.time()
            print(f"[bold red]Completed in {end_time - start_time} seconds.[/bold red]")

            for filepath in modifications["unannotated"]:
                file = open(os.path.join(directory, filepath), "w")
                file.write(modifications["unannotated"][filepath])
                file.close()
        elif "exit" in command:
            print("f[bold red]Exiting Superdocs[/bold red]")
            break

    # Perform the relevant information request queries

# if __name__ == "__main__":
#     typer.run(main)