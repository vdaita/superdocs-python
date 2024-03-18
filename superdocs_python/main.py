import logging
logging.basicConfig(filename="superdocs.log", filemode="w", level=logging.DEBUG)
logging.info("Logging from main.py")

from dotenv import load_dotenv
import os
import typer
from typing import Optional
from typing_extensions import Annotated
import re
import difflib

import time

from .retriever import CodebaseRetriever, SearchRetriever
from .utils.model import create_model, extract_code_block_data, extract_xml_tags
from .code_executor import Executor
from . import code_executor

from rich import print
from rich.prompt import Prompt

load_dotenv(".env")
codebase = CodebaseRetriever("")
files = {}

app = typer.Typer()

@app.command("run")
def main(api_key: Annotated[str, typer.Argument(envvar="TOGETHER_API_KEY")] = None):
    global search_retriever, codebase, files

    logging.basicConfig(filename="superdocs.log", filemode="w", level=logging.DEBUG)
    logging.info("Logging from main function")

    directory = os.getcwd()
    typer.echo(f"Current working directory: {directory}")
    codebase = CodebaseRetriever(directory)
    def extract_text_within_single_quotes(text):
        return re.findall(r"'(.*?)'", text)

    if not(api_key):
        print("[bold red]You must have an Together API key loaded in your environment variables as TOGETHER_API_KEY.[/bold red]")
        return
    
    model = create_model(os.environ["TOGETHER_API_KEY"], "mistralai/Mixtral-8x7B-Instruct-v0.1", base_url="https://api.together.xyz", base_temperature=0.1, base_max_tokens=1024)
    aux_model = create_model(os.environ["TOGETHER_API_KEY"], "mistralai/Mixtral-8x7B-Instruct-v0.1", base_url="https://api.together.xyz", base_temperature=1, base_max_tokens=3096)
    search_retriever = SearchRetriever(model)

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

            google_request = model(f"You are a helpful and intelligent AI assistant.", messages=[f"# Files: \n {code_executor.stringify_files(files)} \n \n # Goal \n {goal} \n Write a Google search query that would be helpful in finding information to achieve the goal."], max_tokens=30)
            remove_chars = ['"', "'"]
            for char in remove_chars:
                google_request = google_request.replace(char, "")

            search_response = search_retriever.search(google_request)
            context = f"# Answer for request {google_request} \n {search_response}"

            # print("Context: ", context)

            executor = Executor(goal, files, context, model, aux_model=aux_model)
            modifications = executor.chain_execute_rewrite()
            end_time = time.time()
            print(f"[bold red]Completed in {end_time - start_time} seconds.[/bold red]")

            d = difflib.Differ()

            for filepath in modifications:
                filediff = d.compare(files[filepath], modifications[filepath])
                print(f"Filepath: {filepath}")

                only_relevant = []
                for line in filediff:
                    if line.startswith("+") or line.startswith("-") or line.startswith("?"):
                        only_relevant.append(line)
                print("\n".join(only_relevant))

                accept = Prompt.ask("[bold red]Should this change be accepted? (y/N)[/bold red]")
                if accept.lower().strip() == "y":
                    file = open(os.path.join(directory, filepath), "w")
                    file.write(modifications[filepath])
                    file.close()
        elif "exit" in command:
            print("f[bold red]Exiting Superdocs[/bold red]")
            break

    # Perform the relevant information request queries

# if __name__ == "__main__":
#     typer.run(main)