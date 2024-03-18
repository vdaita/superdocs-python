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
from .utils import diff_utils

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
def main(
        goal: str,
        filepaths: str,
        search: bool = False, 
        model_name: str ="gpt-3.5-turbo",
        aux_model_name: str="gpt-3.5-turbo",
        use_absolute_filepath: bool = False,
        api_key: Annotated[str, typer.Argument(envvar="OPENAI_API_KEY")] = None, 
        base_url: Annotated[str, typer.Argument(envvar="OPENAI_BASE_URL")] = "https://api.openai.com/v1/"
        ):
    """
    Try running a command with LLMs and directly editing files.
    goal describes the operation you want to run on your files
    filepaths should be a space or comma separated list of files. Each filepath should be enclosed within single quotes
    model_name is the name of the main model that performs planning and executing
    aux_model_name is the name of the auxiliary model (right now mostly used for search)
    use_absolute_filepath should be used if you are inputting the absolute filepaths for each of the files (drag-and-drop from VSCode) 
    api_key and base_url must be defined as environment variables
    By default, base_url points to OpenAI's endpoint
    """
    global search_retriever, codebase, files

    logging.basicConfig(filename="superdocs.log", filemode="w", level=logging.DEBUG)
    logging.info("Logging from main function")

    directory = os.getcwd()
    typer.echo(f"Current working directory: {directory}")
    codebase = CodebaseRetriever(directory)
    def extract_text_within_single_quotes(text):
        return re.findall(r"'(.*?)'", text)

    if not(api_key):
        print("[bold red]You must have an API key loaded in your environment variables as OPENAI_API_KEY.[/bold red]")
        return
    
    model = create_model(os.environ["OPENAI_API_KEY"], model_name, base_url=base_url, base_temperature=1, base_max_tokens=3092)
    aux_model = create_model(os.environ["OPENAI_API_KEY"], aux_model_name, base_url=base_url, base_temperature=1, base_max_tokens=2048)
    search_retriever = SearchRetriever(model)

    extracted_filenames = []
    filepaths = extract_text_within_single_quotes(filepaths)
    for filepath in filepaths:
        if use_absolute_filepath:
            shortened_filename = os.path.relpath(filepath, directory)
        else:
            shortened_filename = filepath
        extracted_filenames.append(shortened_filename)

    files = {}
    for rel_filepath in extracted_filenames:
        logging.info("Current directory: " + directory)
        logging.info("Loading file at path: " + os.path.join(directory, rel_filepath))
        try:
            files[rel_filepath] = open(os.path.join(directory, rel_filepath), "r").read()
        except Exception:
            print(f"[bold red]Error loading file: {rel_filepath}[/bold red]")
    
    start_time = time.time()

    google_request = model(f"You are a helpful and intelligent AI assistant.", messages=[f"# Files: \n {code_executor.stringify_files(files)} \n \n # Goal \n {goal} \n Write a Google search query that would be helpful in finding information to achieve the goal."], max_tokens=30)
    remove_chars = ['"', "'"]
    for char in remove_chars:
        google_request = google_request.replace(char, "")

    context = ""
    if search:
        search_response = search_retriever.search(google_request)
        context = f"# Answer for request {google_request} \n {search_response}"

    executor = Executor(goal, files, context, model, aux_model=aux_model)
    modifications = executor.chain_plan_and_execute_lats()
    end_time = time.time()
    print(f"[bold red]Completed in {end_time - start_time} seconds.[/bold red]")

    for filepath in modifications:
        filediff = difflib.unified_diff(files[filepath].splitlines(), modifications[filepath].splitlines())
        filediff = "\n".join(list(filediff))
        search_replace_blocks = diff_utils.find_hunks(filediff)
        
        for block in search_replace_blocks:
            print(f"[bold blue]{block.filepath}[/bold blue]")
            print(f"[red]{block.search_block}[/red]")
            print(f"[green]{block.replace_block}[/green]")
            accept = Prompt.ask("[bold red]Should this change be accepted? (y/N)[/bold red]")
            if accept.lower().strip() == "y":
                file = open(os.path.join(directory, filepath), "w")
                files[filepath] = files[filepath].replace(block.search_block, block.replace_block)
                file.write(files[filepath])
                file.close()                
    # Perform the relevant information request queries

# if __name__ == "__main__":
#     typer.run(main)