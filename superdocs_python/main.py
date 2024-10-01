import logging
logging.basicConfig(filename="superdocs.log", filemode="w", level=logging.DEBUG)
logging.info("Logging from main.py")

import os
import typer
from typing import Optional
from typing_extensions import Annotated
import re
import difflib

import time
import json

from rich import print
from rich.prompt import Prompt

from utils.timing import timify
from utils.model import OpenAIModel, LocalHostedModel
from utils.compressor import LLMLinguaCompressor
from utils.diff_utils import parse_diff
from rewrite import PlanAndExecuteRewriter
from fl import LLMFileLocator
import os

files = {}

app = typer.Typer()

@app.command("run")
def main(
        goal: str = typer.Argument(..., help="The operation you want to run on your files"),
        directory: Optional[str] = typer.Option(".", help="The directory where the files are located. Defaults to being the current directory."),
        goal_is_file: Optional[bool] = typer.Argument("", help="Whether the goal is a file or a string"),
    ):
    """
    Main function that runs the CLI. More options should become available as this progresses.
    """
    start_time = time.time()
    if goal_is_file:
        with open(goal, "r") as f:
            goal = f.read()

    model = OpenAIModel(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o")
    rewrite_model = LocalHostedModel(url="http://localhost:8001")
    compressor = LLMLinguaCompressor(use_small=True, compress_rate=0.33)

    file_locator = LLMFileLocator()
    
    # Find the files that are required to fix the problem
    time_taken, files = timify(file_locator.extract_top_files(directory, goal))
    print(f"[blue]Found the following files in {time_taken} seconds:[/blue]")
    print(f"[green]{'\n'.join(files)}[/green]")

    # Create a plan and rewrite the relevant files
    rewriter = PlanAndExecuteRewriter(plan_model=model, rewrite_model=rewrite_model, compressor=compressor)
    time_taken, (modified_files, was_modified) = timify(rewriter.rewrite(goal, files))

    # Tell the user that the changes are done, list the files that were modified, show a diff, and create a search replace block to be saved to the files
    for (filename, file_contents) in modified_files:
        if was_modified[filename]:
            print(f"[green]Changes were applied to {filename}[/green]")
            print(f"[red]Diff:[/red]")
            diff = "\n".join(difflib.unified_diff(files[filename].splitlines(), file_contents.splitlines(), n=5))
            print(diff)
            search_replace_blocks = parse_diff(diff)
            text = file_contents
            for sr in search_replace_blocks:
                text = text.replace(sr.search_block, f"""<<<<<<< SEARCH
{sr.search_block}
=======
{sr.replace_block.rstrip()}
>>>>>>> REPLACE""")
            file = open(filename, "w")
            file.write(text)
            file.close()

    print(f"[green]Completed changes in {time.time() - start_time}[/green]")

# if __name__ == "__main__":
#     typer.run(main)
