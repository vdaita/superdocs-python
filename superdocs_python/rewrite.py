"""
This file uses GPT-4 (or some really large model) to draw up a quick, sequential list of changes that should be performed.
After that, the changes are executed sequentially by custom inference.
"""

from abc import abstractmethod, ABC
from utils.model import Model, Message
from utils.compressor import Compressor
from typing import List, Optional, Tuple
from rich import print
from utils.timing import timify
import copy
from rapidfuzz import fuzz

class Rewriter(ABC):
    @abstractmethod
    def rewrite(self, problem_statement: str, files: List[Tuple[str, str]]) -> str:
        pass

def files_list_to_str(files: List[Tuple[str, str]], important_index: int) -> str:
    files_str = ""
    for i in range(len(files)):
        if i != important_index:
            files_str += f"# {files[i][0]}\n{files[i][1]}\n"
    files_str += "# You will be editing the following file:\n"
    files_str += f"# {files[important_index][0]}\n{files[important_index][1]}\n"
    return files_str

class PlanAndExecuteRewriter(Rewriter):
    def __init__(self, plan_model: Model, rewrite_model: Model, compressor: Compressor):
        self.plan_model = plan_model
        self.rewrite_model = rewrite_model
        self.compressor = compressor

    def rewrite(self, problem_statement: str, files: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # Plan the changes
        joined_files = '\n'.join([f"# {filename}\ncontent" for filename, content in files])
        time_taken, joined_files = timify(self.compressor.compress(joined_files))
        print(f"[blue]Time taken to compress files: {time_taken} seconds[/blue]")
        time_taken, plan = timify(self.plan_model.run([Message(role='system', content=problem_statement), Message(role='user', content=f"{joined_files}\n{problem_statement}")]).json_data)
        for change in plan:
            print(f"# File: {change['file']}\n{change['description']}")
        print(f"[blue]Plan generated for editing files in {time_taken} seconds.[/blue]")
        # Execute the changes
        modified_files = copy.deepcopy(files)
        was_modified = {} # a dictionary between the file name and whether it was modified
        for change in plan:
            change_file = change['file']
            change_content = change['content']
            similarities = [fuzz.ratio(change_file, filename) for filename, _ in modified_files]
            closest_file_index = similarities.index(max(similarities))
            closest_file, file_contents = modified_files[closest_file_index]
            
            rewrite_instruction_content = f"{files_list_to_str(modified_files, closest_file_index)}\n# Implement the following change: {change_content}"
            rewritten_file = self.rewrite_model.run([Message(role='user', content=rewrite_instruction_content)]).content
            modified_files[closest_file_index] = (closest_file, rewritten_file)
            was_modified[closest_file] = True
            print(f"[green]Applying changes to the closest file: {closest_file}[/green]")
        return modified_files, was_modified