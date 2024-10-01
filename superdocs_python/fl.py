from utils import Model, Message
import os
import subprocess
from abc import abstractmethod, ABC
from typing import List, Dict
from rapidfuzz import process, fuzz
import tree_sitter
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language

obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path and return at most 5 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```
"""

class FileLocator(ABC):
    def extract_top_files(self, directory: str, problem_statement: str):
        pass

class LLMFileLocator(FileLocator):
    def __init__(self,
                 model: Model):
        self.model = model
    
    def _read_all_files(self, directory):
        # Get the list of all files in the directory
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                all_files.append(os.path.join(root, file))

        # Use git to check which files are not ignored
        result = subprocess.run(['git', 'check-ignore'] + all_files, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        
        ignored_files = set(result.stdout.splitlines())
        
        # Filter out the ignored files
        non_ignored_files = [file for file in all_files if file not in ignored_files]
        
        return non_ignored_files
    
    def _extract_classes_and_methods(self, filename: str, content: str) -> Dict[str, List[str]]:
        # Determine the language of the file based on its extension
        file_extension = os.path.splitext(filename)[1]
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rb': 'ruby',
            '.rs': 'rust',
            '.php': 'php',
            '.ts': 'typescript',
            # Add more mappings as needed
        }
        
        language_name = language_map.get(file_extension, 'python')  # Default to Python
        language = get_language(language_name)
        
        # Initialize the parser
        parser = Parser()
        parser.set_language(language)

        # Parse the content
        tree = parser.parse(bytes(content, "utf8"))
        root_node = tree.root_node

        # Traverse the tree to extract classes and methods
        class_methods = {}

        def traverse(node, current_class=None):
            if node.type == 'class_definition':
                class_name = node.child_by_field_name('name').text.decode('utf-8')
                class_methods[class_name] = []
                current_class = class_name
            elif node.type == 'function_definition' and current_class:
                method_name = node.child_by_field_name('name').text.decode('utf-8')
                class_methods[current_class].append(method_name)

            for child in node.children:
                traverse(child, current_class)

        traverse(root_node)
        return class_methods
    
    def _parse_llm_response(self, response: str) -> List[str]:
        # Extract the code block containing the filenames
        start_marker = "```"
        end_marker = "```"
        
        start_index = response.find(start_marker)
        end_index = response.find(end_marker, start_index + len(start_marker))
        
        if start_index == -1 or end_index == -1:
            return []
        
        # Extract the filenames from the code block
        filenames_block = response[start_index + len(start_marker):end_index].strip()
        filenames = filenames_block.split('\n')
        
        return filenames

    def _map_file_names(self, generated_filenames: List[str], actual_filenames: List[str]):
        # Map the generated filenames to the actual filenames in the repository using rapidfuzz
        mapped_filenames = []
        for generated_filename in generated_filenames:
            match = process.extractOne(generated_filename, actual_filenames, scorer=fuzz.ratio)
            if match:
                mapped_filenames.append(match[0])
        return mapped_filenames
    
    def _generate_repo_tree(self, filenames: List[str]) -> str:
        tree = {}
        
        for filepath in filenames:
            parts = filepath.split(os.sep)
            current_level = tree
            
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
        
        def build_tree_str(current_level, indent="\t"):
            tree_str = ""
            for key, subtree in current_level.items():
                tree_str += f"{indent}{key}/\n"
                tree_str += build_tree_str(subtree, indent + "    ")
            return tree_str
        
        return build_tree_str(tree)


    def extract_top_files(self, directory: str, problem_statement: str):
        # Step 1: Read all non-ignored files
        all_files = self._read_all_files(directory)
        
        # Step 2: Generate repository structure
        repo_structure = self._generate_repo_tree(all_files)
        
        # Step 3: Format the prompt
        prompt = obtain_relevant_files_prompt.format(problem_statement=problem_statement, structure=repo_structure)
        
        # Step 4: Get the response from the model
        messages = [Message(role="system", content=prompt)]
        response = self.model.generate(messages)
        
        # Step 5: Parse the LLM response to get the filenames
        generated_filenames = self._parse_llm_response(response.content)
        
        # Step 6: Map the generated filenames to actual filenames in the repository
        top_files = self._map_file_names(generated_filenames, all_files)
        
        return top_files