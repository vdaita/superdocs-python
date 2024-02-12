import os
import subprocess
from llama_index.node_parser import CodeSplitter
from llama_index.schema import Node

def llm_chunker(contents, model): # All of the AI aspects should be passed in uniform models instead of instantializing independently everywhere
    pass

def tree_sitter_chunker(contents):
    pass

language_map = {
    "py": "python",
    "js": "javascript",
    "java": "java",
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "javascript",
    "cc": "cpp",
    "hpp": "cpp",
    "cpp": "cpp",
    "rb": "ruby"
}

class CodebaseRetriever:
    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose
        pass

    def generate_documents(self, chunker=tree_sitter_chunker):
        result = subprocess.run("git ls-files --cached --others --exclude-standard", shell=True, check=True, text=True, capture_output=True)
        non_ignored_files = result.stdout.splitlines()

        if self.verbose:
            print("Found non ignored files: ", non_ignored_files)

        for rfilepath in non_ignored_files:
            extension = rfilepath.split(".")[-1]
            if not(extension in language_map.keys()):
                continue
            full_filepath = os.path.join(self.path, rfilepath)
            file = open(full_filepath, "r")
            contents = file.read()

            chunked = chunker(contents)
            return [f"Filename: {rfilepath} \n Content: {text}" for text in chunked]
