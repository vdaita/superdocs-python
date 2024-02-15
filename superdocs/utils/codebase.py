import os
import subprocess
from tree_sitter_languages import get_language, get_parser
from tree_sitter import Tree, Node
from dataclasses import dataclasses
from utils.tree_sitter_utils import chunker
import json
from superdocs.utils.gpt_output_utils import extract_json_code_data
from thefuzz import process, fuzz

CODE_SPLIT_PROMPT = """
"""

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

def generate_llm_chunker(client):
    def llm_chunker(contents, filename): # All of the AI aspects should be passed in uniform models instead of instantializing independently everywhere
        try:
            docs = []
            messages = [{
                "role": "system",
                "content": CODE_SPLIT_PROMPT
            }, {
                "role": "human",
                "content": contents
            }]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0116",
                messages=messages,
                max_tokens=512
            )
            response = response.choices[0].message.content
            lines = contents.split("\n")

            response = json.loads(extract_json_code_data(response)[0])
            docs = []
            for chunk in response:
                chunk['end'] = max(chunk['end'], len(lines) - 1) # about what the final index, inclusive should be
                snippet = '\n'.join(lines[chunk['start']:chunk['end'] + 1])
                docs.append(
                    Node(text=f"Filename: {rfilepath} \n Description: {chunk['description']} \n Code: {snippet}")
                )
            docs.extend(docs)
        except:
            return tree_sitter_chunker(contents, filename)
    return llm_chunker

def tree_sitter_chunker(contents, filename):
    extension = filename.split(".")[-1]
    parser = get_parser(language_map[extension])
    tree = parser.parse(contents.encode())
    chunks = chunker(tree, contents.encode())
    return [chunk.extract(contents) for chunk in chunks]

def list_non_ignored_files(directory):
    result = subprocess.run(f"cd {directory} && git ls-files --cached --others --exclude-standard", shell=True, check=True, text=True, capture_output=True)
    non_ignored_files = result.stdout.splitlines()
    return non_ignored_files


def generate_documents(self, path, verbose=False, chunker=tree_sitter_chunker):
    non_ignored_files = list_non_ignored_files(path)

    if verbose:
        print("Found non ignored files: ", non_ignored_files)

    all_chunks = []

    for rfilepath in non_ignored_files:
        extension = rfilepath.split(".")[-1]
        if not(extension in language_map.keys()):
            continue
        full_filepath = os.path.join(path, rfilepath)
        file = open(full_filepath, "r")
        contents = file.read()

        chunked = chunker(contents)
        all_chunks.extend([
            {
                "content": f"Filename: {rfilepath} \n Snippet: ```{text}```",
                "filename": rfilepath
            }
            for text in chunked
        ])

    return all_chunks

def find_closest_file(directory, filepath, threshold=95):
    files = list_non_ignored_files(directory)
    closest_match = process.extractOne(filepath, files, scorer=fuzz.token_sort_ratio)
    print("find_closest_file: closest_match: ", closest_match)
    if closest_match[1] < threshold:
        return filepath
    else:
        print("Found closest file in find_closest_file: ", directory, filepath, closest_match[0])
        return closest_match[0]