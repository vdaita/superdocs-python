from __future__ import annotations

import os
import subprocess
import chromadb
import time
import re
import uuid

from rank_bm25 import BM25Okapi
from gitignore_parser import parse_gitignore

# Chunker from tree-sitter
from dataclasses import dataclass, field
from tree_sitter import Tree, Node
from tree_sitter_languages import get_parser

from trafilatura import fetch_url, extract
from googlesearch import search

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

def non_whitespace_len(s: str) -> int:  # new len function
    return len(re.sub("\s", "", s))

def get_line_number(index: int, source_code: str) -> int:
    total_chars = 0
    for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
        total_chars += len(line)
        if total_chars > index:
            return line_number - 1
    return line_number

@dataclass
class Span:
    # Represents a slice of a string
    start: int = 0
    end: int = 0

    def __post_init__(self):
        # If end is None, set it to start
        if self.end is None:
            self.end = self.start

    def extract(self, s: str) -> str:
        # Grab the corresponding substring of string s by bytes
        return s[self.start : self.end]

    def extract_lines(self, s: str) -> str:
        # Grab the corresponding substring of string s by lines
        return "\n".join(s.splitlines()[self.start : self.end])

    def __add__(self, other: Span | int) -> Span:
        # e.g. Span(1, 2) + Span(2, 4) = Span(1, 4) (concatenation)
        # There are no safety checks: Span(a, b) + Span(c, d) = Span(a, d)
        # and there are no requirements for b = c.
        if isinstance(other, int):
            return Span(self.start + other, self.end + other)
        elif isinstance(other, Span):
            return Span(self.start, other.end)
        else:
            raise NotImplementedError()

    def __len__(self) -> int:
        # i.e. Span(a, b) = b - a
        return self.end - self.start
    
def chunker(
    tree: Tree,
    source_code: bytes,
    MAX_CHARS=512 * 3,
    coalesce=50,  # Any chunk less than 50 characters long gets coalesced with the next chunk
) -> list[Span]:

    # 1. Recursively form chunks based on the last post (https://docs.sweep.dev/blogs/chunking-2m-files)
    def chunk_node(node: Node) -> list[Span]:
        chunks: list[Span] = []
        current_chunk: Span = Span(node.start_byte, node.start_byte)
        node_children = node.children
        for child in node_children:
            if child.end_byte - child.start_byte > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Span(child.end_byte, child.end_byte)
                chunks.extend(chunk_node(child))
            elif child.end_byte - child.start_byte + len(current_chunk) > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Span(child.start_byte, child.end_byte)
            else:
                current_chunk += Span(child.start_byte, child.end_byte)
        chunks.append(current_chunk)
        return chunks

    chunks = chunk_node(tree.root_node)

    # 2. Filling in the gaps
    for prev, curr in zip(chunks[:-1], chunks[1:]):
        prev.end = curr.start
        curr.start = tree.root_node.end_byte

    # 3. Combining small chunks with bigger ones
    new_chunks = []
    current_chunk = Span(0, 0)
    for chunk in chunks:
        current_chunk += chunk
        if non_whitespace_len(
            current_chunk.extract(source_code)
        ) > coalesce and "\n" in current_chunk.extract(source_code):
            new_chunks.append(current_chunk)
            current_chunk = Span(chunk.end, chunk.end)
    if len(current_chunk) > 0:
        new_chunks.append(current_chunk)

    # 4. Changing line numbers
    line_chunks = [
        Span(
            get_line_number(chunk.start, source_code),
            get_line_number(chunk.end, source_code),
        )
        for chunk in new_chunks
    ]

    # 5. Eliminating empty chunks
    line_chunks = [chunk for chunk in line_chunks if len(chunk) > 0]

    return line_chunks

class CodebaseRetriever():
    def __init__(self, directory, splitter=None, use_vectorstore=True):
        self.directory = directory
        self.chroma_client = chromadb.Client()
        self.splitter = splitter
        self.use_vectorstore = use_vectorstore
        self.collection = None
        self.code_extensions = ["py", "js", "jsx", "tsx", "ts", "java", "go", "c", "cpp", "cc", "hpp", "rb"]

    def get_directory_files(self):
        """
        Lists all the files in a directory that are not gitignored, including untracked files.
        
        Returns:
        file_contents: the contenst of files to be considered
        """
        result = subprocess.run(f"cd {self.directory} && git ls-files --cached --others --exclude-standard", shell=True, check=True, text=True, capture_output=True)
        relevant_files = result.stdout.splitlines()
        print("Relevant files: ", relevant_files)
        
        # Read content of relevant files
        file_contents = {}
        for file_path in relevant_files:
            if file_path.split(".")[-1] in self.code_extensions:
                with open(os.path.join(self.directory, file_path), 'r') as file:
                    file_contents[file_path] = file.read()

        return file_contents
    
    def generate_chunks(self):
        contents = self.get_directory_files()
        final_chunks = []
        for filepath in contents:
            extension = filepath.split(".")[-1]
            parser = get_parser(language_map[extension])
            tree = parser.parse(contents[filepath].encode())
            chunks = chunker(tree, contents[filepath])    
            final_chunks.extend(
                [{
                    "content": f"Filepath: {filepath} \n" + chunk.extract_lines(contents[filepath]),
                    "filename": filepath
                } for chunk in chunks]
            )
        return final_chunks

    def load_all_documents(self):        
        chunks = self.generate_chunks()
        print("Chunks of length: ", len(chunks))

        if self.use_vectorstore:
            # self.chroma_client.delete_collection(name="snippets")
            self.collection = self.chroma_client.get_or_create_collection(name="snippets")
            self.collection.add(
                documents=[chunk["content"] for chunk in chunks],
                metadatas=[{"time": time.time(), "filename": chunk["filename"]} for chunk in chunks],
                ids=[str(uuid.uuid4()) for _ in chunks]
            )
        
        # BM25 Tokenizer
        tokenized_chunks = []
        for chunk in chunks:
            tokenized_chunks.append(chunk["content"])
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.bm25_corpus = tokenized_chunks

    def reload_new_files(self):
        # TODO: Check which documents are outdated and then replace them:
        pass

    def retrieve_documents(self, query, vectorstore_n=5, bm25_n=5):
        if not(self.collection):
            return []

        tokenized_query = re.findall(r'\b\w+\b|(?=[A-Z])|_', query)
        token_results = self.bm25.get_top_n(tokenized_query, self.bm25_corpus, n=bm25_n)

        if self.use_vectorstore:
            vectorstore_results = self.collection.query(query_texts=[query], n_results=vectorstore_n)
            print(len(vectorstore_results))
            print("Sample vectorstore result: ", vectorstore_results["documents"][0][0])
            print(len(token_results))
            print("Sample BM25 results: ", token_results[0])
            vectorstore_documents_list = [result for result in vectorstore_results["documents"][0]]
            combined = token_results + vectorstore_documents_list
            return combined
        
        return token_results

class SearchRetriever():
    def __init__(self, model):
        self.model = model
    
    def search(self, query):
        results = search(query, advanced=True)
        urls = [result.url for result in results[:3]]
        content = ""
        for url in urls:
            downloaded = fetch_url(url)
            extracted = extract(downloaded)
            content += extracted + "\n-----\n"
        
        # estimating 10000 tokens = ~40k characters
        content = content[:max(len(content), 40000)]
        return self.model("Summarize the given documentation to provide the most relevant information for a developer looking to add new features or debug their app.", [content], max_tokens=1000)
# TODO: write a test for the internal retriever