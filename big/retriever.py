import os
import chromadb
import time
import re
import uuid

from rank_bm25 import BM25Okapi
from gitignore import parse_gitignore

# Chunker from tree-sitter
from __future__ import annotations
from dataclasses import dataclass, field
from tree_sitter import Tree, Node
from tree_sitter_languages import get_languages, get_parser

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

    def get_directory_files(self):
        # Initialize an empty list to store relevant files
        relevant_files = []

        # Traverse the directory tree recursively
        for root, dirs, files in os.walk(self.directory):
            # Load the .gitignore file if it exists in the current directory
            gitignore_path = os.path.join(root, '.gitignore')
            if os.path.exists(gitignore_path):
                # Parse the .gitignore file
                gitignore_rules = parse_gitignore(gitignore_path)

                # Filter out files that match the .gitignore rules
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if not gitignore_rules(file_path):
                        relevant_files.append(file_path)
            else:
                # If .gitignore doesn't exist, consider all files as relevant
                relevant_files.extend([os.path.join(root, file_name) for file_name in files])

        # Read content of relevant files
        file_contents = {}
        for file_path in relevant_files:
            with open(file_path, 'r') as file:
                file_contents[file_path] = file.read()

        return file_contents
    
    def generate_chunks(self):
        contents = self.get_directory_files()
        final_chunks = []
        for filepath in contents:
            extension = filepath.split(".")[-1]
            parser = get_parser(language_map[extension])
            tree = parser.parse(contents["filepath"].encode())
            chunks = chunker(tree, contents["filepath"])    
            final_chunks.extend(
                [f"Filepath: {filepath} \n" + chunk.extract(contents) for chunk in chunks]
            )
        return final_chunks

    def load_all_documents(self):        
        chunks = self.generate_chunks()

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
            tokenized_chunks.append(re.findall(r'\b\w+\b|(?=[A-Z])|_', chunk["content"]))
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.bm25_corpus = tokenized_chunks

    def reload_new_files(self):
        # TODO: Check which documents are outdated and then replace them:
        pass

    def retrieve_documents(self, query, reranker, vectorstore_n=25, bm25_n=25, reranked_n=10):
        tokenized_query = re.findall(r'\b\w+\b|(?=[A-Z])|_', query)
        token_results = self.bm25.get_top_n(tokenized_query, self.bm25_corpus, n=bm25_n)

        if self.use_vectorstore:
            vectorstore_results = self.collection.query(query_texts=[query], n_results=vectorstore_n)
            vectorstore_documents_list = [result for result in vectorstore_results["documents"][0]]
            combined = token_results + vectorstore_documents_list
            return reranker.rerank(combined, query)
        
        return reranker.rerank(token_results, query)

# TODO: write a test for the internal retriever