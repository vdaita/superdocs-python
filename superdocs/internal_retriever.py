from utils.codebase import list_non_ignored_files, CodebaseRetriever
import os
import chromadb
import time

class CodebaseRetriever():
    def __init__(self, directory, splitter=None):
        self.directory = directory
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="snippets")

        self.codebase_retriever = CodebaseRetriever(self.directory, verbose=True)
        
        if splitter:
            chunks = self.codebase_retriever.generate_documents(splitter=splitter)
        else:
            chunks = self.codebase_retriever.generate_documents()
        
        self.collection.add(
            documents=[chunk["content"] for chunk in chunks],
            metadata=[{"time": time.time(), "filename": chunk["filename"]} for chunk in chunks]
        )

    def reload_new_files(self):
        # TODO: Check which documents are outdated and then replace them:
        pass

    def retrieve_documents(self, base_n=100, reranked_n=10):
        pass