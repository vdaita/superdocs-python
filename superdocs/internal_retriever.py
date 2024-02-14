from utils.codebase import list_non_ignored_files, CodebaseRetriever
import os
import chromadb
import time
import re

from rank_bm25 import BM25Okapi


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

        vectorstore_results = self.collection.query(query_texts=[query], n_results=vectorstore_n)
        vectorstore_documents_list = [result["document"] for result in vectorstore_results]
        
        combined = token_results + vectorstore_documents_list
        return reranker.rerank(combined, query)
