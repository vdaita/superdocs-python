from abc import abstractmethod, ABC
from typing import List, Optional
from utils.types import File
from difflib import unified_diff
from rapidfuzz.distance.Levenshtein import similarity

class Reranker(ABC):
    def rerank(self, changes: List[List[File]]) -> List[File]:
        pass

class SimpleReranker(Reranker): # Pick the generated option that is most similar to the other options in terms of changes
    def __init__(self):
        pass

    def rerank(self, changes: List[List[File]]) -> List[File]:
        best_similarity = 0
        best_patch = changes[0]

        for i in range(len(changes)):
            total_similarity_score = 0
            for j in range(len(changes)):
                cross_similarity_score = 0
                if i == j:
                    continue
                    
                for file in changes[i]:
                    for other_file in changes[j]:
                        if not(file.filepath == other_file.filepath):
                            continue
                        
                        i_content = file.modified_content if file.modified_content else file.original_content
                        j_content = other_file.modified_content if other_file.modified_content else other_file.original_content
                        file_similarity_score = similarity(i_content, j_content)
                        cross_similarity_score += file_similarity_score                    
                total_similarity_score += cross_similarity_score
            if total_similarity_score > best_similarity:
                best_similarity = total_similarity_score
                best_patch = changes[i]
            
        return best_patch
    
class LLMReranker(Reranker): # TODO: Implement a reranker that uses a language model to rerank the patches
    pass