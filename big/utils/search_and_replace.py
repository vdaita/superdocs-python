# From sweep.ai

import re
from dataclasses import dataclass

from rapidfuzz import fuzz
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Match:
    block: str
    score: float

def find_best_match(query_code: str, original_code: str):
    original_lines = original_code.splitlines()
    query_lines = query_code.splitlines()

    best_match = Match("", -1)

    for start_line in range(len(original_lines)):
        min_end = min(len(original_lines), max(start_line, start_line + len(query_lines) - 5)) # +- 5 lines for tolerance
        max_end = min(len(original_lines), start_line + len(query_lines) + 5)
        for end_line in range(min_end, max_end):
            snippet_from_original = "\n".join(original_lines[start_line:end_line + 1]) # the loop already doesn't include max_end
            score = fuzz.ratio(snippet_from_original, query_code)
            
            if score > best_match.score:
                best_match = Match(snippet_from_original, score)
    
    return best_match