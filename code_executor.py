from thefuzz import fuzz
import re
# from .repo import find_closest_file
from unidiff import PatchSet
import json
import os
# from .sweep_search_and_replace import find_best_match
from utils.codebase import find_closest_file
from utils.search_and_replace import find_best_match
from utils.gpt_output_utils import extract_code_block_data, extract_xml_tags
from utils.prompts import SEARCH_REPLACE_PROMPT, PLAN_WRITING_PROMPT, DIFF_PROMPT
from utils.diff_parser import parse_diff

from diff_match_patch import diff_match_patch
from dataclasses import dataclass
from io import StringIO

from pathlib import Path

def process_with_search_replace_blocks(model, directory, input_text):
    response = model(SEARCH_REPLACE_PROMPT, [input_text])
    blocks = extract_xml_tags(response, "block")

    changes = []

    for block in blocks: # do only one search and replace portion at a given time
        filename = extract_xml_tags(block, "filename")
        search_portion = extract_xml_tags(block, "search")
        replace_portion = extract_xml_tags(block, "replace")

        if len(filename) == 0 or len(search_portion) == 0 or len(replace_portion) == 0:
            print("A subblock was missed in search-replace.")
            continue
            
        filename, search_portion, replace_portion = filename[0], search_portion[0], replace_portion[0]

        filename = find_closest_file(directory, filename)
        full_filepath = os.path.join(directory, filename)
        lines = open(full_filepath, "r").read().splitlines()


        search_best_match = find_best_match(search_portion, full_filepath)
        search_portion = "\n".join(lines[search_best_match.start:search_best_match.end])
        changes.append({
            "filename": filename,
            "original": search_portion,
            "new": replace_portion
        })

    return changes


def process_with_diffs(model, directory, input_text, verbose=True):
    response = model(DIFF_PROMPT, [input_text])
    diffs = extract_code_block_data(response, "diff")
    changes = {}
    for diff in diffs:
        if verbose:
            print(diff)
        parsed_diff = parse_diff(diff)
        for hunk in parsed_diff:
            print("Filepath: ", hunk.filepath)
            print("SEARCH:")
            print(hunk.search_block)
            print("REPLACE:")
            print(hunk.replace_block)

        changes.update(fuzzy_process_diff(directory, parsed_diff))
    return changes

def lines_to_chars(lines, mapping):
    new_text = []
    for char in lines:
        new_text.append(mapping[ord(char)])

    new_text = "".join(new_text)
    return new_text

def fuzzy_process_diff(directory, hunks):
    cached_files = {}
    for hunk in hunks:
        filepath = find_closest_file(directory, hunk.filepath)
        contents = cached_files[filepath] if filepath in cached_files else open(os.path.join(directory, filepath)).read() 
        file_lines = contents.splitlines()
        best_match = find_best_match(hunk.search_block, contents)
        matched_search_block = best_match.block
        print("=====SEARCH BLOCK======")
        print(hunk.search_block)
        print("======MATCHED BLOCK======")
        print(matched_search_block)
        print("======REPLACE BLOCK======")
        print(hunk.replace_block)

        contents = contents.replace(matched_search_block, hunk.replace_block)
        cached_files[filepath] = contents
    return cached_files

if __name__ == "__main__":
    directory = "/Users/vijaydaita/Files/uiuc/rxassist/rxmind-nextjs-main"
    diff = open("test_diff.txt", "r").read()
    changes = fuzzy_process_diff(directory, parse_diff(diff))
    for filename in changes:
        print("FILENAME: ", filename)
        print(changes[filename])