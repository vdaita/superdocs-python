from thefuzz import fuzz
import re
from pathlib import Path
from dataclasses import dataclass
from .utils.prompts import EXECUTE_PROMPT, EVALUATION_PROMPT
from .utils.model import extract_code_block_data, extract_xml_tags
from .utils.mcts import ExecutionNode, MCTS
from multiprocess import Pool

import logging

logger = logging.getLogger(__name__)

@dataclass
class Match:
    block: str
    score: float

@dataclass
class Hunk:
    filepath: str
    text: str

@dataclass
class SearchReplaceChange:
    filepath: str
    search_block: str
    replace_block: str


def line_relevant(line):
    return not(len(line.strip()) == 0 or line.startswith("#") or line.startswith("//"))

def find_best_match(query_code: str, original_code: str):
    query_code = query_code.strip()

    original_lines = original_code.splitlines()
    query_lines = query_code.splitlines()

    if len(query_lines) == 0:
        return Match("SUPERDOCSTHISSTRINGWILLNEVEREVERBEFOUND", 100)

    best_match = Match("", -1)

    for start_line in range(len(original_lines)):
        min_end = min(len(original_lines), max(start_line, start_line + len(query_lines) - 5)) # +/- 5 lines for tolerance
        max_end = min(len(original_lines), start_line + len(query_lines) + 5)
        for end_line in range(min_end, max_end):
            full_original_snippet = "\n".join(original_lines[start_line:end_line + 1])

            snippet_from_original = "\n".join([line for line in original_lines[start_line:end_line + 1] if line_relevant(line)]) # the loop already doesn't include max_end
            snippet_from_query = "\n".join([line for line in query_lines if line_relevant(line)])

            stripped_original = " ".join([line.strip() for line in snippet_from_original.splitlines()])
            stripped_query =  " " .join([line.strip() for line in snippet_from_query.splitlines()])

            score = fuzz.ratio(stripped_original, stripped_query)

            # Weighting the first and last lines by 3x
            score += 3*fuzz.ratio(original_lines[start_line], query_lines[0])
            score += 3*fuzz.ratio(original_lines[end_line], query_lines[-1])
        
            if score > best_match.score:
                best_match = Match(full_original_snippet, score)
    return best_match

def find_hunks(diff_string):
    hunks = []
    current_filename = ""
    current_lines = ""
    for line in diff_string.splitlines():
        if line.startswith("---"):
            continue
        elif line.lstrip().startswith("+++"):
            if len(current_filename) > 0:
                hunks.append(Hunk(current_filename, current_lines))
            current_filename = line[3:]
            current_lines = ""
        elif line.lstrip().startswith("@@"):
            if len(current_filename) > 0:
                hunks.append(Hunk(current_filename, current_lines))
            current_lines = ""
        else:
            current_lines += line
            current_lines += "\n"
    hunks.append(Hunk(current_filename, current_lines))
    return hunks

def find_closest_file(filepath, all_filepaths):
    best_match = Match("", -1)
    for fp in all_filepaths:
        score = fuzz.ratio(filepath, fp)
        if score > best_match.score:
            best_match = Match(fp, score)
    
    return best_match.block if best_match.score > 0.7 else filepath        

def parse_diff(diff_string):
    hunks = find_hunks(diff_string)
    search_replace_blocks = []

    for hunk in hunks:
        filepath = hunk.filepath
        text = hunk.text

        search_block = ""
        replace_block = ""

        for line in text.splitlines():
            if line.startswith("-"):
                search_block += " " + line[1:] + "\n"
            elif line.startswith("+"):
                replace_block += " " + line[1:] + "\n"
            else:
                search_block += line + "\n"
                replace_block += line + "\n"
        
        search_replace_blocks.append(
            SearchReplaceChange(filepath, search_block, replace_block)
        )
    
    search_replace_blocks.append(
        SearchReplaceChange(filepath, search_block, replace_block)
    )

    return search_replace_blocks

def stringify_files(file_dictionary):
        file_string = ""
        for file in file_dictionary:
            file_string += "Filepath: " + file + "\n ----- \n"
            file_string += file_dictionary[file]
        return file_string

def outer_evaluate_generation():
    pass

def outer_execute():
    pass

class Executor: # Uses an objective, general context for information, and a bunch of current files as context
    def __init__(self, goal, files, context, model, verbose=False):
        self.goal = goal
        self.context = context
        self.model = model
        self.verbose = verbose
        self.files = files
        self.notes = "## Notes from previous executions: \n"

    def evaluate_generation(self, goal, context, annotated_modified_files):
        output = self.model(EVALUATION_PROMPT, ["Goal: " + goal, "Context: " + context, stringify_files(annotated_modified_files)])
        simplicity = extract_xml_tags(output, "simplicity")
        functionality = extract_xml_tags(output, "functionality")
        integration = extract_xml_tags(output, "integration")
        feedback = extract_xml_tags(output, "feedback")
        # notes = extract_xml_tags(output, "notes")
        
        if len(simplicity) == 0 or len(functionality) == 0 or len(integration) == 0:
            return -1

        simplicity, functionality, integration = int(simplicity[0]), int(functionality[0]), int(integration[0])

        if len(feedback) == 0:
            feedback = ""
        else:
            feedback = feedback[0]

        # if len(notes) == 0:
        #     notes = ""
        # else:
        #     notes = notes[0]

        score = 10 * (((simplicity * 0.25 + functionality) * integration)/(10 * 0.25 + 10))
        logger.info("Scoring determination made: " + f" Simplicity {simplicity} " + f" Functionality: {functionality} " + f" Integration: {integration}" + f" Overall: {score} " + f"Feedback: {feedback}")
        return score, feedback

    def chain_execute(self):
        initial_modifications = [self.execute() for _ in range(3)]
        best_modifications = self.files

        best_modifications_score = 0
        best_modifications_feedback = ""

        for modification in initial_modifications:
            modified_files, annotated_modified_files = modification["unannotated"], modification["annotated"]
            logger.info("Received modification: ")
            logger.info(stringify_files(modified_files))
            logger.info("=============")
            logger.info(stringify_files(annotated_modified_files))
            score, feedback = self.evaluate_generation(self.goal, self.context, annotated_modified_files)
            if score > best_modifications_score:
                best_modifications_score = score
                best_modifications_feedback = feedback
                best_modifications = modified_files

        # Second step of refining the answer
        self.files = best_modifications
        self.goal += f"Make sure to fix think through and fix all possible bugs and use the following feedback to improve your response. In your plan, ensure that every step is clearly specified. Change only what's necessary. Here is some additional feedback to help you: {best_modifications_feedback}"
        
        second_modifications_round = [self.execute() for _ in range(1)]
        for modification in second_modifications_round:
            modified_files, annotated_modified_files = modification["unannotated"], modification["annotated"]
            logger.info("Received modification: ")
            logger.info(stringify_files(modified_files))
            logger.info("=============")
            logger.info(stringify_files(annotated_modified_files))
            score, feedback = self.evaluate_generation(self.goal, self.context, annotated_modified_files)
            if score > best_modifications_score:
                best_modifications_score = score
                best_modifications_feedback = feedback
                best_modifications = modified_files

        return best_modifications

    def execute(self, alternative_files=[], additional_context=""):
        if len(alternative_files) == 0:
            alternative_files = self.files
        output = self.model(EXECUTE_PROMPT, ["Objective: " + self.goal, "Context: " + self.context + ("" if len(additional_context) == 0 else additional_context), "Previous execution notes: " + self.notes, "Files: " + stringify_files(alternative_files)])
        logger.debug("==== RECEIVED EXECUTE RESPONSE ====")
        logger.debug(output)

        diff_blocks = extract_code_block_data(output, "diff")

        sr_blocks = []
        for block in diff_blocks:
            sr_blocks += parse_diff(block)
        
        original_files = self.files.copy()
        modified_files = self.files.copy()
        annotated_modified_files = self.files.copy() # Not going to be returned directly to the users, but is going to show the LLM which lines were modified. 

        for block in sr_blocks:
            if len(block.search_block.strip()) == 0 and len(block.replace_block.strip()) == 0:
                continue

            match_filepath = find_closest_file(block.filepath, list(self.files.keys()))
            
            if len(match_filepath) == 0:
                continue
    
            if len(block.search_block.strip()) == 0:
                if self.verbose:
                    print("Replacing file contents:")
                modified_files[match_filepath] = block.replace_block
                annotated_write_block = "\n".join(f"+ {line}" for line in block.replace_block.splitlines())
                annotated_modified_files[match_filepath] = annotated_write_block
            else:
                logger.debug("Trying to match file that's in: " + match_filepath)
                best_match = find_best_match(block.search_block, original_files[match_filepath])
                if best_match.score > 500:
                    modified_files[match_filepath] = modified_files[match_filepath].replace(best_match.block, block.replace_block)

                    annotated_search_block = "\n".join(f"- {line}" for line in best_match.block.splitlines())
                    annotated_replace_block = "\n".join(f"+ {line}" for line in block.replace_block.splitlines())
                    annotated_modified_files[match_filepath] = modified_files[match_filepath].replace(best_match.block, annotated_search_block + "\n" + annotated_replace_block)

                    
                    logger.debug("Making replacement: ")
                    logger.debug("=====SEARCH=====")
                    logger.debug(block.search_block)
                    logger.debug(f"=====MATCH with closeness {best_match.score}======")
                    logger.debug(best_match.block)
                    logger.debug("=====REPLACE=====")
                    logger.debug(block.replace_block)
                else:
                    logger.debug("Failed to match")
                    logger.debug(block.search_block)
        
        return {"unannotated": modified_files, "annotated": annotated_modified_files} # [0] are the actual modified files and [1] are the annotated_modified_files