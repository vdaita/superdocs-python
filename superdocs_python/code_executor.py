from thefuzz import fuzz
import re
from pathlib import Path
from dataclasses import dataclass
from .utils.model import extract_code_block_data, extract_xml_tags
from multiprocess import Pool
from .utils.node import Reflection, Node
from .utils import diff_utils
import diff_match_patch as dmp_module
import logging
import time
from .utils import prompts
import json
import sys
from pathvalidate import ValidationError, validate_filename

from pydantic import BaseModel, Field
from typing import Literal, List

logger = logging.getLogger(__name__)

tools = [
            {
                "type": "function",
                "function": {
                    "name": "insert_code_before",
                    "description": "Insert new lines of code before original lines",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "The file in which you want to make this edit"
                            },
                            "lines_inserted_before": {
                                "type": "string",
                                "description": "The new lines of code that you want to insert before original lines in the file"
                            },
                            "original_lines": {
                                "type": "string",
                                "description": ""
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "insert_code_after",
                    "description": "Insert new lines of code after original lines",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "The file in which you want to make this edit"
                            },
                            "original_lines": {
                                "type": "string",
                                "description": "A unique stretch of lines from the original file including all whitespace, without skipping any lines"
                            },
                            "lines_inserted_after": {
                                "type": "string",
                                "description": "The new lines of code that you want to insert after original lines in the file"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_and_replace_code",
                    "description": "Replace old lines of code with new lines of code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "The file in which you want to make this edit"
                            },
                            "original_lines": {
                                "type": "string",
                                "description": "A unique stretch of lines from the original file including all whitespace, without skipping any lines"
                            },
                            "new_replacement_lines": {
                                "type": "string",
                                "description": "The new lines of code that you want to place the original lines with"
                            }
                        }
                    }
                }
            }
]


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

class FileModificationOperation(BaseModel):
    reasoning: str = Field(description="What step are you implementing? Think step-by-step about how that step should be implemented.")
    filepath: str = Field(description="The filepath where the change is being made.")
    modification_type: Literal['insert_code_before', 'insert_code_after', 'replace']
    location: str = Field(description="Describes where in the code this change should be made using natural language, relative to other definitions/variables.")
    location_lines: str = Field(description="A unique stretch of lines from the original file including all whitespace, without skipping any lines, where you want this edit to be implemented")
    insertion_lines: str = Field(description="A new stretch of lines including all whitespace that that you are inserting before, after, or using for replacement")

class FileModificationOperationList(BaseModel):
    file_modifications: List[FileModificationOperation]

class FunctionType(BaseModel):
    reasoning: str = Field(description="Think about what the instruction refers in relation to the code itself.")
    request_type: Literal['insert_code_before', 'insert_code_after', 'search_and_replace_code']

class InsertCodeBeforeType(BaseModel):
    reasoning: str = Field(description="Think about how this instruction should be implemented to the best of your ability.")
    filepath: str = Field(description="The filepath corresponding to this step")
    original_lines: str = Field(description="A unique stretch of lines from the original file including all whitespace, without skipping any lines")
    lines_inserted_before: str = Field(description="The new lines of code that you want to insert before the original lines in the file")

class InsertCodeAfterType(BaseModel):
    reasoning: str = Field(description="Think about how this instruction should be implemented to the best of your ability.")
    filepath: str = Field(description="The filepath corresponding to this step")
    original_lines: str = Field(description="A unique stretch of lines from the original file including all whitespace, without skipping any lines")
    lines_inserted_after: str = Field(description="The new lines of code that you want to insert after original lines in the file")

class SearchAndReplaceCodeType(BaseModel):
    reasoning: str = Field(description="Think about how this instruction should be implemented to the best of your ability.")
    filepath: str = Field(description="The filepath corresponding to this step")
    original_lines: str = Field(description="A unique stretch of lines from the original file including all whitespace, without skipping any lines")
    new_replacement_lines: str = Field(description="The new lines of code that you want to insert after original lines in the file")

class InvalidRewriteTypeException(Exception):
    pass

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
            stripped_query =  " ".join([line.strip() for line in snippet_from_query.splitlines()])

            score = fuzz.ratio(stripped_original, stripped_query)

            # Weighting the first and last lines by 3x
            score += 3*fuzz.ratio(original_lines[start_line], query_lines[0])
            score += 3*fuzz.ratio(original_lines[end_line], query_lines[-1])

            if score > best_match.score:
                best_match = Match(full_original_snippet, score)
    return best_match

def find_closest_file(filepath, all_filepaths):
    best_match = Match("", -1)
    for fp in all_filepaths:
        score = fuzz.ratio(filepath, fp)
        if score > best_match.score:
            best_match = Match(fp, score)

    return best_match.block if best_match.score > 0.85 else filepath

def stringify_files(file_dictionary):
        file_string = ""
        for file in file_dictionary:
            file_string += f"{file}\n"
            file_string += f"```\n{file_dictionary[file]}\n```\n"
            file_string += "-----\n"
        return file_string

class Executor: # Uses an objective, general context for information, and a bunch of current files as context
    def __init__(self, goal, files, context, model, aux_model=None, verbose=False):
        self.goal = goal
        self.context = context
        self.model = model
        if not(aux_model):
            self.aux_model = model
        else:
            self.aux_model = aux_model
        self.verbose = verbose
        self.files = files

    def chain_plan_and_execute_lats(self, generation_per_level=3, max_height=3, execution_prompt_type="rewrite"):
        root = Node(self.files, Reflection(feedback="The goal has not yet been implemented.", score=0, found_solution=False))
        if execution_prompt_type == "rewrite":
            system_prompt = prompts.AIDER_REWRITE_PLAN_AND_EXECUTE_PROMPT
        elif execution_prompt_type == "udiff":
            system_prompt = prompts.AIDER_UDIFF_PLAN_AND_EXECUTE_PROMPT
        else:
            raise InvalidRewriteTypeException()

        while root.height < max_height and not(root.is_solved):
            best_child = root.best_child
            if not(best_child):
                best_child = root
            rewrite_user_prompt = [[("user", f"# Original Files:\n{stringify_files(self.files)} \n # Goal to implement: {self.goal}"),
                                    ("assistant", f"# Most Recent Revision:\n{stringify_files(best_child.content)}")
                                    ] for _ in range(generation_per_level)]
            if best_child.content == self.files:
                rewrite_user_prompt = [[("user", f"# Files:\n{stringify_files(self.files)} \n # Goal to implement: {self.goal}")] for _ in range(generation_per_level)]

            rewritten_files = self.model([system_prompt]*generation_per_level, rewrite_user_prompt)

            parse_files = []
            if execution_prompt_type == "rewrite":
                parsed_files = [self.process_rewrite(rewrite) for rewrite in rewritten_files]
            elif execution_prompt_type == "udiff":
                parsed_files = [self.process_unified_diff_output(udiff) for udiff in rewritten_files]
            else:
                raise InvalidRewriteTypeException()

            logger.info(f"Generated parsed files: \n {[stringify_files(parse_file) for parse_file in parsed_files]}")
            reflections = [self.score_code_output(parse_file) for parse_file in parsed_files]
            logger.info(f"Generated reflections: {reflections}")

            new_nodes = [Node(
                content=pf,
                reflection=reflection,
                parent=best_child
            ) for (pf, reflection) in zip(parsed_files, reflections)]
            best_child.children.extend(new_nodes)

        logger.info(f"Generated best_child content: {stringify_files(root.best_child.content)}")
        return root.best_child.content

    def process_unified_diff_output(self, output):
        diff_blocks = extract_code_block_data(output, "diff")

        sr_blocks = []
        for block in diff_blocks:
            sr_blocks += diff_utils.parse_diff(block)

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
                    logger.info("Replacing file contents:")
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

        return modified_files

    def get_single_file(self):
        return self.files[list(self.files.keys())[0]]

    def score_plan_output(self, generated_plan): # Switch to using JSON
        system_prompt = "Score the plan and provide feedback."\
            "Evaluate the plan's simplicity (does the plan use a minimal number of new variables, libraries, and imports), effectiveness (if implemented, how well will it achieve the goal), and detail (for each change, does it describe exactly where/how to make the change within the code)."\
            "Output your response in the following manner, using XML tags: \n <simplicity>Number from 0-10</simplicity> \n <effectiveness>Number from 0-10</effectiveness> \n <detail>Number from 0-10</detail> \n <feedback>Bullet points</feedback>\n"
        user_prompt = [f"# Generated plan: \n {generated_plan} \n # Goal: {self.goal} \n # Files: {stringify_files(self.files)}"]

        comparison = self.model(system_prompt, user_prompt)
        numerical_variables = ["simplicity", "effectiveness", "detail"]
        variable_values = {}

        for variable in numerical_variables:
            extracted = extract_xml_tags(comparison, "simplicity")
            if len(extracted) == 0:
                extracted = 0
            else:
                extracted = re.findall(r'\d+', extracted[0])
                if len(extracted) == 0:
                    extracted = 0
                else:
                    extracted = int(extracted[0])
            variable_values[variable] = min(extracted, 10)

        feedback = extract_xml_tags(comparison, "feedback")
        if len(feedback) == 0:
            feedback = "Iterate on the previous output."
        else:
            feedback = feedback[0]
            if len(feedback) == 0:
                feedback = "Iterate on the previous output."

        score = (variable_values["simplicity"] + variable_values["effectiveness"] + variable_values["detail"])/30.0

        reflection = Reflection(feedback=feedback, score=score, found_solution=(score > 0.9))

        logger.info(f"# OUTPUT EVALUATED: \n {generated_plan} \n # SCORE GIVEN {reflection} \n Simplicity: {variable_values['simplicity']}, Effectiveness: {variable_values['effectiveness']}, Detail: {variable_values['detail']}")

        return reflection

    def score_code_output(self, generated_files):
        system_prompts = "Compare the generated files to the original file. First, state the differences between the original file and the generated files. Then, reflect on how the generated code completed the goal, given the provided context. Write a score from 0-100 and enclose that score within <score></score> XML tags."\
              "Before outputting the score, think step by step. Write feedback, describe the next step of changes that need to be made and enclose it within the <feedback></feedback> XML tags. If the goal is fully solved, while maintaining consistency and style, output <problem-solved/>. Otherwise, output <problem-not-solved/>"

        user_prompts = [
            f"# Original files: \n {stringify_files(self.files)} \n # Goal: \n {self.goal} \n # Generated files: \n {stringify_files(generated_files)}"
        ]

        comparison = self.model(system_prompts, user_prompts)

        score = extract_xml_tags(comparison, "score")
        if len(score) == 0:
            score = 0
        else:
            score = re.findall(r'\d+', score[0])
            if len(score) == 0:
                score = 0
            else:
                score = int(score[0])

        feedback = extract_xml_tags(comparison, "feedback")
        if len(feedback) == 0:
            feedback = "There is no feedback."
        else:
            feedback = feedback[0]
            if len(feedback) == 0:
                feedback = "There is no feedback."

        reflection = Reflection(feedback=feedback, score=score, found_solution=("problem-solved" in comparison))

        logger.info(f"# OUTPUT EVALUATED: \n {stringify_files(generated_files)} \n # SCORE GIVEN {reflection}")

        return reflection

    def score_code_output_two_step(self, generated_files):
        system_prompts = [
        "State whether or not the generated files meet the intended features and is syntactically and logically correct. First, think step by step and then <YES> if the generated code meets the requirements or <NO> otherwise.",
        "Compare the file to the original file. First, state the differences between the original file and the generated files. Then, reflect on how the generated code completed the goal. Write a score from 0-100 and enclose that score within <score></score> XML tags."
         + "Before outputting the score, think step by step. Write feedback and enclose it within the <feedback></feedback> XML tags."
        ]

        user_prompts = [
            [f"# Generated files: \n {stringify_files(generated_files)} \n # Goal: \n {self.goal}"],
            [f"# Original files: \n {stringify_files(self.files)} \n # Goal: \n {self.goal} \n # Generated files: \n {stringify_files(generated_files)}"]
        ]

        meets_description, comparison = self.model(system_prompts, user_prompts)
        meets_description = "<YES>" in meets_description

        score = extract_xml_tags(comparison, "score")
        if len(score) == 0:
            score = 0
        else:
            score = re.findall(r'\d+', score[0])
            if len(score) == 0:
                score = 0
            else:
                score = int(score[0])

        feedback = extract_xml_tags(comparison, "feedback")
        if len(feedback) == 0:
            feedback = "There is no feedback."
        else:
            feedback = feedback[0]
            if len(feedback) == 0:
                feedback = "There is no feedback."

        reflection = Reflection(feedback=feedback, score=score, found_solution=meets_description)

        logger.info(f"# OUTPUT EVALUATED: \n {stringify_files(generated_files)} \n # SCORE GIVEN {reflection}")

        return reflection

    def chain_execute_rewrite(self):
        plan = self.refine_plan()
        logger.info("PLAN")
        logger.info(plan)
        edited_files = self.implement_plan_rewrite(plan)
        return edited_files

    def chain_execute_block_edits(self):
        plan = self.refine_plan()
        logger.info("PLAN")
        logger.info(plan)
        edited_file = self.implement_plan_blocks(plan)
        return edited_file

    def refine_plan(self): # TODO: start with multiple candidate outputs
        plan_system_prompt = f"Write a plan to achieve the given goal. Describe file and code changes in detail. "\
            "Explicitly describe where in the file code changes should be located and provide code snippets." \
            "Respect and use existing conventions, libraries, etc that are already present in the code base. Number each step of the plan 1., 2., 3., etc."\
            f"More specifically, the description for each change should specify the chunk of the original file that the new changes are replacing, " \
                "or the chunk of the original before/after which the new changes go, or if the change goes at the very top or bottom of the file. \n" \
            f"\n Goal: {self.goal}"

        root = Node(f"Implement {self.goal}", Reflection(feedback="Write out a full plan.", score=0.0, found_solution=False))

        while root.height < 3 and not(root.is_solved):
            best_child = root.best_child
            if not(best_child):
                best_child = root

            plan_user_prompt = [("user", f"# Original Files: \n {stringify_files(self.files)} \n # Goal: {self.goal}"),
                                    ("assistant", f"# Most Recent Revision \n {best_child.content}"),
                                    ("user", f"# Feedback \n {best_child.reflection.as_message} \n Please fully rewrite the plan.")]

            if best_child.reflection.score == 0.0:
                plan_user_prompt = [("user", f"# Original Files: \n {stringify_files(self.files)} \n # Goal: {self.goal}")]

            candidate_plans = self.model([plan_system_prompt]*3, [plan_user_prompt]*3)

            # TODO: Evaluate the newly generated parsed
            reflections = [self.score_plan_output(candidate_plan) for candidate_plan in candidate_plans]

            # TODO: Add the new files with evaluations and everything into the solution
            new_nodes = [Node(
                content=candidate_plan,
                reflection=reflection,
                parent=best_child
            ) for (candidate_plan, reflection) in zip(candidate_plans, reflections)]
            best_child.children.extend(new_nodes)

        best_child = root.best_child
        if not(best_child):
            best_child = root

        return best_child.content

    def implement_plan_rewrite(self, plan):
        rewrite_system_prompt = "Now, follow the given plans and rewrite each of the files. Make sure you rewrite each file fully from start to finish - you are a DILIGENT programmer! Format each file rewrite like so: [filepath]\n```\nyour code\n```"
        rewrite_user_prompt = [f"# Original Files:\n{stringify_files(self.files)}\n# Plan:\n{plan}"]
        rewrite = self.aux_model(rewrite_system_prompt, rewrite_user_prompt)
        # additional_rewrite_user_prompts = rewrite_user_prompt + [("assistant", rewrite), ("user", "Continue your rewrite from where you left off if neccessary.")]
        # additional_rewrite = self.model(rewrite_system_prompt, additional_rewrite_user_prompts)
        # rewrite += "\n" + additional_rewrite
        logger.info(f"Generated rewrite text: \n {rewrite}")
        processed_rewrite = self.process_rewrite(rewrite)
        return processed_rewrite

    def implement_plan_blocks_multifunctions(self, plan):
        pattern = r'^\d+\.\s+'
        steps = re.split(pattern, plan, flags=re.MULTILINE)
        new_files = self.files.copy()

        for step in steps:
            logger.info(f"Processing step {step}")
            function_classification = self.aux_model.call_with_json( # TODO: replace with semantic router
                "Classify whether the change requested in the file-editing step below is a search-and-replace request or it can be better classified as an insertion request (either before and after).",
                [f"# Original files: \n {stringify_files(self.files)}\n # Change requested \n{step}"],
                FunctionType
            )
            logger.info(f"Generated function classification: {json.dumps(function_classification, indent=4)}")
            function_classification = function_classification["request_type"]
            logger.info(f"Extracted function classification type {function_classification}")

            function_classification_type_map = {
                "insert_code_before": InsertCodeBeforeType,
                "insert_code_after": InsertCodeAfterType,
                "search_and_replace_code": SearchAndReplaceCodeType
            }

            function_classification_prompt_map = {
                "insert_code_before": "Write an implementation of the change requested by inserting new code before original code.",
                "insert_code_after": "Write an implementation of the change requested by inserting new code after original code.",
                "search_and_replace_code": "Write an implementation of the change requested by replacing original code with new code."
            }

            implemented_change = self.aux_model.call_with_json(
                function_classification_prompt_map[function_classification],
                [f"# Original files:\n{stringify_files(self.files)}\n#Change requested:\n{step}"],
                function_classification_type_map[function_classification]
            )

            logger.info(f"Tool response: \n {json.dumps(implemented_change, indent=4)}")

            filepath = find_closest_file(implemented_change["filepath"], list(self.files.keys()))
            aligned_original_lines = find_best_match(implemented_change["original_lines"], files[filepath]).block

            if function_classification == "insert_code_before":
                new_lines = implemented_change["lines_inserted_before"] + "\n" + aligned_original_lines
            elif function_classification == "insert_code_after":
                new_lines = aligned_original_lines + "\n" + implemented_change["lines_inserted_after"]
            elif function_classification == "search_and_replace_code":
                new_lines = implemented_change["new_replacement_lines"]
            else:
                logger.error(f"There was an invalid input: \n {function_classification} \n {json.dumps(implemented_change, indent=2)}")
                continue

            logger.info(f"Found filepath: \n {filepath}")
            logger.info(f"Found original lines: \n{aligned_original_lines}")
            logger.info(f"Found new lines: \n {new_lines}")

            if filepath in new_files:
                new_files[filepath] = new_files[filepath].replace(aligned_original_lines, new_lines)

        return new_files

    def full_generation_single_function(self): # Doesn't work that well
        new_files = self.files.copy()

        logger.info(f"Processing plan at one go")

        implemented_changes = self.aux_model.call_with_json(
            "You are an intelligent and diligent programming assistant. Write and implement the following change in the file. ",
            [f"# Original files:\n{stringify_files(self.files)}\n#Implement the following objective:\n{self.goal}"],
            FileModificationOperationList
        )

        logger.info(f"Tool response: \n {json.dumps(implemented_changes, indent=4)}")

        for implemented_change in implemented_changes["file_modifications"]:
            filepath = find_closest_file(implemented_change["filepath"], list(self.files.keys()))
            aligned_original_lines = find_best_match(implemented_change["location_lines"], files[filepath]).block
            new_lines = implemented_change["insertion_lines"]

            logger.info(f"Found filepath: \n {filepath}")
            logger.info(f"Found original lines: \n{aligned_original_lines}")
            logger.info(f"Found new lines: \n {new_lines}")

            if filepath in new_files:
                new_files[filepath] = new_files[filepath].replace(aligned_original_lines, new_lines)

        return new_files

    def implement_plan_blocks(self, plan):
        steps = re.compile('\n(?=[0-9].)').split(plan)
        for step in steps:
            file_change_prompt = f"# Original Files: \n {stringify_files(self.files)}\n **You must implement the following change** # Change: {step}"
            choice_description = prompts.EDITBLOCK_PROMPTS
            response = self.aux_model(choice_description, [file_change_prompt], temperature=0.1)
            changes = extract_xml_tags(response, "change")
            edited_file = self.get_single_file()
            logger.info(response)

            for change in changes:
                new_block = extract_xml_tags(change, "new")
                after_block = extract_xml_tags(change, "after")
                before_block = extract_xml_tags(change, "before")
                search_block = extract_xml_tags(change, "search")
                replace_block = extract_xml_tags(change, "replace")
                top_block = extract_xml_tags(change, "top")
                bottom_block = extract_xml_tags(change, "bottom")

                logger.info(f"Change: {change}")

                if len(new_block) == 1 and len(after_block) == 1:
                    matched_search_block = find_best_match(after_block[0], self.get_single_file()).block
                    replace_block = matched_search_block + "\n" + new_block[0]
                    edited_file = edited_file.replace(matched_search_block, replace_block)
                elif len(new_block) == 1 and len(before_block) == 1:
                    matched_search_block = find_best_match(before_block[0], self.get_single_file()).block
                    replace_block = new_block[0] + "\n" + matched_search_block
                    edited_file = edited_file.replace(matched_search_block, replace_block)
                elif len(search_block) == 1 and len(replace_block) == 1:
                    matched_search_block = find_best_match(search_block[0], self.get_single_file()).block
                    replace_block = replace_block[0]
                    edited_file = edited_file.replace(matched_search_block, replace_block)
                elif len(top_block) == 1:
                    top_block = top_block[0]
                    edited_file = top_block + "\n" + edited_file
                elif len(bottom_block) == 1:
                    bottom_block = bottom_block[0]
                    edited_file = edited_file + "\n" + bottom_block

            return edited_file

    def chain_lats_plan_and_execute(self):
        root = Node(self.files, Reflection(feedback="The goal has not yet been implemented.", score=0, found_solution=False))
        # TODO: Generate initial children with a specific prompt

        while root.height < 3 and not(root.is_solved):
            best_child = root.best_child
            if not(best_child):
                best_child = root

            plan_system_prompt = f"First, rewrite the goal to be more specific and expanded. Then, write a search-replace plan to achieve the following goal. Use reflection feedback as guidance on writing a better plan. Describe files and code changes in detail. \n Goal: {self.goal}"
            plan_user_prompt = [("user", f"# Original File: \n {stringify_files(self.files)} \n \n # Goal: {self.goal}"),
                                    ("assistant", f"# Most Recent Revision \n {stringify_files(best_child.content)}"),
                                    ("user", f"# Feedback \n {best_child.reflection.as_message}")]

            if best_child.content == self.files: # The first one was the initial node, just send the original file and the goal.
                plan_system_prompt = f"First, rewrite the goal to be more specific and expanded. Then, write a search-replace plan to achieve the following goal. Describe files and code changes in detail. \n Goal: {self.goal}"
                plan_user_prompt = [("user", f"# Original File: \n {stringify_files(self.files)} \n \n # Goal: {self.goal}")]

            candidate_plans = self.model([plan_system_prompt]*5, [plan_user_prompt]*5)
            rewrite_system_prompt = f"Based on the plan, rewrite each relevant file. Do not truncate code for brevity. Format each file rewrite in the following manner: \n [filepath]\n```language\ncode\n```."
            rewrite_user_prompt = [[("user", f"# Original Files:\n{stringify_files(self.files)}"),
                                    ("assistant", f"# Most Recent Revision:\n{stringify_files(best_child.content)}"),
                                    ("user", f"# Implementation Plan \n {candidate_plan}")] for candidate_plan in candidate_plans]
            if best_child.content == self.files:
                rewrite_user_prompt = [[("user", f"# Files:\n{stringify_files(self.files)}"),
                                ("user", f"# Implementation Plan \n {candidate_plan}")] for candidate_plan in candidate_plans]

            rewritten_files = self.model([rewrite_system_prompt]*5, rewrite_user_prompt)

            # TODO: Add the rewritten files to the messages as a new iteration
            parsed_files = [self.process_rewrite(rewrite) for rewrite in rewritten_files]

            # TODO: Evaluate the newly generated parsed files
            reflections = [self.score_code_output(parse_file) for parse_file in parsed_files]

            # TODO: Add the new files with evaluations and everything into the solution
            new_nodes = [Node(
                content=files,
                reflection=reflection,
                parent=best_child
            ) for (files, reflection) in zip(parsed_files, reflections)]
            best_child.children.extend(new_nodes)

            # TODO: DsPy so that search-replace output can be generated to save tokens and generate larger files
        return root.best_child.content

    def process_rewrite(self, output):
        logger.debug(f"Processing rewrite: \n {output}")

        start_time = time.time()

        new_files = {}
        pattern = r"(.*?)\n```\n([\s\S]*?)\n```"
        matches = re.findall(pattern, output)

        if matches:
            for match in matches:
                filepath = match[0]
                code = match[1]
                new_files[filepath] = code
        logger.info(f"Matches found: \n {matches}")
        logging.info(f"Matches for code block format: {matches}")
        dmp = dmp_module.diff_match_patch()
        dmp.Match_Threshold = 0.90
        dmp.Match_Distance = 10000

        code_block_pattern = r'```.*?```'
        without_code = re.sub(code_block_pattern, '', output, flags=re.DOTALL)

        patches = {}
        patched_files = self.files

        logging.info(f"Processed patch: {new_files}")

        for filepath in new_files:
            original_filepath = filepath
            trimmed_filepath = filepath.replace("[", "").replace("]", "").replace("*", "").strip()
            trimmed_filepath = trimmed_filepath.split(" ")[-1]
            cfilepath = find_closest_file(trimmed_filepath, list(self.files.keys()))
            cfilepath = cfilepath.strip()

            logger.info(f"Closest filepath found for {trimmed_filepath}: {cfilepath}")
            logging.info(f"Closest filepath found for {trimmed_filepath}: {cfilepath}")

            valid_file = True
            if cfilepath in self.files:
                original_file = self.files[cfilepath] + ""
                logger.info(f"Editing existing filepath: {cfilepath}")
                logging.info(f"Editing existing filepath: {cfilepath}")
            else:
                # Is the filepath plausible?
                try:
                    validate_filename(cfilepath)
                    original_file = ""
                    logger.info(f"Identified valid new file for: {cfilepath}")
                    logging.info(f"Identified valid new file for: {cfilepath}")
                except ValidationError as e:
                    valid_file = False
                    continue

            if not(valid_file):
                logger.info(f"File received is not valid file! {cfilepath}")
                logging.info(f"File received is not valid file! {cfilepath}")
                continue

            generated_diff = dmp.diff_main(original_file, new_files[original_filepath])
            dmp.diff_cleanupSemantic(generated_diff)
            generated_patches = dmp.patch_make(original_file, generated_diff)
            logger.info(f"Generated patches: {[str(generated_phrase) for generated_phrase in generated_patches]}")
            logging.info(f"Generated patches: {generated_patches}")
            patched_files[cfilepath] = dmp.patch_apply(generated_patches, original_file)[0]
            logger.info(f"Saved patch for patched_files: {cfilepath} \n \n {patched_files[cfilepath]}")
            patches[cfilepath] = dmp.patch_toText(generated_patches)

        end_time = time.time()
        logging.info(f"Finished processing with patches in: {end_time - start_time}")
        logger.info(f"Patched files: \n {patched_files.keys()} \n \n {stringify_files(patched_files)}")
        logging.info(f"Patched files: \n {patched_files.keys()} \n {stringify_files(patched_files)}")
        return patched_files


if __name__ == "__main__":
    from .utils.model import create_model
    from dotenv import load_dotenv
    import os

    load_dotenv(".env")
    logging.basicConfig(level=logging.INFO)

    rewrite_file = open("test.txt", "r").read()


    filepaths = ["/Users/vijaydaita/Files/projects/microapps/remove-background/src/App.js", "/Users/vijaydaita/Files/projects/microapps/remove-background/src/index.js"]
    goal = "Edit the file so that a modal appears when the quiz finishes. The modal should display the score and have a refresh button."
    files = {filepath: open(filepath, "r").read() for filepath in filepaths}
    model = create_model(os.environ["OPENAI_API_KEY"], "gpt-3.5-turbo", temperature=0.5, max_tokens=3092)
    # # model = create_model(os.environ["TOGETHER_API_KEY"], "mistralai/Mixtral-8x7B-Instruct-v0.1", base_url="https://api.together.xyz/", temperature=0.8, max_tokens=3092)
    # aux_model = create_model(os.environ["TOGETHER_API_KEY"], "mistralai/Mixtral-8x7B-Instruct-v0.1", base_url="https://api.together.xyz/", temperature=0.1, max_tokens=3092)

    executor = Executor(goal, files, "", model, aux_model=model)
    executor.process_rewrite(rewrite_file)
    # plan = open("/Users/vijaydaita/Files/projects/superdocs/superdocs-python/superdocs_python/train-examples/plan.txt", "r").read()
    # generated_file = executor.full_generation_single_function()
    # logger.info(stringify_files(generated_file))
    # logger.info(executor.chain_execute_block_edits())
