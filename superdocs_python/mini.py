"""
While being able to provide a UI is useful in making this more accessible, there were some major errors with the processing side of things that I wanted to fix.
This mini file will make iterating on things much quicker.
"""
import re

from rapidfuzz import fuzz, distance
from tqdm import tqdm
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import os
import string

import asyncio 
import aiohttp
import ssl
import certifi
import time
import json

from langchain_core.output_parsers import BaseOutputParser

from utils.text_lats import LATS
from utils.file_lats import FileLATS

from multiprocess import Pool

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

async def call_chatgpt_async(session, messages, model_name, key): # https://medium.com/@nitin_l/parallel-chatgpt-requests-from-python-6ab48cc2a610
    payload = {
        'model': model_name,
        'messages': messages
    }
    try:
        start_time = time.time()
        async with session.post(
            url='https://api.openai.com/v1/chat/completions',
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json=payload,
            ssl=ssl.create_default_context(cafile=certifi.where())
        ) as response:
            response = await response.json()
        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        end_time = time.time()
        print("Time spent on request: ", (end_time - start_time))
        return response['choices'][0]['message']['content']
    except:
        print("Request failed.")

async def bulk_call(message_sets, model_name, api_key):
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(call_chatgpt_async(session, message_set, model_name, api_key)) for message_set in message_sets]
        responses = await asyncio.gather(*tasks)
    return responses

def create_model(api_key, model_name, base_url="https://api.openai.com/v1", base_temperature=0.2, base_max_tokens=2048): # I don't want to pass the model name in separately
    model = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    def run_model(system_prompt, messages, temperature=base_temperature, max_tokens=base_max_tokens):
        print(len(system_prompt), len(messages))
        if type(system_prompt) == list: # That means that multiple requests should be made
            print("Bulk processing")
            message_sets = []
            for message_index in range(len(system_prompt)):
                new_messages =[
                        {"role": "system", "content": system_prompt[message_index]},
                ] + [{"role": "user", "content": message} for message in messages[message_index]]
                # print(json.dumps(new_messages, indent=4))
                message_sets.append(new_messages)
            results = asyncio.run(bulk_call(message_sets, model_name, api_key))
            return results
        else:
            response = model.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                ] + [ {"role": "user", "content": message} for message in messages],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

    return run_model

# model = create_model(os.environ["OPENAI_API_KEY"], "gpt-4-turbo-preview")
model = create_model(os.environ["OPENAI_API_KEY"], "gpt-3.5-turbo")
lats = LATS(os.environ["OPENAI_API_KEY"])
# model = create_model(os.environ["OPENROUTER_API_KEY"], "anthropic/claude-3-sonnet:beta", base_url="https://openrouter.ai/api/v1")

# plan_model = create_model(os.environ["TOGETHER_API_KEY"], "deepseek-ai/deepseek-coder-33b-instruct", base_url="https://api.together.xyz/")


PLAN_PROMPT = """
Generate a plan that could be implemented by a junior developer for solving the user's goal. 
Be as specific as possible, especially with regards to imports and variable definitions. 
Describe your changes with minimal code - only when necessary.
Think step-by-step.
"""

PLAN_EVAL_PROMPT = """
Evaluate the following plan in its effectiveness to solve the given goal and the ease of implementability by a junior developer. 
Think step-by-step. Assign a single numerical score between 0 and 10 and enclose within <score></score>. 
Enclose specific feedback on how to improve within <feedback> </feedback> tags.
"""

EVALUATION_PROMPT = """
A code bot made some changes to a codebase to achieve the specified goal. You have to evaluate the file as a whole and try to identify any bugs or errors. There are likely to be bugs, so watch out.

First, you should give a score ranging from 1 to 10 based on how effectively the snippets implement or solve the objective and their ability to be integrated.

Here is a description of each rating:
1: This code would not compile and would throw errors if actually used, it does not implement any of the feature requested. It actively removes functionality from the existing application.
3: Some steps were made towards properly implementing the steps but it is not completed.
5: Some steps were made towards properly implementing the steps and it is about halfway completed.
8: This code would compile properly and adequately satisfies the requirements listed.
10: This code would compile and fully satisfies the requirements listed.

Finally, you should provide feedback about specific pitfalls or errors that are in the code and what should be done to solve them. Make your suggestions and minimal as possible so that it is easy to implement and so that new errors are not introduced.
Format in the following manner: 

USER: <some code snippets, objective, etc.>
ASSISTANT:
<score>your number here </score>
<feedback>
1. first point
2. second point
</feedback>
"""

EXECUTE_PROMPT = """
Act as an expert software developer.
You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
Always use best practices when coding.
Respect and use existing conventions, libraries, etc that are already present in the code base.
Do not truncate code for brevity.

Take requests for changes to the supplied code.
If the request is ambiguous, ask questions.

Once you understand the request you MUST:
1. List the files you need to modify. Only suggest changes to a *read-write* files. Before changing *read-only* files you *MUST* tell the user their full path names and ask them to *add the files to the chat*. End your reply and wait for their approval.
2. Think step-by-step and explain the needed changes with a numbered list of short sentences.
3. Describe each change with a *SEARCH/REPLACE block* per the examples below.

All changes to files must use this *SEARCH/REPLACE block* format.

# Example conversation 1:

## USER: These are the *read-write* files:

mathweb/flask/app.py
{fence[0]}python
from flask import Flask

app = Flask(__name__)

@app.route('/div/<int:n>/<int:d>')
def divide(n, d):
    if d == 0:
        return 'Error: Division by zero'
    else:
        result = n / d
        return str(result)

def factorial(n):
    "compute factorial"

    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

@app.route('/factorial/<int:n>')
def get_factorial(n):
    return str(factorial(n))

if __name__ == '__main__':
    print("Starting...")
    app.run()
{fence[1]}

## ASSISTANT: Ok.

## USER: Change get_factorial() to use math.factorial

## ASSISTANT:

Do make this change we need to modify `mathweb/flask/app.py` to:

1. Import the math package.
2. Remove the existing factorial() function.
3. Update get_factorial() to call math.factorial instead.

Here are the *SEARCH/REPLACE* blocks:

mathweb/flask/app.py
{fence[0]}python
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
{fence[1]}

mathweb/flask/app.py
{fence[0]}python
<<<<<<< SEARCH
def factorial(n):
    "compute factorial"

    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

=======
>>>>>>> REPLACE
{fence[1]}

mathweb/flask/app.py
{fence[0]}python
<<<<<<< SEARCH
    return str(factorial(n))
=======
    return str(math.factorial(n))
>>>>>>> REPLACE
{fence[1]}

# Example conversation 2:

## USER: 
# Goal: Refactor hello() into its own file.
# Files:

Filepath: main.py
{fence[0]}python
def hello():
    "print a greeting"

    print("hello")

if __name__ == '__main__':
    hello()
{fence[1]}

## ASSISTANT:

Do make this change we need to modify `main.py` and make a new file `hello.py`:

1. Make a new hello.py file with hello() in it.
2. Remove hello() from main.py and replace it with an import.

Here are the *SEARCH/REPLACE* blocks:

hello.py
{fence[0]}python
<<<<<<< SEARCH
=======
def hello():
    "print a greeting"

    print("hello")
>>>>>>> REPLACE
{fence[1]}

main.py
{fence[0]}python
<<<<<<< SEARCH
def hello():
    "print a greeting"

    print("hello")
=======
from hello import hello
>>>>>>> REPLACE
{fence[1]}

# Rules

Every *SEARCH/REPLACE block* must use this format:
1. The file path alone on a line, eg: main.py
2. The opening fence and code language, eg: {fence[0]}python
3. The start of search block: <<<<<<< SEARCH
4. A contiguous chunk of lines to search for in the existing source code
5. The dividing line: =======
6. The lines to replace into the source code
7. The end of the replace block: >>>>>>> REPLACE
8. The closing fence: {fence[1]}

Every *SEARCH* section must *EXACTLY MATCH* the existing source code, character for character, including all comments, docstrings, etc.

Include *ALL* the code being searched and replaced!

Only *SEARCH/REPLACE* files that are *read-write*.

To move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.

If you want to put code in a new file, use a *SEARCH/REPLACE block* with:
- A new file path, including dir name if needed
- An empty `SEARCH` section
- The new file's contents in the `REPLACE` section

You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
"""

def find_closest_file(filepath, all_filepaths):
    best_match = Match("", -1)
    for fp in all_filepaths:
        score = fuzz.ratio(filepath, fp)
        if score > best_match.score:
            best_match = Match(fp, score)
    
    return best_match.block if best_match.score > 0.7 else filepath
        

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

def stringify_files(file_dictionary):
        file_string = ""
        for file in file_dictionary:
            file_string += "Filepath: " + file + "\n ----- \n"
            ext = file.split(".")[-1]
            file_string += f"```{ext} \n {file_dictionary[file]} \n ```"
        return file_string

def parse_search_replace_blocks(text):
    pattern = re.compile(r'''
        <<<<<<< SEARCH ^(?P<filepath>[^\n]+) \n
        (?P<search_block>.*?)
        =======\n
        (?P<replace_block>.*?)
        >>>>>>> REPLACE\n
    ''', re.MULTILINE | re.DOTALL | re.VERBOSE)

    blocks = []
    for match in pattern.finditer(text):
        blocks.append(
            SearchReplaceChange(
                filepath=match.group("filepath").strip(),
                search_block=match.group("search_block").strip(),
                replace_block=match.group("replace_block").strip()
            )
        )

    return blocks

def extract_code_block_data(md_text, language):
   # Regular expression pattern for matching diff code blocks
   pattern = rf'```{language}([\s\S]*?)```'
   code_blocks = re.findall(pattern, md_text, re.MULTILINE)
   return code_blocks

def extract_xml_tags(text, tag):
    pattern = r'<' + tag + '>(.*?)</' + tag + '>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def evaluate_generation(goal, context, annotated_modified_files):
    output = model(EVALUATION_PROMPT, ["Goal: " + goal, "Context: " + context, stringify_files(annotated_modified_files)])
    score = extract_xml_tags(output, "score")
    feedback = extract_xml_tags(output, "feedback")
    if len(score) == 0: # just assume the plan is mid
        score = ["5"]
    score, feedback = float(score[0]), feedback[0]
    print("Scoring determination made: ", f"Overall: {score}", f"Feedback: {feedback}")
    return score, feedback

class Executor: # Uses an objective, general context for information, and a bunch of current files as context
    def __init__(self, goal, files, context):
        self.goal = goal
        self.context = context
        self.files = files
    
    def chain_execute(self):
        previous_instruction = ""
        previous_comments = ""

        for _ in range(2):
            new_goal = model("""Rewrite the following goal, given the context, to be more specific and actionable for a developer agent. 
                             Do not write code under any circumstance. Writing code will short-circuit the execution process. Make sure that no code is written whatsoever.
                             Rewrite the goal portion, and the goal only.""",
                             [f"# Goal \n {self.goal} \n ------ # Context \n {self.context} \n ------ # Files \n {stringify_files(self.files)}"])
            print("New goal")
            print(new_goal)
            self.goal = new_goal

        plan = lats.run(f"""
                            You are the senior developer. Your junior developer can't test the application themself.
                            Given the files and context, provide an extremely detailed plan of code changes to be performed to your junior developer on ensure the goal is completed accurately.
                            If the change is accurately completed, don't suggest any further changes or optimizations of any form.
                            Make your instructions as simple to implement as possible, so that a beginner programmer can implement it in a simple and error-prone way.  
                            # Goal
                            {self.goal} 
                            ------
                            # Context
                            {self.context} 
                            ------
                            # Original Files
                            {stringify_files(self.files)}
                            """)
        print("STARTING PLAN")
        print(plan)

        input_string = f"""# Follow the instructions in the plan and apply the changes to the files. \n {plan} \n # Goal: \n {self.goal}
# Context:
{self.context}
# Original Files:
{stringify_files(self.files)}
        """

        print(input_string)

        file_lats = FileLATS(os.environ["OPENAI_API_KEY"], self.apply_diff_output_wholefile)
        # file_lats = FileLATS(os.environ["OPENAI_API_KEY"], lambda x: x)
        best_file = file_lats.run(input_string)

        print(best_file)

    def apply_diff_output_wholefile(self, output):
        print(output)
        pattern = r'(?s)(.*?)\n```(.*?)\n(.*?)\n```'
        matches = re.findall(pattern, output)
        triplets = [(match[0], match[1], match[2]) for match in matches]
        new_files = {}
        for triple in triplets:
            new_files[triple[0]] = triple[2]
        

        code_block_pattern = r'```.*?```'
        without_code = re.sub(code_block_pattern, '', output, flags=re.DOTALL)
        
        return f"""Generated output: {without_code} \n \n {stringify_files(new_files)}"""


    def apply_diff_output(self, output):
        print(output)
        sr_blocks = parse_search_replace_blocks(output)

        code_block_pattern = r'```.*?```'
        without_code = re.sub(code_block_pattern, '', output, flags=re.DOTALL)
        
        original_files = self.files.copy()
        modified_files = self.files.copy()
        annotated_modified_files = self.files.copy() # Not going to be returned directly to the users, but is going to show the LLM which lines were modified. 

        for block in sr_blocks:
            if len(block.search_block.strip()) == 0 and len(block.replace_block.strip()) == 0:
                continue

            match_filepath = find_closest_file(block.filepath, list(self.files.keys()))
            print("Trying to match file that's in: ", match_filepath)
            best_match = find_best_match(block.search_block, original_files[match_filepath])
            if best_match.score > 550:
                modified_files[match_filepath] = modified_files[match_filepath].replace(best_match.block, block.replace_block)

                annotated_replace_block = "\n".join(f"+ {line}" for line in block.replace_block.splitlines())
                annotated_modified_files[match_filepath] = modified_files[match_filepath].replace(best_match.block, annotated_replace_block)

                print("Making replacement: ")
                print("=====SEARCH=====")
                print(block.search_block)
                print(f"=====MATCH with closeness {best_match.score}======")
                print(best_match.block)
                print("=====REPLACE=====")
                print(block.replace_block)
            else:
                print("Failed to match:")
                print(block.search_block)

        return f"""
        Generated output: 
        {without_code}
        Outputted files:
        {stringify_files(modified_files)}
        """
def test_matcher():
    filepath = "base_page.txt"
    contents = open(filepath, "r").read()
    match_string = """ setScore(score);
         setQuizFinished(true);
         setIsModalOpen(true); // Open modal on quiz completion
         setPaused(true);
    """
    match = find_best_match(match_string, contents)
    print(match.block)
    print(match.score)

def test():
    filepath = "/Users/vijaydaita/Files/uiuc/rxassist/rxassist/src/app/main/page.tsx"
    goal = "Edit the file so that a modal appears when the quiz finishes. The modal should display the score and have a refresh button."  
    files = {filepath: open(filepath, "r").read()}
    print("Starting file: ")
    print(stringify_files(files))
    executor = Executor(goal, files, "")  
    executor.chain_execute()

if __name__ == "__main__":
    test()
    # test_matcher()