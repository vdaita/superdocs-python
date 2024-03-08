"""
While being able to provide a UI is useful in making this more accessible, there were some major errors with the processing side of things that I wanted to fix.
This mini file will make iterating on things much quicker.
"""
import re
from dataclasses import dataclass

from rapidfuzz import fuzz, distance
from tqdm import tqdm
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import os
import string

from multiprocessing import Pool

load_dotenv(".env")

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

def create_model(api_key, model_name, base_url="https://api.openai.com/v1", base_temperature=0.1, base_max_tokens=2048): # I don't want to pass the model name in separately
    model = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    def run_model(system_prompt, messages, temperature=base_temperature, max_tokens=base_max_tokens):
        print(len(system_prompt), len(messages))
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

# model = create_model(os.environ["OPENAI_API_KEY"], "gpt-4-0125-preview")
model = create_model(os.environ["OPENROUTER_API_KEY"], "anthropic/claude-3-opus:beta", base_url="https://openrouter.ai/api/v1")

EVALUATION_PROMPT = """
A code bot made some changes to a codebase to achieve the specified goal. <ASSISTANT EDITS> tags indicate that you should pay special attention to those portions, as those were where edits were made.
Evaluate the changes made with the following criteria and output in the desired manner. 

You should evaluate the following criteria:
1. Simplicity: are the changes as minimal and simple as possible? This should be a 1 to 10 value.
2. Functionality: will the changes be functional in achieving the user's goal? This should be a 1 to 10 value.
3. Integration: will the changes cause a compilation error or another basic error (for example, misplaced braces creating syntax errors)? This should be a 0 or 1 value.

Finally, you should provide feedback about specific pitfalls and what could be done to solve them. Make your suggestions and minimal as possible so that it is easy to implement and so that new errors are not introduced.
Format in the following manner: 

USER: <some code snippet>
ASSISTANT:
<simplicity>5</simplicity>
<functionality>8</functionality>
<integration>1</integration>
<feedback>
Next steps:
1. Don't use a separate variable to keep track of whether or not the variable should be a certain way - use the existing value. 
2. Make sure that the contrast of the text is there when presenting against a certain background.
3. When showing the modal, you used "&&" instead of "?" and that could cause some erros, fix that.
</feedback>
"""

EXECUTE_PROMPT = """
Act as an expert software developer.
You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!

Rigorously think through, check for any potential bugs within the code, and then diligently fix them.

Always use best practices when coding.
Respect and use existing conventions, libraries, etc that are already present in the code base.
Make sure you do not truncate for brevity.

If there are no changes that need to be made, return DONE. 

Take requests for changes to the supplied code.
If the request is ambiguous, ask questions.

First, before making your diff edits, write out a quick plan (without using code), describing the changes that you are going to make in the form of a step-by-step list.

For each file that needs to be changed, write out the changes similar to a unified diff like `diff -U0` would produce. For example:

# Example conversation 1

## USER: Replace is_prime with a call to sympy.

## ASSISTANT: Ok, I will:

1. Add an imports of sympy.
2. Remove the is_prime() function.
3. Replace the existing call to is_prime() with a call to sympy.isprime().

Here are the diffs for those changes:

```diff
--- mathweb/flask/app.py
+++ mathweb/flask/app.py
@@ ... @@
-class MathWeb:
+import sympy
+
+class MathWeb:
@@ ... @@
-def is_prime(x):
-    if x < 2:
-        return False
-    for i in range(2, int(math.sqrt(x)) + 1):
-        if x % i == 0:
-            return False
-    return True
@@ ... @@
-@app.route('/prime/<int:n>')
-def nth_prime(n):
-    count = 0
-    num = 1
-    while count < n:
-        num += 1
-        if is_prime(num):
-            count += 1
-    return str(num)
+@app.route('/prime/<int:n>')
+def nth_prime(n):
+    count = 0
+    num = 1
+    while count < n:
+        num += 1
+        if sympy.isprime(num):
+            count += 1
+    return str(num)
```

# File editing rules:

Return edits similar to unified diffs that `diff -U0` would produce.

Make sure you include the first 2 lines with the file paths.
Don't include timestamps with the file paths.

Include headers, top-level variable definitions, imports, etc.

Start each hunk of changes with a `@@ ... @@` line.
Don't include line numbers like `diff -U0` does.
The user's patch tool doesn't need them.

The user's patch tool needs CORRECT patches that apply cleanly against the current contents of the file!
Think carefully and make sure you include and mark all lines that need to be removed or changed as `-` lines.
Make sure you mark all new or modified lines with `+`.
Don't leave out any lines or the diff patch won't apply correctly.

Indentation matters in the diffs!

Start a new hunk for each section of the file that needs changes.

Only output hunks that specify changes with `+` or `-` lines.
Skip any hunks that are entirely unchanging ` ` lines.

Output hunks in whatever order makes the most sense.
Hunks don't need to be in any particular order.

Ensure that all variables are appropriately defined. 

When editing a function, method, loop, etc use a hunk to replace the *entire* code block.
Delete the entire existing version with `-` lines and then add a new, updated version with `+` lines.
This will help you generate correct code and correct diffs.

To move code within a file, use 2 hunks: 1 to delete it from its current location, 1 to insert it in the new location.

To make a new file, show a diff from `--- /dev/null` to `+++ path/to/new/file.ext`.

You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
Please do not truncate code for brevity.

For each hunk in your diff, write at least 10 lines of original code to provide context.
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

def extract_code_block_data(md_text, language):
   # Regular expression pattern for matching diff code blocks
   pattern = rf'```{language}([\s\S]*?)```'
   code_blocks = re.findall(pattern, md_text, re.MULTILINE)
   return code_blocks

def extract_xml_tags(text, tag):
    pattern = r'<' + tag + '>(.*?)</' + tag + '>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def stringify_files(file_dictionary):
        file_string = ""
        for file in file_dictionary:
            file_string += "Filepath: " + file + "\n ----- \n"
            file_string += file_dictionary[file]
        return file_string

def evaluate_generation(goal, context, annotated_modified_files):
    output = model(EVALUATION_PROMPT, ["Goal: " + goal, "Context: " + context, stringify_files(annotated_modified_files)])
    simplicity = extract_xml_tags(output, "simplicity")
    functionality = extract_xml_tags(output, "functionality")
    integration = extract_xml_tags(output, "integration")
    feedback = extract_xml_tags(output, "feedback")
    
    if len(simplicity) == 0 or len(functionality) == 0 or len(integration) == 0 or len(feedback) == 0:
        return -1

    simplicity, functionality, integration, feedback = int(simplicity[0]), int(functionality[0]), int(integration[0]), feedback[0]

    score = 10 * (((simplicity * 0.25 + functionality) * integration)/(10 * 0.25 + 10))
    print("Scoring determination made: ", f"Simplicity {simplicity}", f"Functionality: {functionality}", f"Integration: {integration}", f"Overall: {score}", f"Feedback: {feedback}")
    return score, feedback

class Executor: # Uses an objective, general context for information, and a bunch of current files as context
    def __init__(self, goal, files, context):
        self.goal = goal
        self.context = context
        self.files = files

    def chain_execute(self):
        initial_modifications = [self.execute() for _ in range(3)]
        best_modifications = self.files
        best_modifications_score = 0
        best_modifications_feedback = ""

        for modification in initial_modifications:
            modified_files, annotated_modified_files = modification["unannotated"], modification["annotated"]
            print("Received modification: ")
            print(stringify_files(modified_files))
            print("=============")
            print(stringify_files(annotated_modified_files))
            score, feedback = evaluate_generation(self.goal, self.context, annotated_modified_files)
            if score > best_modifications_score:
                best_modifications_score = score
                best_modifications_feedback = feedback
                best_modifications = modified_files

        # Second step of refining the answer
        self.files = best_modifications
        self.goal += f"Make sure to fix think through and fix all possible bugs and use the following feedback to improve your response. In your plan, ensure that every step is clearly specified. Change only what's necessary. Here is some additional feedback to help you: {best_modifications_feedback}"
        
        second_modifications_round = [self.execute() for _ in range(3)]
        for modification in second_modifications_round:
            modified_files, annotated_modified_files = modification["unannotated"], modification["annotated"]
            print("Received modification: ")
            print(stringify_files(modified_files))
            print("=============")
            print(stringify_files(annotated_modified_files))
            score, feedback = evaluate_generation(self.goal, self.context, annotated_modified_files)
            if score > best_modifications_score:
                best_modifications_score = score
                best_modifications_feedback = feedback
                best_modifications = modified_files

        for filepath in best_modifications:
            file = open(filepath, "w")
            file.write(best_modifications[filepath])
            file.close()


    def execute(self):
        output = model(EXECUTE_PROMPT, ["Objective: " + self.goal, "Context: " + self.context, "Files: " + stringify_files(self.files)])

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
        
        return {"unannotated": modified_files, "annotated": annotated_modified_files} # [0] are the actual modified files and [1] are the annotated_modified_files

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
    goal = "Now, change the Box into a Modal for when the quiz finishes, which will display a score and a button that reloads the page."  
    files = {filepath: open(filepath, "r").read()}
    print("Starting file: ")
    print(stringify_files(files))
    executor = Executor(goal, files, "")  
    executor.chain_execute()

if __name__ == "__main__":
    test()
    # test_matcher()