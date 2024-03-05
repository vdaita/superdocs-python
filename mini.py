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

EXECUTE_PROMPT = """
Act as an expert software developer.
You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
Always use best practices when coding.
Respect and use existing conventions, libraries, etc that are already present in the code base.
Make sure you do not truncate for brevity.

If there are no changes that need to be made, return DONE. 

Take requests for changes to the supplied code.
If the request is ambiguous, ask questions.

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

            # Weighting the first and last lines by 5x
            score += 5*fuzz.ratio(original_lines[start_line], query_lines[0])
            score += 5*fuzz.ratio(original_lines[end_line], query_lines[-1])

            # if score > 100:
            #     print(f"======SIMILARITY SCORE {score}======")
            #     print(f"===SEARCH===")
            #     print(snippet_from_query)
            #     print("Stripped: ", stripped_query)
            #     print("===MATCH===")
            #     print(snippet_from_original)
            #     print("Stripped: ", stripped_original)
        
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

def execute(goal, context, files, model): # TODO: make a class out of these.
    output = model(EXECUTE_PROMPT, ["Objective: " + goal, "Context: " + context])

    diff_blocks = extract_code_block_data(output, "diff")

    sr_blocks = []
    for block in diff_blocks:
        sr_blocks += parse_diff(block)
    
    original_files = {}
    modified_files = {}
    for block in sr_blocks:
        match_filepath = find_closest_file(block.filepath, files)
        if not(match_filepath in original_files):
            print("Reading and setting: ", match_filepath, " out of ", files)
            original_files[match_filepath] = open(match_filepath, "r").read()
            modified_files[match_filepath] = original_files[match_filepath]

    for block in sr_blocks:
        if len(block.search_block.strip()) == 0 and len(block.replace_block.strip()) == 0:
            continue

        match_filepath = find_closest_file(block.filepath, files)
        print("Trying to match file that's in: ", match_filepath)
        best_match = find_best_match(block.search_block, original_files[match_filepath])
        if best_match.score > 0.7:
            modified_files[match_filepath] = modified_files[match_filepath].replace(best_match.block, block.replace_block)
            print("Making replacement: ")
            print("=====SEARCH=====")
            print(block.search_block)
            print(f"=====MATCH with closeness {best_match.score}======")
            print(best_match.block)
            print("=====REPLACE=====")
            print(block.replace_block)
    
    return modified_files

def chain_execute(goal, filepath, model):
    for step_i in range(3):
        contents = open(filepath, "r").read()
        context = f"Filepath: {filepath} \n ----- \n {contents}"
        plan = model(
            "Generate a step-by-step plan that can be followed sequentially to make the desired code change or fix by the user. Ensure that you explicitly enumerate all the variables that need to be defined. Don't write any code yourself.",
            [f"Objective: {goal}", f"Context: {context}"]
        )

        context += "\nPlan: \n" + plan

        print("===== CONTEXT =====")
        print(context)

        modified_files = execute(goal, context, [filepath], model)
        if len(modified_files) == 0:
            break
        for filepath in modified_files:
            file = open(filepath, "w+")
            file.write(modified_files[filepath])
            print(f"REWRITING {filepath}")
            print(modified_files[filepath])
            file.close()

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
    model = create_model(
            os.environ["OPENAI_API_KEY"],
            "gpt-4-0125-preview"
        )
    
    filepath = "/Users/vijaydaita/Files/uiuc/rxassist/rxassist/src/app/main/page.tsx"
    goal = "Now, change the Card into a Modal for when the quiz finishes."    
    chain_execute(goal, filepath, model)

if __name__ == "__main__":
    test()
    # test_matcher()