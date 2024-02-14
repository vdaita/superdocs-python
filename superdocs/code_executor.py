from thefuzz import fuzz
import re
# from .repo import find_closest_file
from unidiff import PatchSet
import json
import os
# from .sweep_search_and_replace import find_best_match
from superdocs.utils.codebase import find_closest_file
from utils.search_and_replace import find_best_match
from utils.gpt_output_utils import extract_code_block_data, extract_xml_tags

def process_with_search_replace_blocks(model, directory, input_text):
    response = model.chat.completions.create(
        model="gpt-4-turbo-0116",
        messages=[{
            "role": "system",
            "content": SEARCH_REPLACE_PROMPT
        }, {
            "role": "user",
            "content": input_text
        }],
        max_tokens=2048,
        temperature=0.1
    )
    response = response.choices[0].message.content
    blocks = extract_xml_tags(response, "block")

    changes = []

    for block in blocks: # do only one search and replace portion at a given time
        filename = extract_xml_tags(block, "filename")
        search_portion = extract_xml_tags(block, "search")
        replace_portion = extract_xml_tags(block, "replace")

        filename = find_closest_file(filename)
        full_filepath = os.path.join(directory, filename)
        lines = open(full_filepath, "r").read().splitlines()


        search_best_match = find_best_match(search_portion, full_filepath)
        search_portion = "\n".join(lines[search_best_match.start:search_best_match.end])
        changes.append({
            "filename": filename,
            "original": search_portion,
            "replace": replace_portion
        })

    return changes


def process_with_diffs(model, directory, input_text):
    response = model.chat.completions.create(
        model="gpt-4-turbo-0116",
        messages=[{
            "role": "system",
            "content": DIFF_PROMPT
        }, {
            "role": "user",
            "content": input_text
        }],
        max_tokens=2048,
        temperature=0.1
    )
    response = response.choices[0].message.content
    diffs = extract_code_block_data(response, "diff")
    changes = []
    for diff in diffs:
        changes.extend(fuzzy_process_diff(directory, diff))
    return changes


def fuzzy_process_diff(directory, diff, verbose=False): # expects the contents of a fenced block.
    changes = parse_diff(diff)

    for change_index, change in enumerate(changes):
        print("Currently processing change: ", change_index, " with values: ", json.dumps(change, indent=4))

        # Filepath:
        change["filepath"] = find_closest_file(directory, change["filepath"], threshold=30)
        full_filepath = directory + "/" + change["filepath"]
        full_filepath = full_filepath.replace("//", "/")
        file = open(full_filepath, "r+")
        file_contents = file.read()
        file_lines = file_contents.split("\n")
        
        if len(change["original"]) == 0:
            changes[change_index]["original"] = ""
            changes[change_index]["new"] = "\n".join([row[1] for row in change["new"]])
            continue

        if verbose:
            print("Change: ", json.dumps(change, indent=4))

        # Process the original
        # Everything here should be original
        original_string = ("\n".join([row[1] for row in change["original"]])).rstrip()
        if verbose:
            print("Found a match for: ", original_string)
        original_string = find_best_match(original_string, file_contents)
        if verbose:
            print("         ", original_string)
        original_string = "\n".join([line for line in file_lines[original_string.start:original_string.end]])

        # Process the replacement
        replacement_string = ""
        current_string = ""
        current_type = ""

        for (row_type, row_content) in change["new"]:
            if not(current_type == row_type): # this just changed type
                if verbose:
                    print("processing current chunk: ", current_string, current_type)
                if len(current_string) > 0:
                    if current_type == "original":
                        # get the match of this segment
                        best_match = find_best_match(current_string, file_contents)
                        lines = "\n".join([line for line in file_lines[best_match.start:best_match.end]])

                        replacement_string += lines
                    else:
                        replacement_string += current_string + "\n"
                current_string = ""
            current_string += row_content + "\n"
            current_type = row_type

        if current_type == "original":
            replacement_string += current_string + "\n"
        replacement_string = replacement_string.rstrip()

        change["original"] = original_string
        change["new"] = replacement_string  

        changes[change_index] = change      
        
    matched_diffs = []

    if verbose:
        print("Changes: ", json.dumps(changes, indent=4))

    for change in changes:
        filepath = change["filepath"]
        old_code = change["original"].rstrip()
        new_code = change["new"].rstrip()

        if len(old_code) > 0 and len(new_code) > 0:
            matched_diffs.append(
                {
                    "filepath": filepath,
                    "old": old_code,
                    "new": new_code
                }
            )

    if verbose: 
        print("Matched diffs: ", json.dumps(matched_diffs, indent=4))
        
    return matched_diffs

def parse_diff(diff_string):
    diff_string = diff_string.replace("...", "\n...")
    lines = diff_string.split("\n")
    changes = []
    current_filepath, current_to_remove, current_to_write = "", [], []

    previous_symbol = ""

    for line in lines:
        if line.startswith("+++"):
            if len(current_filepath) > 0:
                changes.append({
                    "filepath": current_filepath,
                    "original": current_to_remove,
                    "new": current_to_write
                })
                current_filepath, current_to_remove, current_to_write = "", [], []
            current_filepath = line[3:].rstrip()
            print("Processing new filepath: ", current_filepath)
        elif line.rstrip().endswith("@@"):
            if len(current_to_remove) > 0 or len(current_to_write) > 0:
                changes.append({
                    "filepath": current_filepath,
                    "original": current_to_remove,
                    "new": current_to_write
                })
                current_to_remove, current_to_write = [], []
        elif line.startswith("-"):
            if not(line.startswith("---")):
                current_to_remove.append(("original", line[1:]))
            if previous_symbol == "+":
                # save the prevous chunk that was created
                changes.append({
                    "filepath": current_filepath,
                    "original": current_to_remove,
                    "new": current_to_write
                })
                current_to_remove = []
                current_to_write = []
            previous_symbol = "-"
        elif line.startswith("+"):
            current_to_write.append(("new", line[1:]))
            previous_symbol = "+"
        else:
            current_to_remove.append(("original", line))
            current_to_write.append(("original", line))


    changes.append({
                    "filepath": current_filepath,
                    "original": current_to_remove,
                    "new": current_to_write
                })
    return changes

SEARCH_REPLACE_PROMPT = """
Act as an expert software developer.
You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
Always use best practices when coding.
Respect and use existing conventions, libraries, etc that are already present in the code base.
Make sure you do not truncate for brevity.

Take requests for changes to the supplied code.
If the request is ambiguous, ask questions.

For each file that needs to be changed, write a series of search replace blocks.

Please do not truncate code for brevity.

# Example conversation 1

## USER: Replace is_prime with a call to sympy

## ASSISTANT: Ok, I will: 

1. Add an imports of sympy.
2. Remove the is_prime() function.
3. Replace the existing call to is_prime() with a call to sympy.isprime().

Here are the search-replace blocks for those changes:

<blocks>

<block>
<filename>mathweb/flask/app.py</filename>
<search>
class MathWeb:
</search>
<replace>
import sympy

class MathWeb:
</replace>
</block>

<block>
<filename></filename>
<search></search>
<replace></replace>
</block>
</blocks>
"""

DIFF_PROMPT = """
Act as an expert software developer.
You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
Always use best practices when coding.
Respect and use existing conventions, libraries, etc that are already present in the code base.
Make sure you do not truncate for brevity.

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

When editing a function, method, loop, etc use a hunk to replace the *entire* code block.
Delete the entire existing version with `-` lines and then add a new, updated version with `+` lines.
This will help you generate correct code and correct diffs.

To move code within a file, use 2 hunks: 1 to delete it from its current location, 1 to insert it in the new location.

To make a new file, show a diff from `--- /dev/null` to `+++ path/to/new/file.ext`.

You are diligent and tireless!
You NEVER leave comments describing code without implementing it!
You always COMPLETELY IMPLEMENT the needed code!
Please do not truncate code for brevity.
"""