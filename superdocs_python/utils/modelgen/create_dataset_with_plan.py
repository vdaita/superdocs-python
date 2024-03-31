from brx import BRX, sftoq, uif
from superdocs_python.utils.diff_utils import parse_diff
from datasets import load_dataset
from dotenv import load_dotenv
import os
import json
from superdocs_python.utils.model import create_model
import tiktoken
from datasets import Dataset

# the prompt for generating is "Suggest rewrites that..."
#
# The output should first be a brief plan and then an execution based on the results

BRX_SCHEMA = {"description":"Generate a plan given a set of inputs and a ground truth generated diffsr","brxName":"Plan Generation BRK","brxId":"27d49dfc-fa3f-43e3-b502-6c229a65c870","dependantBrxIds":{},"processType":7,"schemas":{"mainBrxId":"27d49dfc-fa3f-43e3-b502-6c229a65c870","schemas":{"_isMap":true,"data":[["main_brx_entry_schema",{"schemaFields":{"_isMap":true,"data":[["input_text",{"fieldValueDataType":"string","fieldValue":"testval"}],["search_replaces",{"fieldValueDataType":"string","fieldValue":"testval"}]]},"brxName":"Plan Generation BRK","brxId":"27d49dfc-fa3f-43e3-b502-6c229a65c870"}]]}}}

DELIM = "-----"

load_dotenv("../../.env")

dataset = load_dataset("princeton-nlp/SWE-bench_oracle", split="train")
dataset = dataset.to_iterable_dataset()

PATCH_FORMATTING_INST = """Please respond with a single patch file in the following format. <patch> --- a/file.py +++ b/file.py @@ -1,27 +1,35 @@ def euclidean(a, b): - while b: - a, b = b, a % b - return a + if b == 0: + return a + return euclidean(b, a % b) def bresenham(x0, y0, x1, y1): points = [] dx = abs(x1 - x0) dy = abs(y1 - y0) - sx = 1 if x0 < x1 else -1 - sy = 1 if y0 < y1 else -1 - err = dx - dy + x, y = x0, y0 + sx = -1 if x0 > x1 else 1 + sy = -1 if y0 > y1 else 1 - while True: - points.append((x0, y0)) - if x0 == x1 and y0 == y1: - break - e2 = 2 * err - if e2 > -dy: + if dx > dy: + err = dx / 2.0 + while x != x1: + points.append((x, y)) err -= dy - x0 += sx - if e2 < dx: - err += dx - y0 += sy + if err < 0: + y += sy + err += dx + x += sx + else: + err = dy / 2.0 + while y != y1: + points.append((x, y)) + err -= dx + if err < 0: + x += sx + err += dy + y += sy + points.append((x, y)) return points </patch>"""
SEARCH_REPLACE_INST = """
Please respond with a series of search-replace and new-file blocks. Here's an example:

First, in order to add an import for sympy at the top of the program, we must copy over the first couple lines for context.

<edit>
<filepath>mathweb/flask/app.py</filepath>
<search>
from flask import Flask

app = Flask(__name__)
</search>
<replace>
import sympy
from flask import Flask

app = Flask(__name__)
</replace>
</edit>

Second, let's remove the is_prime function completely.
<edit>
<filepath>mathweb/flask/app.py</filepath>
<search>
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
</search>
<replace>
</replace>
</edit>

Third, let's rewrite nth_prime to use sympy.isprime() instead of is_prime. We need to copy over is_prime completely to be able to properly make the replacement.
<edit>
<filepath>mathweb/flask/app.py</filepath>
<search>
    @app.route('/prime/<int:n>')
    def nth_prime(n):
        count = 0
        num = 1
        while count < n:
            num += 1
            if is_prime(num):
                count += 1
        return str(num)
</search>
<replace>
    @app.route('/prime/<int:n>')
    def nth_prime(n):
        count = 0
        num = 1
        while count < n:
            num += 1
            if sympy.isprime(num):
                count += 1
        return str(num)
</replace>
</edit>

Here's a quick example of generating a new file:
<newfile>
<filepath>mathweb/custom_fibonacci.py</filepath>
<content>
def nth_fibonacci(n):
    if n <= 1:
        return n
    else:
        fib = [0, 1]
        for i in range(2, n+1):
            fib.append(fib[i-1] + fib[i-2])
        return fib[n]
</content>
</newfile>
"""

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

current_row = next(iter(dataset))
def process_row():
    global current_row

    input_text = row["text"]
    input_text = input_text.replace(PATCH_FORMATTING_INST, SEARCH_REPLACE_INST)
    row_id = row["instance_id"]

    input_text += f"\n\n{DELIM}\n\n"

    diff_string_trimmed = row["patch"].replace("<patch>", "").replace("</patch>", "")
    # Maybe clean up some more strings at the top
    parsed_diff = parse_diff(diff_string_trimmed)
    fmtd_search_replaces = ""
    # Need to remove the a/ and the b/ at the start
    for block in parsed_diff:
        if "dev/null" in block.previous_filepath:
            fmtd_search_replaces += f"\n<newfile>\n<filepath>{block.filepath[2:]}</filepath>\n<content>{block.contents}</content>\n</newfile>"
        else:
            fmtd_search_replaces += f"\n<edit>\n<filepath>{block.filepath[2:]}</filepath>\n<search>{block.search_block}</search>\n<replace>{block.replace_block}</replace>\n</edit>\n"

    token_count = len(encoding.encode(input_text)) + len(encoding.encode(fmtd_search_replaces))

    if token_count > 14000:
        print(f"Token count is too large: {row_id}")
        current_row = next(iter(current_row))
    
    
for row in dataset:
    row_processing_worked = process_row(row)
    if row_processing_worked:
        print("Successfully generated a plan for a diff - skip!")
        break