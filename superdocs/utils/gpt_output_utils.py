import json
import re

def extract_code_block_data(md_text, language):
   # Regular expression pattern for matching diff code blocks
   pattern = rf'```{language}([\s\S]*?)```'
   # Find all diff code blocks using the pattern
   diff_code_blocks = re.findall(pattern, md_text, re.MULTILINE)
   return diff_code_blocks

def extract_xml_tags(text, tag):
    pattern = r'<' + tag + '>(.*?)</' + tag + '>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches
