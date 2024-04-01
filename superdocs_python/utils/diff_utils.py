from dataclasses import dataclass

@dataclass
class Hunk:
    filepath: str
    text: str
    previous_filepath: str = ""

@dataclass
class SearchReplaceChange:
    filepath: str
    search_block: str
    replace_block: str
    previous_filepath: str = ""

def find_hunks(diff_string):
    hunks = []
    previous_filename = ""
    current_filename = ""
    current_lines = ""
    for line in diff_string.splitlines():
        if line.startswith("---"):
            previous_filename = line[3:]
        elif line.lstrip().startswith("+++"):
            if len(current_filename) > 0:
                hunks.append(Hunk(current_filename, current_lines, previous_filepath=previous_filename))
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

        if len(search_block.strip()) > 0 and len(replace_block.strip()) > 0:
            search_replace_blocks.append(
                SearchReplaceChange(filepath, search_block, replace_block)
            )
        filepath = ""
        search_block = ""
        replace_block = ""

    search_replace_blocks.append(
        SearchReplaceChange(filepath, search_block, replace_block)
    )

    return search_replace_blocks
