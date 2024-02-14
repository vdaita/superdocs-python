import os

def run_refinement_chain(changes, directory, objective, information, execution_function):
    # First, load each file involved in changes and apply the corresponding edits
    # Return those changes to the user or a boolean if the changes are finally done. 
    files = {}
    for change in changes:
        full_filepath = os.path.join(directory, change["filename"])
        contents = open(full_filepath, "r").read()
        files[change["filename"]] = contents

    for change in changes:
        files[change["filename"]].replace(change["old"], change["new"])
    
    changes = execution_function(f"You have tried to make edits to the given files to solve {objective}. Now, make edits to ensure that your solution is accurate.", information)
    pass