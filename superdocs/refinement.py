import os

def run_refinement_chain(directory, changes, objective, information, model, execution_function):
    # First, load each file involved in changes and apply the corresponding edits
    # Return those changes to the user or a boolean if the changes are finally done. 
    files = {}
    for change in changes:
        full_filepath = os.path.join(directory, change["filename"])
        contents = open(full_filepath, "r").read()
        files[change["filename"]] = contents

    for change in changes:
        files[change["filename"]].replace(change["old"], change["new"])
    

    new_changes = execution_function(model, directory, f"You have tried to make edits to the given files to solve {objective}. Now, make edits to ensure that your solution is accurate. If no changes are required, make no changes. You have the following context: {information}")
    
    if len(changes) == 0:
        return changes + new_changes
    else:
        return run_refinement_chain(directory, changes + new_changes, objective, information, execution_function)