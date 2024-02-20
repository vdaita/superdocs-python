import os

def run_refinement_chain(directory, changes, objective, information, model, execution_function):
    # First, load each file involved in changes and apply the corresponding edits
    # Return those changes to the user or a boolean if the changes are finally done. 
    files = {}
    for change in changes:
        full_filepath = os.path.join(directory, change["filename"])
        contents = open(full_filepath, "r").read()
        files[change["filename"]] = {
            "original": contents,
            "new": contents
        }

    for change in changes:
        files[change["filename"]]["new"].replace(change["original"], change["new"])
    
    for file in files:
        new_changes = execution_function(
            model, 
            directory, 
            f"""
            You were responsible accomplising the following objective: {objective}. \n
            -----
            You were given the following information and context: {information} \n
            -----
            Here is the original code for the files you chose to modify:
            {files[file]["original"]}
            -----
            These are the new files, with your modifications included.
            {files[file]["new"]}
            ----
            Check to make sure that all of the code changes you made are accurate. Be meticulous and think step-by-step.
            Ensure that imports and dependencies are correctly implemented. 
            If you don't require any further changes (i.e. this looks good) print done.
            Otherwise, make your edits!
            """
        ) # Ensure that the changes are accurate (requires the contextual data) and the code changes are correct
        
    if len(changes) == 0:
        return changes + new_changes
    else:
        return run_refinement_chain(directory, changes + new_changes, objective, information, model, execution_function)