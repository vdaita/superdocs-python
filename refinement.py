import os

def run_refinement_chain(directory, changes, objective, information, model, execution_function, iteration=3):
    # First, load each file involved in changes and apply the corresponding edits
    # Return those changes to the user or a boolean if the changes are finally done. 
    print("Running refinement chain on iteration: ", (3 - iteration))
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

    new_changes = []
    
    for file in files:
        file_new_changes = execution_function(
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
            Check the new file to ensure that all necessary changes are applied. Be meticulous and think step-by-step.

            Ensure that imports and dependencies are correctly implemented. 
            If the new file includes all necessary changes, output "DONE" and DO NOT make any further diff edits. 
            Otherwise, make your edits!
            """
        ) # Ensure that the changes are accurate (requires the contextual data) and the code changes are correct
        new_changes += file_new_changes

    if len(new_changes) == 0 or (iteration - 1) == 0:
        print("Refinements completed.")
        return changes + new_changes
    else:
        return run_refinement_chain(directory, changes + new_changes, objective, information, model, execution_function, iteration=iteration - 1)