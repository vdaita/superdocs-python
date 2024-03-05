import os

def run_refinement_chain(directory, changes, objective, information, model, execution_function, iteration=3):
    # First, load each file involved in changes and apply the corresponding edits
    # Return those changes to the user or a boolean if the changes are finally done. 
    print("Running refinement chain on iteration: ", (3 - iteration))

    modified_files = []
    
    for change in changes:
        file_new_changes = execution_function(
            model, 
            directory, 
            f"""
            You were responsible accomplising the following objective: {objective}. \n
            -----
            You were given the following information and context: {information} \n
            -----
            Here is the original code for the files you chose to modify:
            {change["search"]}
            -----
            These are the new files, with your modifications included.
            {change["replace"]}
            ----
            Check the new file to ensure that all necessary changes are applied. Be meticulous and think step-by-step.

            Ensure that imports and dependencies are correctly implemented. 
            Ensure that tags are closed properly, etc.
            Make sure you add enough lines before and after your changes to provide sufficient context.
    
            If the new file includes all necessary changes, output "DONE" and DO NOT make any further diff edits. 
            Otherwise, make your edits!
            """,
            previous_modifications={changes}
        ) # Ensure that the changes are accurate (requires the contextual data) and the code changes are correct
        modified_files += file_new_changes

    original_mod_files_length = len(modified_files)

    for previous_change in changes:
        flag = False
        for modified_file in modified_files:
            if modified_file["filepath"] == previous_change["filepath"]:
                flag = True

        if not(flag):
            modified_files.append(previous_change)
        
    if original_mod_files_length == 0 or (iteration - 1) == 0:
        return 
    else:
        return run_refinement_chain(directory, changes + new_changes, objective, information, model, execution_function, iteration=iteration - 1)