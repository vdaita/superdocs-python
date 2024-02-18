# Superdocs-Python

Currently, the main Superdocs repository has both the VSCode extension and the Python server. However, for ease of extensibility and testing, I wanted to break out the code-editing functionality into a Python library.

This project is largely inspired by Sweep.dev and Aider.

There are few main steps to creating diffs that Superdocs goes through:
1. Information retrieval
2. Planning
3. Writing code changes
4. Editing the code changes for accuracy.

Ideally, these should be as customizable as possible, with there being the ability to swap out different as