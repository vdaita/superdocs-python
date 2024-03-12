# Superdocs-Python

Currently, the main Superdocs repository has both the VSCode extension and the Python server. However, for ease of extensibility and testing, I wanted to break out the code-editing functionality into a Python library.

This project is largely inspired by Sweep.dev and Aider. 

There are few main steps to creating diffs that Superdocs goes through:
1. Information retrieval (both externally searching for content and internally searching within the codebase)
2. Planning (which is currently combined into the prompt of the code change-writing step)
3. Writing code changes
4. Editing the code changes for accuracy.

Ideally, these should be as customizable as possible, with there being the ability to swap out different output processing functions and models at will.

I have an implementation of Monte Carlo Tree Search that currently isn't being used. Instead, I'm using two-layer tree of thoughts to generate the output. 