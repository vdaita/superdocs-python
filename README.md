# Superdocs-Python

Currently, the main Superdocs repository has both the VSCode extension and the Python server. However, for ease of extensibility and testing, I wanted to break out the code-editing functionality into a Python library.
This is very much experimental.

This project is largely inspired by Sweep.dev and Aider. 

There are few main steps to creating diffs that Superdocs goes through:
1. Information retrieval (both externally searching for content and internally searching within the codebase)
2. Planning (which is currently combined into the prompt of the code change-writing step)
3. Writing code changes
4. Editing the code changes for accuracy.

Ideally, these should be as customizable as possible, with there being the ability to swap out different output processing functions and models at will.

I have an implementation of Monte Carlo Tree Search that currently isn't being used. Instead, I'm using two-layer tree of thoughts to generate the output. 

You can install superdocs by running ```pip install git+https://github.com/vdaita/superdocs-python.git```.
Start running superdocs by going into your current directory and running ```superdocs```. It requires that you have an environment variable set called ```OPENAI_API_KEY``` already set to the OpenAI API key that you plan on using (```export OPENAI_API_KEY="your api key"```).
Once you run it, you can select 'add' and drag and drop filepath(s) from your editor to be processed.
After that, you can run and state your objective. This takes quite a while to run, especially with some of the larger models!

Running Superdocs can take a lot of credits and is pretty computationally expensive. I'm currently trying out alternative methods of making the edits in the mini folder.
Please be as specific and detailed as you can be with your objective.

Next steps:
1. Parallelizing "execution" calls to improve speed (done!)
2. Separating plan and execute steps back out again (implemented in mini.py)
3. Allowing for separate models to be used for summarization and code implementation (done!)
4. Improving generation quality on smaller LLMs.

If you have any suggestions for going about these, please open a GH Issue or email me at vdaita@gmail.com

Thank you!