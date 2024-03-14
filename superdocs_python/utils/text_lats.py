# Just using the implementation directly from Langchain

from __future__ import annotations
import json

import math
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict
from langchain.chains import create_structured_output_runnable
from langchain.output_parsers.openai_tools import (
    PydanticToolsParser,
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompt_values import ChatPromptValue

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda

from collections import defaultdict, deque

from langgraph.graph import END, StateGraph

from .prompts import EXECUTE_BLOCKS_PROMPT


def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 3:
        return END
    return "expand"

class FileOutputParser(BaseOutputParser):
    def __init__(self, files):
        self.files = files

    def parse(self, text: str) -> str:
        return ""

class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional[Node] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child(self):
        """Select the child with the highest UCT to search next."""
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent



class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str


class Reflection(BaseModel):
    reflections: str = Field(description="")
    normalized_score: float = Field(description="")
    found_solution: bool = Field(description="")

class PlanReflection(Reflection):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response. Provide feedback and next steps for improving upon the output."
    )
    simplicity: int = Field(
        description="Score from 0-10 on the simplicity of the candidate response to implement. Simpler solutions require reduce the number of new variables, files, and complex structures unless explicitly requested.",
        gte=0,
        lte=10,
    )
    effectiveness: int = Field(
        description="Score from 0-10 on the effectiveness and detail of the candidate response. Effective solutions are ones that, when implemented, are guaranteed to satisfy the user's goal. ",
        gte=0,
        lte=10
    )
    correctness: int = Field(
        description="Score from 0-10 on how correct the provided solution is in implementing the desired outcome. Correct solutions are ones that are prone to not introduce new errors.",
        gte=0,
        lte=10
    )

    found_solution: bool = Field(
        description="Does this response fully solve the question or task simply, effectively, and correctly?"
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\Simplicity: {self.simplicity}\Effectiveness: {self.effectiveness}\Correctness: {self.correctness}"
        )

    @property
    def normalized_score(self) -> float:
        return (self.simplicity + self.effectiveness + self.correctness) / 30.0


class LATS:
    def __init__(self, api_key, 
                 reflection=PlanReflection, 
                 reflection_name="PlanReflection", 
                 next_step_prompt="Improve upon the previous plan written given the current feedback and completely rewrite your plan. Do not truncate for brevity.", 
                 width=4, base_url="https://api.openai.com/v1/", 
            model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
        self.reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Reflect and grade the assistant response to the user question below.",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="candidate"),
            ]
        )

        self.width = width
        self.reflection = reflection

        self.reflection_llm_chain = (
            self.reflection_prompt
            | self.llm.bind_tools(tools=[self.reflection], tool_choice=reflection_name).with_config(
                run_name=reflection_name
            )
            | PydanticToolsParser(tools=[self.reflection])
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    next_step_prompt,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )


        self.initial_answer_chain = prompt_template | self.llm.with_config(
            run_name="GenerateInitialCandidate"
        )

        def generate_candidates(messages: ChatPromptValue):
            chat_result = self.llm.generate(
                [messages.to_messages()],
                n=self.width,
                run_name="GenerateCandidates",
            )
            return [gen.message for gen in chat_result.generations[0]]

        self.expansion_chain = prompt_template | generate_candidates

        builder = StateGraph(TreeState)
        builder.add_node("start", self.generate_initial_response)
        builder.add_node("expand", self.expand)
        builder.set_entry_point("start")


        builder.add_conditional_edges(
            "start",
            # Either expand/rollout or finish
            should_loop,
        )
        builder.add_conditional_edges(
            "expand",
            # Either continue to rollout or finish
            should_loop,
        )

        self.graph = builder.compile()

    def expand(self, state: TreeState) -> dict:
        """Starting from the "best" node in the tree, generate N candidates for the next step."""
        root = state["root"]
        best_candidate: Node = root.best_child if root.children else root
        messages = best_candidate.get_trajectory()

        # Generate N candidates from the single child candidate
        output_messages = self.expansion_chain.invoke(
            {"input": state["input"], "messages": messages}
        )
        output_messages = [[output_message] for output_message in output_messages]

        # Reflect on each candidate
        # For tasks with external validation, you'd add that here.
        reflections = self.get_reflection_chain().batch(
            [{"input": state["input"], "candidate": msges} for msges in output_messages],
        )
        # Grow tree
        child_nodes = [
            Node(cand, parent=best_candidate, reflection=reflection)
            for cand, reflection in zip(output_messages, reflections)
        ]

        if root.children:
            state["root"].best_child.children.extend(child_nodes)
        else:
            state["root"].children.extend(child_nodes)
        # We have already extended the tree directly, so we just return the state
        return state

    def generate_initial_response(self, state: TreeState) -> dict:
        """Generate the initial candidate response."""
        res = self.initial_answer_chain.invoke({"input": state["input"]})
        output_messages = [res]
        reflection = self.get_reflection_chain().invoke(
            {"input": state["input"], "candidate": output_messages}
        )
        root = Node(output_messages, reflection=reflection)
        return {
            **state,
            "root": root,
        }
    
    def get_reflection_chain(self):
        @as_runnable
        def reflection_chain(inputs) -> self.reflection:
            tool_choices = self.reflection_llm_chain.invoke(inputs)
            reflection = tool_choices[0]
            return reflection
        return reflection_chain

    def run(self, question):
        for step in self.graph.stream({"input": question}):
            step_name, step_state = next(iter(step.items()))
            print(step_name)
            print("rolled out: ", step_state["root"].height)
            print("---")
        solution_node = step["__end__"]["root"].get_best_solution()
        best_trajectory = solution_node.get_trajectory(include_reflections=False)
        return best_trajectory[-1].content


def main():
    from dotenv import load_dotenv
    import os
    load_dotenv("../.env")
    
    lats = LATS(os.environ["OPENAI_API_KEY"])
    print(lats.run("Write a plan to implements DFS over a tree."))

if __name__ == "__main__":
    main()