from typing import List
from langchain.agents import AgentType, ConversationalAgent
from langchain.agents import initialize_agent # type: ignore
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI # type: ignore


from tools.fallback import FallbackTool
from tools.python_REPL import PythonREPLTool

""" Tool Imports"""
from tools.guarded_retriever import GuardedRetrieverTool
from tools.reasoning import ReasoningTool
from tools.retriever import RetrieverTool
from tools.websearch import WebSearchTool
from tools.calculator import CalculatorTool
from tools.longterm_memory import LongTermMemoryTool
from longterm_memory import LongTermMemory

# from transformers import pipeline
from langchain_ollama import OllamaLLM

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from dotenv import load_dotenv
load_dotenv()


def load_llm():
    return OllamaLLM(model="mistral", temperature=0.3)
    # return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

def get_tools(memory: ConversationBufferMemory) -> List[BaseTool]:
    retriever = RetrieverTool()
    guarded_retriever_tool = GuardedRetrieverTool(retriever=retriever)
    websearch_tool = WebSearchTool()
    calculator_tool = CalculatorTool()
    fallback_tool = FallbackTool()
    longterm_store = LongTermMemory()
    longterm_tool = LongTermMemoryTool(memory_store=longterm_store)
    reasoning_tool = ReasoningTool(name="Reasoning", memory=memory)
    python_repl_tool = PythonREPLTool()

    return [
        websearch_tool,
        python_repl_tool,
        calculator_tool,
        guarded_retriever_tool,
        longterm_tool,
        reasoning_tool,
        fallback_tool,
    ]


# Initialize the LLM and tools
def get_agent():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = get_tools(memory)
    llm = load_llm()

    format_instructions = (
        "To use a tool, use the following format:\n\n"
        "Thought: Do I need to use a tool? Yes\n"
        "Action: The action to take, must be one of [{tool_names}]\n"
        "Action Input: The input to the action\n\n"
        "When you have the final answer, use:\n\n"
        "Thought: Do I need to use a tool? No\n"
        "Final Answer: [your answer here]"
    )

    agent_prompt = ConversationalAgent.create_prompt(
        tools=tools,
        prefix=(
            "You are a helpful assistant who can use tools. "
            "Below is the conversation so far:\n\n"
            "{chat_history}\n\n"
            "Use this to understand what the user has previously told you."
        ),
        format_instructions=format_instructions,
        suffix="{input}",
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={"prompt": agent_prompt},
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )

    return agent, memory