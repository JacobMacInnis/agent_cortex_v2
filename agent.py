from langchain.agents import Tool, initialize_agent, AgentType, ConversationalAgent
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

""" Tool Imports"""
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

from langchain.tools import tool

@tool
def reason_from_memory(prompt: str) -> str:
    """Use this tool when you think the answer is already known or can be reasoned from prior conversation without using any other tools."""
    return f"(Reasoned directly without tools): {prompt}"



def load_llm():
    return OllamaLLM(model="mistral", temperature=0.3)
    # return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)


""" V1 we were using the Hugging Face LLM pipeline for text generation."""
# def load_llm():
#     """
#     Load the Hugging Face LLM pipeline for text generation.
#     """
#     pipe = pipeline(
#         "text2text-generation",
#         model="google/flan-t5-large",
#         device=-1,
#         max_new_tokens=256,
#         temperature=0.3
#     )
#     return HuggingFacePipeline(pipeline=pipe)

def get_tools(memory: ConversationBufferMemory) -> list[BaseTool]:
    retriever = RetrieverTool()
    websearch = WebSearchTool()
    calculator = CalculatorTool()
    fallback = FallbackTool()
    longterm_store = LongTermMemory()
    longterm_tool = LongTermMemoryTool(memory_store=longterm_store)
    reasoning_tool = ReasoningTool(name="Reasoning", memory=memory)


    def guarded_retriever(q: str) -> str:
        forbidden_keywords = ["weather", "forecast", "temperature", "time", "today", "tomorrow"]
        if any(k in q.lower() for k in forbidden_keywords):
            return "(Retriever skipped: question appears to require real-time data.)"
        return "\n".join(retriever.query(q))
    return [
        Tool(
            name="WebSearch",
            func=websearch.search,
            description="Useful when the question needs real-time or current information from the internet."
        ),
        Tool(
            name="Calculator",
            func=calculator.evaluate,
            description="Useful for answering math questions or evaluating expressions."
        ),
        Tool(
            name="Retriever",
            func=guarded_retriever,
            description="Use this tool to retrieve documents from the knowledge base. "
                        "Input should be a query string."
        ),
        Tool(
            name="LongTermMemory", 
            func=longterm_tool.run, 
            description=longterm_tool.description
        ),
        Tool(
            name="Reasoning",
            func=reason_from_memory,
            description="Use this when the answer is likely already known from prior conversation or does not require any tool."
        ),
        Tool(
            name="Fallback",
            func=fallback.run,
            description="Used when the input is ambiguous, self-referential, or not clearly directed at a specific task."
        ),
        reasoning_tool,
    ]

class FallbackTool:
    def run(self, query: str) -> str:
        return "I'm not sure how to help with that yet, but I'm still learning!"

    def __call__(self, query: str) -> str:
        return self.run(query)


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