from langchain.agents import Tool, initialize_agent, AgentType
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool

from tools.retriever import RetrieverTool
from tools.websearch import WebSearchTool
from tools.calculator import CalculatorTool

# from transformers import pipeline
from langchain_ollama import OllamaLLM

def load_llm():
    return OllamaLLM(model="mistral", temperature=0.3)


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

def get_tools() -> list[BaseTool]:
    retriever = RetrieverTool()
    websearch = WebSearchTool()
    calculator = CalculatorTool()
    fallback = FallbackTool()

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
            func=lambda q: "\n".join(retriever.query(q)),  # type: ignore
            description="Use this tool to retrieve documents from the knowledge base. "
                        "Input should be a query string."
        ),
        Tool(
            name="Fallback",
            func=fallback.run,
            description="Used when the input is ambiguous, self-referential, or not clearly directed at a specific task."
        )
    ]

class FallbackTool:
    def run(self, query: str) -> str:
        return "I'm not sure how to help with that yet, but I'm still learning!"

    def __call__(self, query: str) -> str:
        return self.run(query)


# Initialize the LLM and tools
def get_agent():
    tools = get_tools()
    llm = load_llm()

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
        # return_intermediate_steps=True
    )