from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from pydantic import PrivateAttr



class ReasoningInput(BaseModel):
    """Schema for the ReasoningTool input."""
    question: str = Field(description="The question to reason about")


class ReasoningTool(BaseTool):
    """A tool that tries to answer a question using only chat history, without calling external tools."""

    name: str = "Reasoning"
    description: str = (
        "Use this tool when the answer is likely already known from prior conversation "
        "or can be reasoned from memory."
    )
    args_schema: Type[BaseModel] = ReasoningInput
    _memory = PrivateAttr()

    def __init__(self, memory, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory

    def _run(self, question: Optional[str] = None, **kwargs) -> str:
        """Synchronous execution of the tool."""
        question = question or kwargs.get("question", "")

        for msg in reversed(self._memory.chat_memory.messages):
            if "my name is" in msg.content.lower():
                return f"You told me earlier: {msg.content}"

        return (
            f"I tried to reason about your question: '{question}', "
            "but I couldn't find anything in memory."
        )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")
