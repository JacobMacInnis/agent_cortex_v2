from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type, Any

class FallbackInput(BaseModel):
    query: str = Field(description="Fallback query")

class FallbackTool(BaseTool):
    name: str = "Fallback"
    description: str = "Used when the input is ambiguous or not clearly directed at a specific task."
    args_schema: Type[BaseModel] = FallbackInput # type: ignore

    def _run(self, query: Optional[str] = None, **kwargs: dict[str, Any]) -> str:
        return "I'm not sure how to help with that yet, but I'm still learning."

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported.")
