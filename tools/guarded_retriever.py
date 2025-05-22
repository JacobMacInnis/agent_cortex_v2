from typing import Any, Dict, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from tools.retriever import RetrieverTool

class GuardedRetrieverInput(BaseModel):
    query: str = Field(description="The knowledge base query.")

class GuardedRetrieverTool(BaseTool):
    name: str = "Retriever"
    description: str = "Retrieves documents unless the query looks like it needs real-time data."
    args_schema: Type[BaseModel] = GuardedRetrieverInput # type: ignore

    def __init__(self, retriever: RetrieverTool, **kwargs: Any):
        super().__init__(**kwargs)
        self._retriever = retriever

    def _run(self, query: Optional[str] = None, **kwargs: Dict[str, Any]) -> str:
        q = str(query or kwargs.get("query", ""))
        if any(k in q.lower() for k in ["weather", "forecast", "temperature", "time", "today", "tomorrow"]):
            return "(Retriever skipped: question appears to require real-time data.)"
        return "\n".join(self._retriever.query(q))

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported.")
