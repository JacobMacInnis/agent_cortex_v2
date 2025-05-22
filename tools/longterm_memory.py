from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Type, Any, Dict
from longterm_memory import LongTermMemory

class LongTermMemoryInput(BaseModel):  # use LangChain’s v1 BaseModel if needed
    query: str = Field(description="Fact or question to retrieve from long-term memory")

class LongTermMemoryTool(BaseTool):
    name: str = "LongTermMemory"
    description: str = "Use this to retrieve known long-term facts from previous sessions or user history."
    args_schema: Type[BaseModel] = LongTermMemoryInput  # type: ignore

    _memory_store: LongTermMemory = PrivateAttr()

    def __init__(self, memory_store: LongTermMemory, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self._memory_store = memory_store

    def _run(self, query: Optional[str] = None, **kwargs: Any) -> str:
        query_str = query if query is not None else str(kwargs.get("query", ""))
        results = self._memory_store.query(query_str)
        return "\n".join(results) if results else "I couldn't find anything in long-term memory."

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported.")

