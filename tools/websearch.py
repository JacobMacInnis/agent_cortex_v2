from typing import Optional, Dict, Any, Type, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from duckduckgo_search import DDGS


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the internet.")


class WebSearchTool(BaseTool):
    name: str = "WebSearch"
    description: str = "Search the internet using DuckDuckGo. Useful for retrieving current or factual information."
    args_schema: Type[BaseModel] = WebSearchInput # type: ignore

    def _run(self, query: Optional[str] = None, **kwargs: Dict[str, Any]) -> str:
        q = query if query is not None else str(kwargs.get("query", ""))
        return self.search(q)


    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported.")

    def search(self, query: str, num_results: int = 3) -> str:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            formatted: List[str] = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "")
                snippet = result.get("body", "")
                url = result.get("href", "")
                formatted.append(f"[{i}] {title}\n{snippet}\n{url}\n")
            return "\n\n".join(formatted) if formatted else "No results found."







# from duckduckgo_search import DDGS
# from langchain.tools import BaseTool

# class WebSearchTool(BaseTool):
#     """
#     A tool for performing live internet search using DuckDuckGo.

#     Returns the top few search result snippets or URLs.
#     """

#     def search(self, query: str, num_results: int = 3) -> str:
#         with DDGS() as ddgs:
#             results = ddgs.text(query, max_results=num_results)
#             from typing import List
#             formatted: List[str] = []

#             for i, result in enumerate(results, 1):
#                 title = result.get("title", "")
#                 snippet = result.get("body", "")
#                 url = result.get("href", "")
#                 formatted.append(f"[{i}] {title}\n{snippet}\n{url}\n")

#             return "\n\n".join(formatted) if formatted else "No results found."