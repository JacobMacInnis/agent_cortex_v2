from duckduckgo_search import DDGS

class WebSearchTool:
    """
    A tool for performing live internet search using DuckDuckGo.

    Returns the top few search result snippets or URLs.
    """

    def search(self, query: str, num_results: int = 3) -> str:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            from typing import List
            formatted: List[str] = []

            for i, result in enumerate(results, 1):
                title = result.get("title", "")
                snippet = result.get("body", "")
                url = result.get("href", "")
                formatted.append(f"[{i}] {title}\n{snippet}\n{url}\n")

            return "\n\n".join(formatted) if formatted else "No results found."