from langchain.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    Input should be a specific search query.
    Use this ONLY when the provided document context does not contain the answer.
    """
    try:
        search_tool = DuckDuckGoSearchResults(max_results=5)
        results = search_tool.run(query)
        return str(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"

# Export all tools
all_tools = [web_search_tool]