import os
import requests
import json
from typing import Optional, List
from google.adk.tools import BaseTool
from qa_system_with_a2a_demo.config import *

class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="WebSearchTool",
            description="Use this tool to search the web for current information or factual data."
        )

    def _search_serper(self, query: str) -> List[str]:
        api_key = SERPAPI_API_KEY
        if not api_key:
            return "Serper API key not found in environment variables."

        url = "https://google.serper.dev/search"
        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': api_key
        }
        payload = json.dumps({"q": query})

        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            results = response.json()
            organic = results.get("organic", [])[:10] # for top 10 result
            # return [result.get('snippet', '') for result in organic]
            return self._format_results(organic)
        except Exception as e:
            return f"Web search failed: {str(e)}"

    def _format_results(self, organic_results):
        if not organic_results:
            return "No results found."
        result_strings = []
        for result in organic_results[:5]:  # limit to top 5
            title = result.get('title', 'No Title')
            link = result.get('link', '#')
            snippet = result.get('snippet', 'No snippet available.')
            result_strings.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n---")
        return '\n'.join(result_strings)

    def run(self, input: str, state: Optional[dict] = None) -> str:
        return self._search_serper(input)
