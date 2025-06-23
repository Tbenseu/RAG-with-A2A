# import os
# import asyncio
# from typing import List, Dict, Any

# from google.adk.agents import Agent
# from fastapi import FastAPI, Request
# import uvicorn

# from qa_system_with_a2a_demo.tools.web_search_tool import WebSearchTool  # your BaseTool subclass
# from qa_system_with_a2a_demo.tools.llm import load_llm      # reuse your GroqLLM loader
# from qa_system_with_a2a_demo.config import *

# # 1) Instantiate tools
# search_tool = WebSearchTool()

# # 2) Instantiate the same Groq LLM
# rag_llm = load_llm({
#     "api_key": GROQ_API_KEY,
#     "model_name": "mixtral-8x7b-32768",
#     "max_tokens": 1000,
#     "temperature": 0.5
# })

# # 3) Define the “summarize” step that takes raw search text → answer
# async def summarize_search(query: str, search_results: str) -> Dict[str, Any]:
#     """
#     Given a user query and the raw WebSearchTool output,
#     craft a prompt that asks the LLM to produce a concise, factual answer.
#     """
#     prompt = f"""
# You are a web-savvy assistant.  A user asked:
#   "{query}"

# Below are the top search results (with titles, links & snippets).  Use only this information to answer the question as directly as possible:
# {search_results}

# Answer:
# """
#     completion = await rag_llm.generate(prompt)
#     return {
#         "answer": completion["text"].strip(),
#         "usage": completion.get("usage", {})
#     }

# # 4) Wire up the agent
# agent = Agent(
#     name="websearch_agent",
#     model=rag_llm.model_name,
#     instruction=(
#         "1) Call the `WebSearchTool` with the user’s query to gather fresh web snippets.  "
#         "2) Then call `summarize_search(query, search_results)` to produce a concise answer."
#     ),
#     description="An agent that searches the web via Serper and summarizes with Groq LLM",
#     tools=[search_tool, summarize_search],
#     # streaming=True,            # optional: stream token-by-token
# )

# app = FastAPI()

# @app.post("/run")
# async def handle_websearch_query(request: Request):
#     """
#     Accepts POST request with 'query', performs web search, summarizes results with LLM.
#     """
#     data = await request.json()
#     query = data.get("input", {}).get("query", "")
#     if not query:
#         return {"error": "Missing 'query' field in input"}

#     # Step 1: Perform web search
#     search_result = search_tool.run(query)
#     if isinstance(search_result, dict):
#         search_text = search_result.get("text", "")
#     else:
#         search_text = str(search_result)

#     # Step 2: Summarize the results
#     summary = await summarize_search(query, search_text)

#     return {
#         "answer": summary["answer"],
#         "usage": summary.get("usage", {}),
#         "search_snippets": search_text
#     }

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8002, log_level="info")


from concurrent import futures
import grpc
from qa_system_with_a2a_demo.agents.web_search_agent import web_search_agent_pb2
from qa_system_with_a2a_demo.agents.web_search_agent import web_search_agent_pb2_grpc
# from your_web_tool import WebSearchTool  # Your actual WebSearchTool
from qa_system_with_a2a_demo.tools.web_search_tool import WebSearchTool  # your BaseTool subclass
from qa_system_with_a2a_demo.tools.llm import load_llm      # reuse your GroqLLM loader
from qa_system_with_a2a_demo.config import *

import logging
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchServicer(web_search_agent_pb2_grpc.WebSearchServiceServicer):
    def __init__(self):
        self.tool = WebSearchTool()
    
    def Search(self, request, context={}):
        try:
            logger.info("Commencing Web Search...")
            result = self.tool.run(request.query)
            logger.info("Web Search Completed, returning response...")
            return web_search_agent_pb2.SearchResponse(
                result=result,
                metadata={"source": "serper", "results_count": str(len(result.split('---')))}
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return web_search_agent_pb2.SearchResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    web_search_agent_pb2_grpc.add_WebSearchServiceServicer_to_server(WebSearchServicer(), server)
    server.add_insecure_port('[::]:8002')
    server.start()
    logger.info("Starting server on port 8002...")
    # await server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)
        logger.info("Server shut down gracefully")

if __name__ == '__main__':
    serve()