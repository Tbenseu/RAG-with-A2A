# import os
# import asyncio
# from typing import List, Dict

# from google.adk.agents import Agent             # ← only this import is needed
# from fastapi import FastAPI, Request
# import uvicorn

# from qa_system_with_a2a_demo.tools.document_tool import DocumentTool
# from qa_system_with_a2a_demo.tools.llm import load_llm  # your modules
# from qa_system_with_a2a_demo.config import *

# # 1) Instantiate your tools
# # doc_tool = DocumentTool(model_name="all-mpnet-base-v2")
# doc_tool = DocumentTool(model_name="BAAI/bge-small-en-v1.5")
# rag_llm = load_llm({
#     "api_key": GROQ_API_KEY,
#     "model_name": "mixtral-8x7b-32768",
#     "max_tokens": 4000,
#     "temperature": 0.7
# })

# # 2) Define your two tool functions
# def retrieve_chunks(query: str, top_k: int = 10) -> Dict:
#     """
#     Retrieve the top-k relevant chunks for the user’s query.
#     """
#     chunks = doc_tool.retrieve_relevant_chunks(query, top_k=top_k)
#     return {"chunks": chunks}

# async def generate_answer(query: str, chunks: List[str]) -> Dict:
#     """
#     Given the original query and retrieved chunks, form a prompt
#     and ask the Groq LLM for an answer.
#     """
#     prompt = (
#         f"You are a helpful assistant. Answer the user’s question:\n"
#         f"  \"{query}\"\n\n"
#         f"Use only the following document excerpts:\n"
#         + "\n---\n".join(chunks)
#     )
#     completion = await rag_llm.generate(prompt)
#     return {
#         "answer": completion["text"],
#         "usage": completion.get("usage", {})
#     }

# # 3) Build your agent, passing the functions directly
# rag_agent = Agent(
#     name="rag_agent",
#     model=rag_llm.model_name,
#     instruction=(
#         "First call `retrieve_chunks(query, top_k)` to grab relevant "
#         "text chunks, then call `generate_answer(query, chunks)` to "
#         "compose the final response."
#     ),
#     description="A retrieval-augmented generation agent (RAG) using FAISS + Groq",
#     tools=[retrieve_chunks, generate_answer],     # ← plain functions
#     # streaming=True,
# )

# app = FastAPI()

# @app.post("/run")
# async def handle_rag_query(request: Request):
#     """
#     Accepts POST input with 'query', optionally 'top_k',
#     and returns an answer from RAG pipeline.
#     """
#     data = await request.json()
#     query = data.get("input", {}).get("query", "")
#     top_k = data.get("input", {}).get("top_k", 10)

#     if not query:
#         return {"error": "Missing 'query' field in input"}

#     chunks_resp = retrieve_chunks(query, top_k)
#     chunks = chunks_resp["chunks"]
#     answer_resp = await generate_answer(query, chunks)
#     return {
#         "answer": answer_resp["answer"],
#         "usage": answer_resp.get("usage", {}),
#         "chunks_used": len(chunks)
#     }

# # 5) Uvicorn entry point
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")

# #     rag_agent.run(api_server=True, port=8001)


from concurrent import futures
import grpc
from qa_system_with_a2a_demo.agents.rag_agent import rag_agent_pb2
from qa_system_with_a2a_demo.agents.rag_agent import rag_agent_pb2_grpc
import io
# from your_document_tool import DocumentTool  # Your actual DocumentTool class

from qa_system_with_a2a_demo.tools.document_tool import DocumentTool
from qa_system_with_a2a_demo.tools.llm import load_llm  # your modules
from qa_system_with_a2a_demo.config import *

import logging
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagAgentServicer(rag_agent_pb2_grpc.RagAgentServiceServicer):
    def __init__(self):
        self.tool = DocumentTool()
        
    def ProcessDocument(self, request, context):
        try:
            file_stream = io.BytesIO(request.content)
            self.tool.handle_uploaded_file(file_stream, request.filename)
            return rag_agent_pb2.ChunkResponse(
                chunks=["Processing completed"],
                metadata={"status": "success"}
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return rag_agent_pb2.ChunkResponse()

    def RetrieveChunks(self, request, context):
        try:
            logger.info("Commencing Data Retrieval..")
            chunks = self.tool.retrieve_relevant_chunks(request.query)
            logger.info("Retrieval completed")
            return rag_agent_pb2.ChunkResponse(
                chunks=chunks,
                metadata={"source": "chromadb", "model": self.tool.model_name}
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return rag_agent_pb2.ChunkResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rag_agent_pb2_grpc.add_RagAgentServiceServicer_to_server(RagAgentServicer(), server)
    server.add_insecure_port('[::]:8001')

    logger.info("Starting server on port 8001...")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)
        logger.info("Server shut down gracefully")

if __name__ == '__main__':
    serve()

