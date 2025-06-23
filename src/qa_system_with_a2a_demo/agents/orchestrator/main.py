# def serve():
#     """Create and start the orchestrator server"""
#     orchestrator = OrchestratorAgent()
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
#     # Start on port 8000 (default orchestrator port)
#     port = '8000'
#     server.add_insecure_port(f'[::]:{port}')
#     server.start()
#     print(f"Orchestrator Agent server started on port {port}")
    
#     try:
#         while True:
#             # Keep the server running
#             pass
#     except KeyboardInterrupt:
#         server.stop(0)
    

# # # FastAPI app to expose the orchestrator
# app = FastAPI()

# @app.post("/run")
# async def handle_query(request: Request):
#     """
#     Accepts POST requests with input.query, routes it using the orchestrator logic.
#     """
#     data = await request.json()
#     query = data.get("input", {}).get("query", "")
#     if not query:
#         return {"error": "Missing query"}
#     serve()
#     return await OrchestratorAgent().handle_request(query)

# # Run the app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)


from concurrent import futures
import grpc
from grpc import aio 
import json
import asyncio
from google.protobuf.struct_pb2 import Struct
from qa_system_with_a2a_demo.agents.orchestrator import orchestrator_agent_pb2
from qa_system_with_a2a_demo.agents.orchestrator import orchestrator_agent_pb2_grpc
from qa_system_with_a2a_demo.agents.rag_agent import rag_agent_pb2_grpc
from qa_system_with_a2a_demo.agents.web_search_agent import web_search_agent_pb2_grpc
from qa_system_with_a2a_demo.agents.orchestrator.agent import OrchestratorAgent 

import logging
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestratorServicer(orchestrator_agent_pb2_grpc.OrchestratorServiceServicer):
    def __init__(self):
        # Initialize gRPC channels
        self.rag_channel = grpc.insecure_channel('localhost:8001')
        self.web_channel = grpc.insecure_channel('localhost:8002')

        # Create gRPC stubs
        self.rag_stub = rag_agent_pb2_grpc.RagAgentServiceStub(self.rag_channel)
        self.web_stub = web_search_agent_pb2_grpc.WebSearchServiceStub(self.web_channel)
        
        # Pass stubs to the agent during initialization
        self.agent = OrchestratorAgent(
            rag_stub=self.rag_stub,
            web_stub=self.web_stub
        )
        

    async def ProcessQuery(self, request, context):
        try:
            # Convert gRPC request to dict
            query_data = {
                "query": request.query,
                "context": dict(request.context)

            }

            # print(query_data)
            
            # Use async execution
            result = self.agent.handle_request(query_data)

            # print(result.get("response").get("text"))
            
            # Convert response to protobuf Struct
            response_struct = Struct()
            response_struct.update(result)

            logger.info("Process completed successfully")
            
            return orchestrator_agent_pb2.OrchestrationResponse(
                response=response_struct,
                source=result.get("source", "unknown"),
                confidence=result.get("confidence", 0.5)
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            # print("Error")
            return orchestrator_agent_pb2.OrchestrationResponse()


async def serve():
    server = aio.server()  # Use async server
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orchestrator_agent_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorServicer(), server)
    server.add_insecure_port('[::]:8000')
    

    logger.info("Starting server on port 8000...")
    # await server.start()
    await server.start()
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop(0)
        logger.info("Server shut down gracefully")

if __name__ == '__main__':
    # serve()
    asyncio.run(serve())  # Run with asyncio



# python rag_agent/main.py &
# python web_search_agent/main.py &
# python orchestrator/main.py