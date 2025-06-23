import os
import grpc
import re
from typing import Dict, Any, Annotated
import logging
from qa_system_with_a2a_demo.tools.llm import load_llm
from langchain.tools import tool
import json
import asyncio
from qa_system_with_a2a_demo.config import *
import asyncio
import concurrent.futures
from qa_system_with_a2a_demo.agents.rag_agent import rag_agent_pb2
from qa_system_with_a2a_demo.agents.web_search_agent import web_search_agent_pb2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewerAgent:
    def __init__(self):
        self.llm = load_llm({
            'api_key': os.getenv('GROQ_API_KEY'),
            'model_name': 'mistral-saba-24b',
            'temperature': 0.3  # More deterministic evaluations
        })
        self.evaluation_prompt = """Evaluate the response quality based on:
1. RELEVANCE (0-1): Directly addresses the query without digressions
2. ACCURACY (0-1): Factually correct based on available knowledge
3. COMPLETENESS (0-1): Fully answers the query vs partial answer
4. SOURCE_QUALITY (0-1): Reliability of information sources

Query: {query}
Response: {response}
Source Type: {source_type}

Return JSON ONLY with scores and a boolean 'needs_web_search' flag if web search could provide better information:
{{
    "relevance": score,
    "accuracy": score,
    "completeness": score,
    "source_quality": score,
    "needs_web_search": boolean,
    "explanation": "brief rationale"
}}"""

    # @tool
    def compare_responses(self, query: str, responses: Dict[str, str]) -> Dict[str, Any]:
        """Compare multiple responses from different sources"""
        logger.info("Commencing the response comparison function")
        comparison_prompt = """Compare these responses for the query: '{query}'

        RAG Response (from documents):
        {rag_response}

        Web Search Response:
        {web_response}

        Return JSON with:
        - better_source: 'rag'|'web'|'both'
        - combined_suggestion: how to merge the best parts
        - rag_improvements: how RAG could be improved
        - confidence: 0-1 score on comparison reliability"""

        try:
            result = asyncio.run(self.llm.generate(
                comparison_prompt.format(
                    query=query,
                    rag_response=responses.get('rag', ''),
                    web_response=responses.get('web', '')
                ))
            )
            return json.loads(result["text"].strip())
        except Exception as e:
            logging.error(f"Comparison failed: {str(e)}")
            return {"error": str(e)}


class OrchestratorAgent():
    def __init__(self, rag_stub=None, web_stub=None):  # Add stub parameters
        super().__init__()
        self.reviewer = ReviewerAgent()
        self.rag_client = rag_stub  # Store RAG gRPC client
        self.web_client = web_stub  # Store WebSearch gRPC client
        self.channel = None
        self.recency_classifier_prompt = """You are a review agent that is an expert at reviewing text. You specialise at identifying only keywords that points to timeline.
                                            I have this multiagent system that has a raga agent and a websearch agent. you are at the first phase of my workflow. 
                                            Your task is to review the given query if it has any time keywords (such as today, recently, etc.), and then use your review to classify if this query requires recent/current information:
                        Query: {query}. =
                        

                        Your response should be a simple json. It should strictly be in the format below:
                        {{
                            'requires_recency': boolean,
                            'confidence': 0-1,
                            'keywords_found': [list of relevant terms]
                        }}
                        """
        
        self.sythesizer_prompt = """You are an expert at putting together a comprehensive report from given text. 
                                    You have been provided with the given text below
                                    text: {text}
                        

                                Your task is to synthensize a response to the query {query} using context from the provided text. You are free to cite the text if avaialable"""
    
        
        # Recent time indicators
        self.recency_indicators = [
            'current', 'latest', 'recent', 'today', 'now', 'new', 
            'updated', 'this week', 'this month', 'this year',
            '2024', '2023', 'as of', 'breaking'
        ]
       

    def is_rag_available(self):
        try:
            return self.rag_client # and self._check_service_health(self.rag_channel)
        except:
            return False

    def is_websearch_available(self):
        try:
            return self.web_client # and self._check_service_health(self.web_channel)
        except:
            return False
        
    @staticmethod
    def _check_service_health(channel):
        try:
            grpc.channel_ready_future(channel).result(timeout=1)
            return True
        except grpc.FutureTimeoutError:
            return False
        
        
    # @tool
    def route_query(self, query: str) -> Dict[str, Any]:
        """Intelligently route queries using proper dependencies"""
        try:
            # Use self.rag_client and self.web_client directly
            if self._requires_recency(query):
                logging.error(f"Query requires an recent (updated) response. Commencing websearch")
                web_response = self.call_websearch(query)
                ouput = self._synthensize_response(query, web_response)
                return {"source": "web_search", "response": ouput}
            
            # Use parallel processing with gRPC
            logger.info("Commencing parallel processing of query")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                logging.error(f"Making a call to the Rag Agent during parallel processing")
                rag_future = executor.submit(self.call_rag, query)
                # print(rag_future)
                logging.error(f"Making a call to the websearch Agent during parallel processing")
                web_future = executor.submit(self.call_websearch, query)
                # print(web_future)
                
                logging.error(f"Compiling results from parallel processing")
                rag_response = rag_future.result()
                # print(f"{rag_response}\n\n")
                web_response = web_future.result()
                # print(f"{web_response}\n\n")

            

            # Add proper error handling for responses
            if "error" in rag_response or "error" in web_response:
                raise Exception(f"RAG: {rag_response.get('error')}, Web: {web_response.get('error')}")

            # Compare responses
            logger.info("Commencing comparison between rag and web search")
            comparison = self.reviewer.compare_responses(
                query, {'rag': rag_response, 'web': web_response}
            )

            logger.info(f"Comparison completed. Decision is that {comparison.get('better_source')} response is the best")
            if comparison.get('better_source') == 'web':
                ouput = self._synthensize_response(query, web_response)

                return {
                    "source": "web_search_after_comparison",
                    "response": ouput,
                    "comparison": comparison
                }
            
            elif comparison.get('better_source') == 'both':
                response = {'rag': rag_response, 'web': web_response}
                ouput = self._synthensize_response(query, response)

                return {
                    "source": "web_search_after_comparison",
                    "response": ouput,
                    "comparison": comparison
                }
            else:
                ouput = self._synthensize_response(query, rag_response)
                return {
                    "source": "rag_after_comparison",
                    "response": ouput,
                    "comparison": comparison
                }

        except Exception as e:
            logging.error(f"Orchestration failed - Routing error: {str(e)}")
            return {"source": "error_fallback", "error": str(e)}
    def _synthensize_response(self, query, text):
        try:
            logger.info("Composing sythesizer_ prompt with query")
            prompt = self.sythesizer_prompt.format(query=query, text=text)
            logger.info("Commencing report composition")
            result = self.reviewer.llm.generate(prompt)
            logger.info("Output composition Concluded")

            return result
        except:
            return {
                "status": 404,
                "output": "Error synthensizing response from llm"
            }


    def _requires_recency(self, query: str) -> bool:
        """LLM-powered recency classification"""
        try:
            logger.info("Composing recency classifier prompt with query")
            prompt = self.recency_classifier_prompt.format(query=query)
            logger.info("Commencing LLM Review for recency")
            result = self.reviewer.llm.generate(prompt)
            logger.info("LLM Reviews concluded. Curating a json dump of response")
            print(result)
            try:
                classification = json.loads(result["text"].strip())
            except:
                json_match = re.search(r'\{.*\}', result["text"], re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).replace("True", "true").replace("False", "false")
                    classification = json.loads(json_str)
            

            return classification.get('requires_recency', False)
        except:
            return any(indicator in query["query"].lower() for indicator in self.recency_indicators)


    # def _call_agent(self, agent_client, request):
    #     """Generic method to call any agent"""
    #     try:
    #         response = agent_client.stub.HandleRequest(
    #             adk.agent_pb2.AgentRequest(request=json.dumps(request))
    #         )
    #         return json.loads(response.response)
    #     except Exception as e:
    #         logger.error(f"Error calling agent: {str(e)}")
    #         return None
    

    def call_rag(self, query: Dict) -> Dict[str, Any]:
        """Use gRPC instead of HTTP"""
        if not self.rag_client:
            logger.info("error: RAG client not initialized")
            return {"error": "RAG client not initialized"}
        
        try:
            # Convert query to protobuf format
            logger.info("Creating protobuf format of the request")
            request = rag_agent_pb2.ChunkRequest(query=query)
            # Call RAG service via gRPC
            response = self.rag_client.RetrieveChunks(request)
            return {
                "answer": response.chunks,
                "metadata": dict(response.metadata)
            }
        except Exception as e:
            logger.error(f"RAG call failed: {str(e)}")
            return {"error": str(e)}

    def call_websearch(self, query:Dict) -> Dict[str, Any]:
        """Use gRPC instead of HTTP"""
        if not self.web_client:
            logger.info("error: Web client not initialized")
            return {"error": "Web client not initialized"}

        
        try:
            # Convert query to protobuf format
            logger.info("Computing Web search Request")
            request = web_search_agent_pb2.SearchRequest(query=query)
            # Call WebSearch service via gRPC
            response = self.web_client.Search(request)
            return {
                "result": response.result,
                "metadata": dict(response.metadata)
            }
        except Exception as e:
            logger.error(f"WebSearch call failed: {str(e)}")
            return {"error": str(e)}
    

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main request handler for the orchestrator"""
        action = request.get('query')
        try:
            if self.is_rag_available():
                logger.info("RAG Agent is available")
            if self.is_websearch_available():
                logger.info("Web Search Agent is available")

        
            if action != '':
                query = request.get('query', '')
                print(query)
                try:
                    # loop = asyncio.get_event_loop()
                    logger.info("Commencing query routing")
                    return self.route_query(query)
                except Exception as e:
                    print(f"Error message from Handle Request function {e}")
                    return self.route_query(query)
            else:
                output =  {
                    "status": "error",
                    "message": f"Unknown action: {action}",
                    "valid_actions": ["query"]
                }
                return(output)
        except Exception as e:
            return {"Error": "RAG or Websearch Agent unavailable"}



# @tool
# def route_query(query: Annotated[str, "This contains the query sent by user"]):
#                     #   input: AnaDict[str, Any]) -> Dict[str, Any]:
#     """Intelligently route queries to appropriate agents"""
#     # query = input["query"]
#     print(f"query >> {query}")
#     Orchestrator = OrchestratorAgent()
#     try:
#         # Step 1: Check for recency requirements
#         print("Commencing Recency Check")
#         needs_recency = Orchestrator._requires_recency(query)
#         print("Recency Check Completed")
#         if needs_recency:
#             logger.info("Routing to WebSearch (recency requirement)")
#             print("Routing to WebSearch (recency requirement)")
#             web_response = Orchestrator.call_websearch(query)

#             return {
#                 "source": "web_search",
#                 "response": web_response,
#                 "confidence": 1.0
#             }
        

#         # Get both responses in parallel
#         print("Running Calls to Rag Agent and Web search Agent in Parallel")
#         # rag_response, web_response = asyncio.gather(
#         #     Orchestrator.call_rag(query),
#         #     Orchestrator.call_websearch(query)
#         # )
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future1 = executor.submit(Orchestrator.call_rag(query))
#             future2 = executor.submit(Orchestrator.call_websearch(query))
            
#             rag_response = future1.result()
#             web_response = future2.result()

#         # Compare responses
#         comparison = Orchestrator.reviewer.compare_responses(
#             query=query,
#             responses={'rag': rag_response.get('answer', ''), 'web': web_response}
#         )

#         if comparison.get('better_source') == 'web':
#             # Get summarized web response if better
#             web_response = Orchestrator.call_websearch(query)
#             return {
#                 "source": "web_search_after_comparison",
#                 "response": web_response,
#                 "comparison": comparison
#             }
        
#         return {
#             "source": "rag_after_comparison",
#             "response": rag_response,
#             "comparison": comparison
#         }

#     except Exception as e:
#         logging.error(f"Orchestration failed - Routing error: {str(e)}")
#         # Final fallback - direct web search
#         print(f"Orchestration failed - Routing error: {str(e)}")
#         web_response = Orchestrator._call_agent(
#             Orchestrator.web_client,
#             {"action": "search", "query": query, "summarize": True}
#         )
#         return {
#             "source": "web_search_error_fallback",
#             "response": web_response,
#             "confidence": 0.5
#         }