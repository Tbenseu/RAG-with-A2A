import os
import groq
from typing import Dict, Optional, Union, List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from qa_system_with_a2a_demo.config import *

class GroqLLM:
    def __init__(self, config: Dict):
        self.api_key = config.get('api_key', GROQ_API_KEY)
        self.model_name = config.get('model_name', 'mistral-saba-24b')
        self.max_tokens = config.get('max_tokens', 4000)
        self.temperature = config.get('temperature', 0.7)
        self.client = groq.Client(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str, format=None, **kwargs) -> Dict:
        """Generate text completion with retry logic"""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=kwargs.get('model', self.model_name),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                stop=kwargs.get('stop', None),
                # response_format=format
            )
            
            return {
                "text": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            self.logger.error(f"Groq API call failed: {str(e)}")
            return {"error": str(e), "text": ""}
            
    def chat(self, messages: List[Dict], **kwargs) -> Dict:
        """Chat completion interface"""
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=kwargs.get('model', self.model_name),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "usage": dict(response.usage),
                "model": response.model
            }
            
        except Exception as e:
            self.logger.error(f"Groq chat failed: {str(e)}")
            return {"error": str(e), "content": ""}

def load_llm(config: Dict) -> GroqLLM:
    return GroqLLM(config)


# python rag_agent/main.py &
# python web_search_agent/main.py &
# python orchestrator/main.py