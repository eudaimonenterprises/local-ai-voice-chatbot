'''
LM Studio client module for OpenAI-compatible API integration.
'''


import requests
import json


class LMStudioClient:
    def __init__(self, api_url="http://localhost:1234/v1/chat/completions",
                 model_name="", timeout=60):
        self.api_url = api_url
        self.model_name = model_name  # If empty, use the loaded model in LM Studio
        self.timeout = timeout

    def generate_response(self, messages, temperature=0.7, top_p=0.9, max_tokens=150):
        """
        Generate a response from LM Studio's OpenAI-compatible API.
        
        Args:
            user_input (str): The user's input text
            prompt_template (str): Template for formatting the prompt
            temperature (float): Temperature parameter for generation
            top_p (float): Top-p sampling parameter
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated response from LM Studio
        """
        try:

            payload = {
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

            if self.model_name:
                payload["model"] = self.model_name

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract content from OpenAI-compatible response format
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"LM Studio API connection failed: {e}")
