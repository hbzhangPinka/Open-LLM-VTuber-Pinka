import atexit
import requests
import json
from loguru import logger
from .openai_compatible_llm import AsyncLLM
from typing import AsyncIterator, List, Dict, Any


class OllamaLLM(AsyncLLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        keep_alive: float = -1,
        unload_at_exit: bool = True,
    ):
        self.keep_alive = keep_alive
        self.unload_at_exit = unload_at_exit
        self.cleaned = False
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
        )
        try:
            # preload model
            logger.info("Preloading model for Ollama")
            # Send the POST request to preload model
            logger.debug(
                requests.post(
                    base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": model,
                        "keep_alive": keep_alive,
                    },
                )
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to preload model: {e}")
            logger.critical(
                "Fail to connect to Ollama backend. Is Ollama server running? Try running `ollama list` to start the server and try again.\nThe AI will repeat 'Error connecting chat endpoint' until the server is running."
            )
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
        # If keep_alive is less than 0, register cleanup to unload the model
        if unload_at_exit:
            atexit.register(self.cleanup)

    def __del__(self):
        """Destructor to unload the model"""
        self.cleanup()

    def cleanup(self):
        """Clean up function to unload the model when exitting"""
        if not self.cleaned and self.unload_at_exit:
            logger.info(f"Ollama: Unloading model: {self.model}")
            # Unload the model
            # unloading is just the same as preload, but with keep alive set to 0
            logger.debug(
                requests.post(
                    self.base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": self.model,
                        "keep_alive": 0,
                    },
                )
            )
            self.cleaned = True

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str = None
    ) -> AsyncIterator[str]:
        """
        用 requests 直连 Ollama 的 /v1/chat/completions，避免 openai-python SDK 兼容性问题
        """
        url = self.base_url
        if not url.endswith("/chat/completions"):
            if url.endswith("/v1"):
                url = url + "/chat/completions"
            else:
                url = url.rstrip("/") + "/v1/chat/completions"
        
        logger.info(f"Ollama LLM: Requesting URL: {url}")
        logger.info(f"Ollama LLM: Model: {self.model}")
        logger.info(f"Ollama LLM: Messages: {messages}")
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if system:
            # Ollama 支持 system prompt，可以加到 messages 前面
            payload["messages"] = [{"role": "system", "content": system}] + messages

        # 支持流式
        payload["stream"] = True

        try:
            with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as resp:
                logger.info(f"Ollama LLM: Response status: {resp.status_code}")
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        text = line.decode("utf-8").strip()
                        logger.debug(f"Ollama LLM: Raw line: {text}")
                        if not text.startswith("data: "):
                            continue
                        content = text[len("data: "):]
                        if content == "[DONE]":
                            logger.info("Ollama LLM: Stream completed")
                            break
                        try:
                            data = json.loads(content)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    logger.debug(f"Ollama LLM: Yielding content: {content}")
                                    yield content
                        except json.JSONDecodeError as e:
                            logger.warning(f"Ollama LLM: JSON decode error: {e}, content: {content}")
                            # 跳过无效的 JSON 行
                            continue
        except Exception as e:
            logger.error(f"Ollama LLM: Error in chat_completion: {e}")
            raise
