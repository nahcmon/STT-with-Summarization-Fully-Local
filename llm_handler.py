import aiohttp
from typing import List, Optional

class LLMHandler:
    def __init__(self):
        self.server_url: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=60)

    def _ensure_session(self):
        """Ensure session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self._timeout)

    async def connect(self, server_url: str):
        """Connect to LM Studio server"""
        # Ensure URL format
        if not server_url.startswith("http"):
            server_url = f"http://{server_url}"

        if server_url.endswith("/"):
            server_url = server_url[:-1]

        self.server_url = server_url

        # Test connection
        self._ensure_session()

        try:
            async with self.session.get(f"{self.server_url}/v1/models") as response:
                if response.status != 200:
                    raise Exception(f"Failed to connect to LM Studio: {response.status}")
        except Exception as e:
            raise Exception(f"Could not connect to LM Studio at {server_url}: {str(e)}")

    async def get_models(self) -> List[str]:
        """Get list of available models from LM Studio"""
        if not self.server_url:
            raise Exception("Not connected to LM Studio server")

        self._ensure_session()

        try:
            async with self.session.get(f"{self.server_url}/v1/models") as response:
                if response.status != 200:
                    raise Exception(f"Failed to get models: {response.status}")

                data = await response.json()
                models = [model["id"] for model in data.get("data", [])]
                return models
        except Exception as e:
            raise Exception(f"Error getting models: {str(e)}")

    async def process_text(self, text: str, model: str, system_prompt: str) -> str:
        """Send text to LLM for processing"""
        if not self.server_url:
            raise Exception("Not connected to LM Studio server")

        self._ensure_session()

        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }

            async with self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLM request failed: {response.status} - {error_text}")

                data = await response.json()
                result = data["choices"][0]["message"]["content"]
                return result
        except Exception as e:
            raise Exception(f"Error processing text with LLM: {str(e)}")

    async def process_text_streaming(self, text: str, model: str, system_prompt: str):
        """Send text to LLM for processing with streaming response"""
        if not self.server_url:
            raise Exception("Not connected to LM Studio server")

        self._ensure_session()

        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": True  # Enable streaming
            }

            async with self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLM request failed: {response.status} - {error_text}")

                # Stream the response line by line
                buffer = b''
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line_text = line.decode('utf-8').strip()

                        if line_text.startswith("data: "):
                            data_str = line_text[6:]  # Remove "data: " prefix
                            if data_str == "[DONE]":
                                break

                            try:
                                import json
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield {
                                            "type": "chunk",
                                            "content": content
                                        }
                            except json.JSONDecodeError:
                                pass
                            except Exception:
                                pass

        except Exception as e:
            raise Exception(f"Error processing text with LLM: {str(e)}")

    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
