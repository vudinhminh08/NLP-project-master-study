"""
llm_client.py — Wrapper thống nhất cho OpenAI, Gemini, và Ollama.

Thiết kế:
- Interface giống nhau cho các provider
- Rate limiting tự động (tránh bị block)
- Retry với exponential backoff
- Cache responses để tiết kiệm API cost
- Log số tokens dùng
"""

import os, time, json, hashlib
import urllib.request
import urllib.error
from typing import Optional


class LLMClient:
    """
    Unified client cho OpenAI GPT, Google Gemini, và Ollama.

    Usage:
        client = LLMClient(provider="openai", api_key="sk-...")
        response = client.complete(messages=[...])
    """

    SUPPORTED = {"openai", "gemini", "ollama"}

    def __init__(
        self,
        provider: str,                    # "openai" | "gemini" | "ollama"
        api_key: Optional[str] = None,    # None → đọc từ env
        model: Optional[str] = None,      # None → dùng default
        cache_dir: str = "outputs/llm_cache",
        max_retries: int = 5,
        retry_delay: float = 5.0,
    ):
        assert provider in self.SUPPORTED, f"Provider phải là {self.SUPPORTED}"
        self.provider = provider
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_tokens = 0
        os.makedirs(cache_dir, exist_ok=True)

        # Setup model name
        if provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Cần OPENAI_API_KEY env var hoặc truyền api_key")
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)

        elif provider == "gemini":
            self.model = model or "gemini-2.0-flash"
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Cần GEMINI_API_KEY env var hoặc truyền api_key")
            from google import genai
            self.client = genai.Client(api_key=self.api_key)

        elif provider == "ollama":
            self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
            # Optional for local Ollama; commonly required for cloud endpoints.
            self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
            self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def _cache_key(self, messages: list) -> str:
        """Hash messages làm cache key."""
        content = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "messages": messages,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[str]:
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)["response"]
        return None

    def _save_cache(self, key: str, response: str):
        path = os.path.join(self.cache_dir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"response": response}, f, ensure_ascii=False)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,   # 0 = deterministic
        use_cache: bool = True,
    ) -> str:
        """
        Gửi messages → nhận text response.

        Args:
            messages: list of {"role": ..., "content": ...}
            temperature: 0.0 cho reproducibility
            use_cache: True → cache responses (tiết kiệm API cost)

        Returns:
            str response text
        """
        # Check cache
        if use_cache:
            key = self._cache_key(messages)
            cached = self._load_cache(key)
            if cached:
                return cached

        # Call API với retry
        for attempt in range(self.max_retries):
            try:
                response = self._call_api(messages, temperature)

                if use_cache:
                    self._save_cache(key, response)
                return response

            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[ERROR] {self.provider} API failed after {self.max_retries} retries: {e}")
                    return "{}"
                wait = self.retry_delay * (2 ** attempt)
                print(f"[RETRY] Attempt {attempt+1} failed: {e}. Waiting {wait}s...")
                time.sleep(wait)

    def _call_api(self, messages: list, temperature: float) -> str:
        """Actual API call — khác nhau giữa OpenAI, Gemini và Ollama."""
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=512,
            )
            self.total_tokens += resp.usage.total_tokens
            return resp.choices[0].message.content

        elif self.provider == "gemini":
            # Convert OpenAI format → google.genai format
            from google.genai import types

            system_instruction = None
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=512,
                system_instruction=system_instruction,
                response_mime_type="application/json",  # force JSON output
            )
            resp = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return resp.text

        elif self.provider == "ollama":
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            }

            # 1) Native Ollama endpoint (local + some hosted setups)
            native_resp = self._post_json(
                url=f"{self.base_url}/api/chat",
                payload=payload,
                headers=headers,
                timeout=120,
                swallow_http_errors=True,
            )
            if native_resp is not None:
                prompt_tokens = int(native_resp.get("prompt_eval_count", 0) or 0)
                completion_tokens = int(native_resp.get("eval_count", 0) or 0)
                self.total_tokens += prompt_tokens + completion_tokens
                message = native_resp.get("message", {})
                return str(message.get("content", ""))

            # 2) OpenAI-compatible fallback (common for cloud gateways)
            compat_payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 512,
                "stream": False,
            }
            compat_resp = self._post_json(
                url=f"{self.base_url}/v1/chat/completions",
                payload=compat_payload,
                headers=headers,
                timeout=120,
                swallow_http_errors=False,
            )
            usage = compat_resp.get("usage", {}) if isinstance(compat_resp, dict) else {}
            self.total_tokens += int(usage.get("total_tokens", 0) or 0)

            choices = compat_resp.get("choices", []) if isinstance(compat_resp, dict) else []
            if not choices:
                return "{}"
            message = choices[0].get("message", {})
            return str(message.get("content", ""))

    def _post_json(
        self,
        url: str,
        payload: dict,
        headers: dict,
        timeout: int,
        swallow_http_errors: bool,
    ) -> Optional[dict]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError:
            if swallow_http_errors:
                return None
            raise
        except urllib.error.URLError:
            if swallow_http_errors:
                return None
            raise

    def get_usage_stats(self) -> dict:
        if self.provider == "ollama":
            est_cost = 0.0
        else:
            est_cost = self.total_tokens / 1_000_000 * (
                0.15 if "gpt-4o-mini" in self.model else
                0.10 if "gemini-2.0-flash" in self.model else 0.075
            )
        return {
            "provider": self.provider,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": est_cost,
        }
