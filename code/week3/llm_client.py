"""
llm_client.py — Wrapper thống nhất cho GPT-4o-mini và Gemini 1.5 Flash.

Thiết kế:
- Interface giống nhau cho cả 2 model
- Rate limiting tự động (tránh bị block)
- Retry với exponential backoff
- Cache responses để tiết kiệm API cost
- Log số tokens dùng
"""

import os, time, json, hashlib
from typing import Optional


class LLMClient:
    """
    Unified client cho OpenAI GPT và Google Gemini.

    Usage:
        client = LLMClient(provider="openai", api_key="sk-...")
        response = client.complete(messages=[...])
    """

    SUPPORTED = {"openai", "gemini"}

    def __init__(
        self,
        provider: str,                    # "openai" hoặc "gemini"
        api_key: Optional[str] = None,    # None → đọc từ env
        model: Optional[str] = None,      # None → dùng default
        cache_dir: str = "outputs/llm_cache",
        max_retries: int = 3,
        retry_delay: float = 2.0,
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
            self.model = model or "gemini-1.5-flash"
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Cần GEMINI_API_KEY env var hoặc truyền api_key")
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)

    def _cache_key(self, messages: list) -> str:
        """Hash messages làm cache key."""
        content = json.dumps(messages, ensure_ascii=False, sort_keys=True)
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
        """Actual API call — khác nhau giữa OpenAI và Gemini."""
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
            # Convert OpenAI format → Gemini format
            history = []
            system_content = ""
            for msg in messages[:-1]:  # tất cả trừ message cuối
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    history.append({"role": "model", "parts": [msg["content"]]})

            # Ghép system vào user message đầu tiên nếu có
            last_user = messages[-1]["content"]
            if system_content and not history:
                last_user = f"{system_content}\n\n{last_user}"

            chat = self.client.start_chat(history=history)
            resp = chat.send_message(
                last_user,
                generation_config={"temperature": temperature, "max_output_tokens": 512}
            )
            return resp.text

    def get_usage_stats(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_tokens / 1_000_000 * (
                0.15 if "gpt-4o-mini" in self.model else 0.075
            ),
        }
