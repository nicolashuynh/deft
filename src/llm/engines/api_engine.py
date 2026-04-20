from omegaconf import DictConfig

from src.llm.engines.unified_api_engine import (
    StandardRequestAdapter,
    UnifiedAPIEngine,
)


class APIEngine(UnifiedAPIEngine):
    """Default OpenAI/Azure chat-completions engine."""

    def __init__(
        self,
        content: str,
        temperature: float = 0.75,
        top_p: float = 0.95,
        max_tokens: int = 8000,
        max_attempts: int = 100,
        rate_limiter=None,
        request_timeout: int = 20,
        config: DictConfig = None,
        request_adapter=None,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            content=content,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            rate_limiter=rate_limiter,
            request_timeout=request_timeout,
            config=config,
            request_adapter=request_adapter or StandardRequestAdapter(),
        )
