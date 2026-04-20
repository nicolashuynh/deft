import re
import time
from abc import abstractmethod

from omegaconf import DictConfig


class GenericEngine:

    def __init__(
        self,
        content: str,
        temperature: float = 0.75,
        top_p: float = 0.95,
        max_tokens: int = 100,
        max_attempts: int = 100,
        rate_limiter=None,
        print_parsing_errors: bool = True,
        MAX_RETRIES: int = 100, # This was 4 before
        config: DictConfig = None,
    ) -> None:

        """Initialize the instance."""
        self.content = content
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.rate_limiter = rate_limiter

        self.print_parsing_errors = print_parsing_errors
        self.MAX_RETRIES = MAX_RETRIES
        self.retry_backoff = [10, 30, 60, 120, 150, 180, 210, 240, 270, 300] * 10
        assert len(self.retry_backoff) >= self.MAX_RETRIES, "The retry backoff list must be at least as long as the number of retries."
        self.config = config

        # Track token rate with a timer
        self.a = time.time()
        self.total_token = 0

    @abstractmethod
    async def _async_generate(self, user_message, n_generations_per_prompt):
        """Handle async generate."""
        pass

    def _extract_retry_time(self, exception: str, attempt_num: int) -> int:
        """Calculate exact retry time from openai.error.RateLimitError exception message."""
        match = re.search(r"retry after (\d+) seconds", exception)
        if match:
            return int(match.group(1)) + 1
        else:
            return self.retry_backoff[attempt_num]
