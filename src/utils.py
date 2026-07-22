import time
import functools
from src.logger import get_logger

logger = get_logger("ResiliencyUtil")

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Exponential backoff decorator for resilient API execution.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Max retries ({retries}) reached for {func.__name__}. Error: {e}")
                        raise e
                    sleep = (backoff_in_seconds * (2 ** x))
                    logger.warning(f"Retrying {func.__name__} in {sleep}s due to error: {e}")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator