import logging
import time
import tracemalloc
import asyncio
import sys
from pathlib import Path
from typing import Callable
from functools import wraps
from contextvars import ContextVar

# ─────────────────────────────────────────────────────────────
# GLOBAL SETUP
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

tracemalloc.start()

current_logger: ContextVar[logging.Logger] = ContextVar("current_logger", default=None)

# ─────────────────────────────────────────────────────────────
# MEMORY HANDLER (optional, silent)
# ─────────────────────────────────────────────────────────────

class MemoryHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        self.logs.append(self.format(record))

# ─────────────────────────────────────────────────────────────
# LOGGER SETUP (CLEAN)
# ─────────────────────────────────────────────────────────────

def setup_logger(trace_id: str) -> logging.Logger:
    logger = logging.getLogger(f"app.{trace_id}")

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)  # 🔥 cleaner than DEBUG
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )

    # Console handler (clean output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler (optional detailed logs)
    fh = logging.FileHandler(LOG_DIR / f"{trace_id}.log")
    fh.setLevel(logging.DEBUG)  # full logs go to file
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s"
    ))

    # Memory handler (optional)
    mh = MemoryHandler()
    mh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(mh)

    logger.memory_handler = mh
    return logger

# ─────────────────────────────────────────────────────────────
# LOGGER FETCHER
# ─────────────────────────────────────────────────────────────

# In src/Utils/logger_setup.py

def get_log(name: str = None) -> logging.Logger:
    logger = current_logger.get()
    if logger:
        return logger

    # If no context logger, return a named logger or default
    log_name = f"app.{name}" if name else "default"
    
    # Fallback minimal configuration
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s"
    )
    return logging.getLogger(log_name)

# ─────────────────────────────────────────────────────────────
# PERFORMANCE DECORATOR (MINIMAL)
# ─────────────────────────────────────────────────────────────

def track_performance(func: Callable):

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_log()
        start = time.perf_counter()

        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            return None  # 🔥 prevent noisy crash
        finally:
            duration = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_log()
        start = time.perf_counter()

        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            return None
        finally:
            duration = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper