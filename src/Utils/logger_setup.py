import logging
import time
import tracemalloc
import asyncio
from pathlib import Path
from typing import Callable, Any
from functools import wraps
from contextvars import ContextVar

# ──────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL SETUP
# ──────────────────────────────────────────────────────────────────────────────

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Start tracking memory allocations
tracemalloc.start()

# ContextVar ensures each async task or thread has its own logger instance
current_logger: ContextVar[logging.Logger] = ContextVar("current_logger")

# ──────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM HANDLERS & HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class MemoryHandler(logging.Handler):
    """Stores log records in a list for easy retrieval (e.g., returning in an API)."""
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        self.logs.append(self.format(record))

def setup_logger(trace_id: str) -> logging.Logger:
    """Creates a unique logger for a specific trace/request."""
    logger = logging.getLogger(f"app.{trace_id}")
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    formatter = logging.Formatter(
        f"[%(asctime)s] [TRACE:{trace_id}] [%(module)s.%(funcName)s:%(lineno)d] %(levelname)s - %(message)s"
    )

    # Handlers: File, Console (Stream), and Memory
    fh = logging.FileHandler(LOG_DIR / f"{trace_id}.log")
    ch = logging.StreamHandler()
    mh = MemoryHandler()

    for h in [fh, ch, mh]:
        h.setFormatter(formatter)
        logger.addHandler(h)

    # Attach MemoryHandler to the logger for easy access to logger.memory_handler.logs
    logger.memory_handler = mh 
    return logger

def get_log(name: str = "default") -> logging.Logger:
    """
    Helper to fetch a logger. 
    If a name is provided, it returns a logger with that name.
    Otherwise, it attempts to fetch the context-specific logger.
    """
    # 1. Try to get the logger from the ContextVar (for tracing)
    ctx_logger = current_logger.get(None)
    
    if ctx_logger:
        return ctx_logger
        
    # 2. If no context logger exists (like during startup/lifespan), 
    # return a named logger
    return logging.getLogger(name)

# ──────────────────────────────────────────────────────────────────────────────
# 3. PERFORMANCE TRACKER DECORATOR
# ──────────────────────────────────────────────────────────────────────────────

def track_performance(func: Callable):
    """
    Decorator that measures execution time and memory delta.
    Automatically detects if the decorated function is sync or async.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_log()
        start_time = time.perf_counter()
        start_mem, _ = tracemalloc.get_traced_memory()

        try:
            return await func(*args, **kwargs)
        finally:
            end_mem, peak_mem = tracemalloc.get_traced_memory()
            duration = time.perf_counter() - start_time
            logger.info(
                f"Executed async '{func.__name__}' in {duration:.4f}s | "
                f"Mem: {(end_mem - start_mem) / 1024:.2f}KB (Peak: {peak_mem / 1024:.2f}KB)"
            )

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_log()
        start_time = time.perf_counter()
        start_mem, _ = tracemalloc.get_traced_memory()

        try:
            return func(*args, **kwargs)
        finally:
            end_mem, peak_mem = tracemalloc.get_traced_memory()
            duration = time.perf_counter() - start_time
            logger.info(
                f"Executed sync '{func.__name__}' in {duration:.4f}s | "
                f"Mem: {(end_mem - start_mem) / 1024:.2f}KB (Peak: {peak_mem / 1024:.2f}KB)"
            )

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper