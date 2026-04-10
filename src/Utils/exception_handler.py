import sys
import logging
from typing import Optional
from src.Utils.logger_setup import get_log

def error_message_detail(error: Exception) -> str:
    """
    Extracts the exact file name and line number where the error occurred.
    """
    _, _, exc_tb = sys.exc_info()
    
    # Fallback if no traceback context is available
    if exc_tb is None:
        return str(error)

    # Get the specific file and line number from the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error in [{file_name}] "
        f"at line [{line_number}] "
        f"→ Message: {str(error)}"
    )

class CustomException(Exception):
    """
    A smart exception wrapper that:
    1. Captures detailed file/line trace info.
    2. Logs the error automatically to the current context-aware logger.
    3. Prevents 'double-logging' if the exception is caught and re-raised.
    """
    def __init__(self, error: Exception, logger: Optional[logging.Logger] = None):
        # 1. Check if this exception has already been processed by CustomException
        if hasattr(error, 'is_custom'):
            self.detailed_error = str(error)
            self.is_custom = True
            # Skip initialization and logging; it was handled at the source
            return

        # 2. Extract detailed info for new/raw exceptions
        self.detailed_error = error_message_detail(error)
        self.is_custom = True
        
        # Initialize the base Exception class
        super().__init__(self.detailed_error)

        # 3. Log the error immediately to the trace-specific log
        # It uses the passed logger or fetches the one from the ContextVar
        log_to_use = logger or get_log()
        if log_to_use:
            log_to_use.error(self.detailed_error)

    def __str__(self) -> str:
        """Returns the formatted error string when the exception is printed."""
        return self.detailed_error