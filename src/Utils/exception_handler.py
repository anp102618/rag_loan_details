import logging
import traceback
from typing import Optional
from src.Utils.logger_setup import get_log


def error_message_detail(error: Exception) -> str:
    tb = traceback.extract_tb(error.__traceback__)

    if not tb:
        return str(error)

    last_trace = tb[-1]
    file_name = last_trace.filename
    line_number = last_trace.lineno

    return (
        f"Error in [{file_name}] "
        f"at line [{line_number}] "
        f"→ Message: {str(error)}"
    )


class CustomException(Exception):
    def __init__(self, error: Exception, logger: Optional[logging.Logger] = None):
        
        if hasattr(error, 'is_custom'):
            self.detailed_error = str(error)
            self.is_custom = True
            super().__init__(self.detailed_error)
            return

        self.detailed_error = error_message_detail(error)
        self.is_custom = True

        super().__init__(self.detailed_error)

        log_to_use = logger or get_log()

        if log_to_use and hasattr(log_to_use, "error"):
            log_to_use.error(self.detailed_error)

    def __str__(self) -> str:
        return self.detailed_error