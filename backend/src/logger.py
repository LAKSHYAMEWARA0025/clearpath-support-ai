import logging
import json
import sys

class ExactJSONFormatter(logging.Formatter):
    """
    Custom formatter to ensure the output strictly matches the assignment rubric:
    { query, classification, model_used, tokens_input, tokens_output, latency_ms }
    """
    def format(self, record):
        if isinstance(record.msg, dict):
            # If we pass a dictionary, dump it directly to JSON
            return json.dumps(record.msg)
        # Fallback for standard text logs
        return json.dumps({"level": record.levelname, "message": record.getMessage()})

def setup_router_logger():
    logger = logging.getLogger("router_logger")
    logger.setLevel(logging.INFO)
    
    # Prevent adding multiple handlers if called more than once
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ExactJSONFormatter())
        logger.addHandler(console_handler)
        
    # Prevent logs from propagating to the root logger and printing twice
    logger.propagate = False 
    
    return logger

# Initialize our router logger
router_log = setup_router_logger()

# Standard logger for general app debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app_log = logging.getLogger("app_logger")