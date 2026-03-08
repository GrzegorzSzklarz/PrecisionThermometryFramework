"""
logger_setup.py - Application Logging Configuration
"""

import logging
import sys
import os

def setup_logger(log_file_path: str):
    """
    Configures the root logger for the application.

    - All messages from INFO level and up are saved to a dedicated, clean log file.
    - Only messages from WARNING level and up are displayed in the console.
    - This setup is safe to call multiple times; it correctly reconfigures handlers.
    """
    # --- 1. Directory Integrity Check ---
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Access the root logger
    logger = logging.getLogger()
    
    # Set the base processing level to INFO
    logger.setLevel(logging.INFO)

    # --- 2. Handler Lifecycle Management ---
    # We iterate over a copy [:] of handlers to safely remove outdated instances.
    # This prevents the 'duplicate log entry' bug common in iterative workflows.
    console_handler = None
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            # Remove all old file handlers from previous runs.
            logger.removeHandler(handler)
        elif isinstance(handler, logging.StreamHandler):
            # Find an existing console handler (e.g., one added by Spyder).
            console_handler = handler

    # --- 3. Format Specification ---
    # Standardized timestamped format for professional traceability
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- 4. File Handler Configuration (Verbose) ---
    # 'mode=w' ensures each new dataset analysis starts with a clean log fil
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # --- 5. Console Handler Configuration (Silent/Minimalist) ---
    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)
        
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)