#!/usr/bin/env python3
"""
Reliable startup script for Bajaj Document Analyzer
"""

import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_server():
    """Start the server with proper error handling"""
    try:
        logger.info("🚀 Starting Bajaj Document Analyzer Server...")
        
        # Ensure we're in the right directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if datasets exist
        if not os.path.exists("Datasets"):
            logger.error("❌ Datasets folder not found!")
            return False
        
        # Start server
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8001,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        return False

if __name__ == "__main__":
    start_server()
