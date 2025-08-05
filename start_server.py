#!/usr/bin/env python3
"""Fixed API Server with comprehensive error handling"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from api import app
    print("✅ API module loaded successfully")
    
    if __name__ == "__main__":
        print("🚀 Starting optimized API server...")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8001, 
            log_level="info",
            reload=False
        )
        
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
