#!/usr/bin/env python3
"""
Simple test server for Bajaj Document Analyzer
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Bajaj Document Analyzer - Test Server")

# Import processor
try:
    from lightweight_main import LightweightDocumentProcessor
    processor = LightweightDocumentProcessor()
    processor._create_sample_data()
    processor._build_knowledge_base()
    print(f"✅ Processor ready with {len(processor.clause_database)} clauses")
except Exception as e:
    print(f"❌ Processor failed: {e}")
    processor = None

@app.get("/")
def root():
    return {
        "message": "Bajaj Document Analyzer Test Server",
        "status": "running",
        "processor_available": processor is not None,
        "clauses": len(processor.clause_database) if processor else 0
    }

@app.post("/test_query")
def test_query(query_data: dict):
    if not processor:
        return {"error": "Processor not available"}
    
    query = query_data.get("query", "")
    if not query:
        return {"error": "No query provided"}
    
    try:
        result = processor.process_query(query)
        return {
            "query": query,
            "decision": result.get("decision"),
            "clauses": len(result.get("clauses_mapping", [])),
            "confidence": result.get("confidence"),
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "processor": "available" if processor else "unavailable",
        "clauses": len(processor.clause_database) if processor else 0
    }

if __name__ == "__main__":
    print("🚀 Starting test server on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
