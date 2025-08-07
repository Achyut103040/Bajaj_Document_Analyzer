import os
import sys
import logging
from typing import Optional
import json

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the best available processor
processor = None
processor_type = "none"

def initialize_processor():
    """Initialize the best available processor"""
    global processor, processor_type
    
    # Try lightweight processor first (most reliable)
    try:
        from lightweight_main import LightweightDocumentProcessor
        processor = LightweightDocumentProcessor()
        
        # Load documents
        if os.path.exists("Datasets"):
            processor.load_documents("Datasets")
            processor_type = "lightweight"
            logger.info(f"✅ Lightweight processor ready: {len(processor.documents)} docs, {len(processor.clause_database)} clauses")
            return True
    except Exception as e:
        logger.error(f"❌ Lightweight processor failed: {e}")
    
    # Try enhanced processor as fallback
    try:
        from enhanced_main import OptimizedDocumentProcessor
        processor = OptimizedDocumentProcessor()
        
        if os.path.exists("Datasets"):
            processor.load_documents("Datasets")
            processor_type = "enhanced"
            logger.info(f"✅ Enhanced processor ready: {len(processor.documents)} docs, {len(processor.clause_database)} clauses")
            return True
    except Exception as e:
        logger.error(f"❌ Enhanced processor failed: {e}")
    
    return False

# Initialize FastAPI app
app = FastAPI(
    title="🤖 Bajaj Document Analyzer API",
    description="Insurance policy document analysis using LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    include_detailed_response: bool = True

class QueryResponse(BaseModel):
    query: str
    decision: str
    amount: float
    confidence: float
    justification: str
    detailed_response: Optional[str] = None
    clauses_mapping: list
    parsed_query: dict
    risk_factors: list
    recommendations: list
    processing_metadata: dict

@app.on_event("startup")
async def startup_event():
    """Initialize processor on startup"""
    logger.info("🚀 Starting Bajaj Document Analyzer...")
    if not initialize_processor():
        logger.error("❌ Failed to initialize any processor")
    else:
        logger.info(f"✅ System ready with {processor_type} processor")

@app.get("/")
async def root():
    """Root endpoint with system status"""
    if processor is None:
        return HTMLResponse("""
        <html>
            <head><title>Bajaj Document Analyzer</title></head>
            <body>
                <h1>🤖 Bajaj Document Analyzer</h1>
                <p style="color: red;">❌ System not initialized</p>
                <p>Please restart the server</p>
            </body>
        </html>
        """)
    
    return HTMLResponse(f"""
    <html>
        <head><title>Bajaj Document Analyzer</title></head>
        <body>
            <h1>🤖 Bajaj Document Analyzer</h1>
            <p style="color: green;">✅ System ready with {processor_type} processor</p>
            <p>Documents loaded: {len(processor.documents)}</p>
            <p>Clauses available: {len(processor.clause_database)}</p>
            <h2>Test Query</h2>
            <form action="/process_query" method="post">
                <textarea name="query" placeholder="Enter your insurance query..." rows="3" cols="50"></textarea><br>
                <button type="submit">Analyze</button>
            </form>
            <h2>API Endpoints</h2>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/stats">System Statistics</a></li>
            </ul>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    return {
        "status": "healthy",
        "processor_type": processor_type,
        "documents_loaded": len(processor.documents),
        "clauses_available": len(processor.clause_database),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    return {
        "processor_type": processor_type,
        "documents": len(processor.documents),
        "clauses": len(processor.clause_database),
        "processing_stats": getattr(processor, 'processing_stats', {}),
        "document_list": list(processor.documents.keys()) if processor.documents else []
    }

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language insurance queries"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
    
    try:
        logger.info(f"🔍 Processing query: {query[:100]}...")
        result = processor.process_query(query)
        
        # Ensure response format
        response_data = {
            "query": query,
            "decision": result.get("decision", "unknown"),
            "amount": float(result.get("amount", 0)),
            "confidence": float(result.get("confidence", 0)),
            "justification": result.get("justification", "Analysis completed"),
            "detailed_response": result.get("detailed_response") if request.include_detailed_response else None,
            "clauses_mapping": result.get("clauses_mapping", []),
            "parsed_query": result.get("parsed_query", {}),
            "risk_factors": result.get("risk_factors", []),
            "recommendations": result.get("recommendations", []),
            "processing_metadata": {
                **result.get("processing_metadata", {}),
                "processor_type": processor_type
            }
        }
        
        logger.info(f"✅ Query processed: {response_data['decision']} with {len(response_data['clauses_mapping'])} clauses")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Hackathon endpoint
@app.post("/hackrx/run")
async def hackathon_endpoint(request: dict):
    """Hackathon submission endpoint"""
    try:
        questions = request.get("questions", [])
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        answers = []
        for question in questions:
            result = processor.process_query(question)
            
            # Format answer for hackathon
            answer = f"Decision: {result.get('decision', 'unknown')}\n"
            answer += f"Confidence: {result.get('confidence', 0):.1%}\n"
            answer += f"Amount: ₹{result.get('amount', 0):,.0f}\n"
            answer += f"Justification: {result.get('justification', '')}\n"
            
            if result.get('clauses_mapping'):
                answer += f"Evidence: Based on {len(result['clauses_mapping'])} relevant policy clauses"
            
            answers.append(answer)
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"❌ Hackathon endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
