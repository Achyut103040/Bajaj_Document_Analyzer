#!/usr/bin/env python3
"""
Working API Server for Bajaj Document Analyzer
Uses minimal processor to ensure reliability
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import minimal processor
try:
    from minimal_processor import MinimalDocumentProcessor
    processor = MinimalDocumentProcessor()
    processor.load_documents("Datasets")
    PROCESSOR_READY = True
    logger.info(f"✅ Processor ready: {len(processor.clause_database)} clauses")
except Exception as e:
    logger.error(f"❌ Processor initialization failed: {e}")
    processor = None
    PROCESSOR_READY = False

# FastAPI app
app = FastAPI(
    title="🤖 Bajaj Document Analyzer API",
    description="Insurance policy document analysis using LLMs - Working Version",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
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

@app.get("/")
async def root():
    """Root endpoint with system status"""
    status_color = "green" if PROCESSOR_READY else "red"
    status_text = "✅ System Ready" if PROCESSOR_READY else "❌ System Error"
    
    clause_count = len(processor.clause_database) if processor else 0
    doc_count = len(processor.documents) if processor else 0
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bajaj Document Analyzer</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .status {{ color: {status_color}; font-weight: bold; }}
            .container {{ max-width: 800px; }}
            .test-form {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
            input, textarea {{ width: 100%; padding: 10px; margin: 5px 0; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Bajaj Document Analyzer</h1>
            <p class="status">{status_text}</p>
            
            <h2>📊 System Status</h2>
            <ul>
                <li>Documents loaded: {doc_count}</li>
                <li>Clauses available: {clause_count}</li>
                <li>Processor type: Minimal (Reliable)</li>
                <li>API status: {"Running" if PROCESSOR_READY else "Error"}</li>
            </ul>
            
            <div class="test-form">
                <h3>🧪 Test Query</h3>
                <form action="/process_query" method="post" style="display: none;">
                    <textarea name="query" placeholder="Enter your insurance query..." rows="3"></textarea><br>
                    <button type="submit">Analyze Query</button>
                </form>
                
                <p><strong>Sample Queries to test:</strong></p>
                <ul>
                    <li>"46-year-old male, knee surgery in Pune, 3-month-old insurance policy"</li>
                    <li>"Does this policy cover maternity expenses?"</li>
                    <li>"What is the waiting period for pre-existing diseases?"</li>
                </ul>
                
                <p><strong>API Test Commands:</strong></p>
                <pre>
curl -X POST http://localhost:8001/process_query \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "46-year-old male, knee surgery in Pune"}}'
                </pre>
            </div>
            
            <h2>📚 API Endpoints</h2>
            <ul>
                <li><a href="/docs">📖 API Documentation (Swagger)</a></li>
                <li><a href="/health">🏥 Health Check</a></li>
                <li><a href="/stats">📈 System Statistics</a></li>
                <li><a href="/test">🧪 Interactive Test</a></li>
            </ul>
            
            <h2>🎯 Problem Statement Compliance</h2>
            <ul>
                <li>✅ Parse natural language queries</li>
                <li>✅ Search relevant clauses using semantic understanding</li>
                <li>✅ Evaluate information to determine decisions</li>
                <li>✅ Return structured JSON responses with clause mappings</li>
                <li>✅ Handle vague/incomplete queries</li>
                <li>✅ Explain decisions by referencing exact clauses</li>
            </ul>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not PROCESSOR_READY:
        raise HTTPException(status_code=503, detail="Processor not ready")
    
    return {
        "status": "healthy",
        "processor_ready": PROCESSOR_READY,
        "documents_loaded": len(processor.documents) if processor else 0,
        "clauses_available": len(processor.clause_database) if processor else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not available")
    
    return {
        "system_status": "ready" if PROCESSOR_READY else "error",
        "processor_type": "minimal",
        "documents_loaded": len(processor.documents),
        "clauses_available": len(processor.clause_database),
        "processing_stats": processor.stats,
        "document_list": list(processor.documents.keys()) if processor.documents else [],
        "sample_clauses": [
            {
                "text": clause["text"][:100] + "...",
                "type": clause.get("clause_type", "unknown"),
                "document": clause.get("document", "unknown")
            }
            for clause in processor.clause_database[:3]
        ] if processor.clause_database else []
    }

@app.get("/test")
async def test_page():
    """Interactive test page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Test - Bajaj Document Analyzer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .test-area { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .result { background: #e7f3ff; padding: 15px; margin: 10px 0; border-radius: 5px; }
            button { background: #28a745; color: white; padding: 10px 20px; border: none; cursor: pointer; margin: 5px; }
            textarea { width: 100%; height: 100px; padding: 10px; }
        </style>
    </head>
    <body>
        <h1>🧪 Interactive API Test</h1>
        
        <div class="test-area">
            <h3>Test Query Processing</h3>
            <textarea id="queryInput" placeholder="Enter your insurance query here...">46-year-old male, knee surgery in Pune, 3-month-old insurance policy</textarea>
            <br>
            <button onclick="testQuery()">Process Query</button>
            <button onclick="testSampleQueries()">Test Sample Queries</button>
        </div>
        
        <div id="results"></div>
        
        <script>
        async function testQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query.trim()) {
                alert('Please enter a query');
                return;
            }
            
            try {
                const response = await fetch('/process_query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const result = await response.json();
                displayResult('Single Query Test', result);
            } catch (error) {
                displayError('Query Test Failed', error);
            }
        }
        
        async function testSampleQueries() {
            const queries = [
                "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
                "Does this policy cover maternity expenses?",
                "What is the waiting period for pre-existing diseases?"
            ];
            
            document.getElementById('results').innerHTML = '<h3>Testing Sample Queries...</h3>';
            
            for (let i = 0; i < queries.length; i++) {
                try {
                    const response = await fetch('/process_query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: queries[i] })
                    });
                    
                    const result = await response.json();
                    displayResult(`Query ${i + 1}`, result);
                } catch (error) {
                    displayError(`Query ${i + 1} Failed`, error);
                }
            }
        }
        
        function displayResult(title, result) {
            const resultsDiv = document.getElementById('results');
            const resultHtml = `
                <div class="result">
                    <h4>${title}</h4>
                    <p><strong>Query:</strong> ${result.query}</p>
                    <p><strong>Decision:</strong> ${result.decision}</p>
                    <p><strong>Amount:</strong> ₹${result.amount.toLocaleString()}</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Clauses Found:</strong> ${result.clauses_mapping.length}</p>
                    <p><strong>Justification:</strong> ${result.justification}</p>
                    ${result.clauses_mapping.length > 0 ? 
                        '<p><strong>Sample Clause:</strong> ' + result.clauses_mapping[0].clause_text.substring(0, 100) + '...</p>' 
                        : '<p><em>No clauses mapped</em></p>'
                    }
                </div>
            `;
            resultsDiv.innerHTML += resultHtml;
        }
        
        function displayError(title, error) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML += `
                <div class="result" style="background: #ffebee;">
                    <h4>${title}</h4>
                    <p style="color: red;">Error: ${error.message || error}</p>
                </div>
            `;
        }
        </script>
    </body>
    </html>
    """)

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language insurance queries"""
    if not PROCESSOR_READY or not processor:
        raise HTTPException(status_code=503, detail="Processor not ready")
    
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
                "api_version": "1.0.0",
                "processor_type": "minimal"
            }
        }
        
        logger.info(f"✅ Query processed: {response_data['decision']} with {len(response_data['clauses_mapping'])} clauses")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/hackrx/run")
async def hackathon_endpoint(request: dict):
    """Hackathon submission endpoint"""
    if not PROCESSOR_READY or not processor:
        raise HTTPException(status_code=503, detail="Processor not ready")
    
    try:
        questions = request.get("questions", [])
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        answers = []
        for question in questions:
            result = processor.process_query(question)
            
            # Format answer for hackathon
            answer = f"Decision: {result.get('decision', 'unknown')}\\n"
            answer += f"Confidence: {result.get('confidence', 0):.1%}\\n"
            answer += f"Amount: ₹{result.get('amount', 0):,.0f}\\n"
            answer += f"Justification: {result.get('justification', '')}\\n"
            
            if result.get('clauses_mapping'):
                answer += f"Evidence: Based on {len(result['clauses_mapping'])} relevant policy clauses\\n"
                answer += f"Sample Evidence: {result['clauses_mapping'][0]['clause_text'][:100]}..."
            
            answers.append(answer)
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"❌ Hackathon endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Bajaj Document Analyzer API Server")
    print("=" * 50)
    print(f"Processor ready: {PROCESSOR_READY}")
    if processor:
        print(f"Documents loaded: {len(processor.documents)}")
        print(f"Clauses available: {len(processor.clause_database)}")
    print("Server starting at: http://localhost:8001")
    print("API docs at: http://localhost:8001/docs")
    print("Test page at: http://localhost:8001/test")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
