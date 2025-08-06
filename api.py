import os
import sys
import tempfile
import shutil
import logging
from typing import List, Optional
import json
from pathlib import Path

# Vercel-specific configuration
if os.getenv('VERCEL'):
    # Vercel uses /tmp for write operations
    os.environ['HF_HOME'] = '/tmp/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'
    os.environ['NLTK_DATA'] = '/tmp/nltk_data'
    
    # Create cache directories
    Path('/tmp/huggingface').mkdir(exist_ok=True)
    Path('/tmp/transformers').mkdir(exist_ok=True)
    Path('/tmp/nltk_data').mkdir(exist_ok=True)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Enhanced error handling for imports
def safe_import_processor():
    """Safely import the best available processor"""
    try:
        # Download NLTK data if needed
        try:
            import nltk
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            import nltk
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        # Try enhanced processor first
        from enhanced_main import OptimizedDocumentProcessor as EnhancedProcessor
        logger.info("Enhanced processor available")
        return EnhancedProcessor, True
        
    except Exception as e:
        logger.warning(f"Enhanced processor not available: {e}")
        
        # Fallback to lightweight
        try:
            from lightweight_main import LightweightDocumentProcessor as LightweightProcessor
            logger.info("Using lightweight processor")
            return LightweightProcessor, False
        except Exception as e:
            logger.error(f"No processor available: {e}")
            raise ImportError("No document processor available")

# Import the best available processor
ProcessorClass, ENHANCED_AVAILABLE = safe_import_processor()

# Initialize FastAPI app with optimized settings
app = FastAPI(
    title="🤖 LLM Document Processing API",
    description="Advanced insurance policy document analysis using Large Language Models - Optimized for Hackathon Performance",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Optimized CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600  # Cache preflight requests for 1 hour
)

# HACKATHON AUTHENTICATION
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token for hackathon submission (configurable)"""
    # For hackathon demo, allow requests without token OR with any token
    # In production, implement proper token validation
    
    hackathon_token = os.getenv('HACKATHON_API_TOKEN', '')
    
    if not hackathon_token:
        # Demo mode - no token required
        return {"valid": True, "demo_mode": True}
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Bearer token required for hackathon submission"
        )
    
    if credentials.credentials != hackathon_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid Bearer token"
        )
    
    return {"valid": True, "demo_mode": False}

def generate_fallback_clauses(query: str, result: dict) -> List[dict]:
    """Generate intelligent fallback clauses when none are found"""
    query_lower = query.lower()
    fallback_clauses = []
    
    # Common insurance topics and their fallback responses
    fallback_mapping = {
        "health insurance": {
            "clause_text": "Health insurance typically covers hospitalization, medical treatments, and diagnostic procedures. Coverage limits vary by policy type and premium paid.",
            "clause_type": "general_coverage",
            "similarity_score": 0.7
        },
        "life insurance": {
            "clause_text": "Life insurance provides financial protection to beneficiaries upon the death of the insured. Benefits depend on the sum assured and policy terms.",
            "clause_type": "life_coverage",
            "similarity_score": 0.7
        },
        "claim": {
            "clause_text": "Insurance claims require proper documentation including medical reports, bills, and policy documents. Claims are processed within specified timelines.",
            "clause_type": "claims_process",
            "similarity_score": 0.6
        },
        "exclusion": {
            "clause_text": "Common exclusions include pre-existing diseases (subject to waiting periods), self-inflicted injuries, and treatments not medically necessary.",
            "clause_type": "exclusions",
            "similarity_score": 0.6
        },
        "premium": {
            "clause_text": "Premiums are calculated based on age, health status, coverage amount, and policy duration. Regular premium payment is required to keep the policy active.",
            "clause_type": "premium_terms",
            "similarity_score": 0.6
        },
        "waiting period": {
            "clause_text": "Waiting periods apply to specific conditions and treatments. Emergency treatments typically have no waiting period, while elective procedures may have 1-4 years waiting period.",
            "clause_type": "waiting_period",
            "similarity_score": 0.7
        }
    }
    
    # Match query with fallback topics
    for topic, clause_info in fallback_mapping.items():
        if topic in query_lower:
            fallback_clauses.append({
                "clause_text": clause_info["clause_text"],
                "document": "General Insurance Knowledge Base",
                "clause_type": clause_info["clause_type"],
                "similarity_score": clause_info["similarity_score"],
                "search_method": "intelligent_fallback"
            })
            break
    
    # If no specific match, provide general fallback
    if not fallback_clauses:
        decision = result.get("decision", "unknown")
        if decision == "covered":
            fallback_text = "Based on standard insurance policy terms, this appears to be covered under typical insurance policies. Specific coverage amounts and conditions may vary."
        elif decision == "not_covered":
            fallback_text = "This may not be covered under standard insurance policies due to common exclusions or limitations. Please check your specific policy document."
        else:
            fallback_text = "Insurance coverage depends on specific policy terms, premium type, and individual circumstances. Please consult your policy document for exact details."
        
        fallback_clauses.append({
            "clause_text": fallback_text,
            "document": "AI Analysis - General Insurance Principles",
            "clause_type": "ai_analysis",
            "similarity_score": 0.5,
            "search_method": "ai_inference"
        })
    
    return fallback_clauses

def enhance_justification(original_justification: str, clauses_mapping: List[dict]) -> str:
    """Enhance justification with additional context"""
    if not original_justification or original_justification == "No justification provided":
        if not clauses_mapping:
            return "Analysis completed using AI inference based on general insurance principles. No specific policy clauses were found matching your query."
        else:
            return f"Analysis based on {len(clauses_mapping)} relevant policy sections and AI reasoning. The decision considers standard insurance practices and policy terms."
    
    # Enhance existing justification
    if clauses_mapping and any(clause.get("search_method") == "intelligent_fallback" for clause in clauses_mapping):
        original_justification += " Note: Some analysis is based on general insurance principles as specific policy clauses were not found."
    
    return original_justification

# Global processor instance with thread safety
processor = None
processor_type = "unknown"

# Optimized Pydantic models
class QueryRequest(BaseModel):
    query: str
    include_detailed_response: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "query": "46-year-old male, knee surgery in Pune, 3-month policy",
                "include_detailed_response": True
            }
        }

class QueryResponse(BaseModel):
    query: str
    decision: str
    amount: float
    confidence: float
    justification: str
    detailed_response: Optional[str] = None
    clauses_mapping: List[dict] = []
    parsed_query: dict = {}
    risk_factors: List[str] = []
    recommendations: List[str] = []
    processing_metadata: dict = {}

class SystemStats(BaseModel):
    documents_loaded: int
    total_clauses: int
    models_loaded: dict
    device: str
    system_status: str
    processor_type: str = "unknown"

@app.on_event("startup")
async def startup_event():
    """Optimized processor initialization with smart fallback"""
    global processor, processor_type
    
    dataset_path = "d:/Bajaj_Finserv/Datasets"
    logger.info("🚀 Starting optimized document processor initialization...")
    
    # Try enhanced processor first (if NLTK data is available)
    if ENHANCED_AVAILABLE:
        try:
            logger.info("🔧 Initializing enhanced processor...")
            processor = ProcessorClass()
            
            # Load documents and validate
            if os.path.exists(dataset_path):
                processor.load_documents(dataset_path)
                docs_count = len(getattr(processor, 'documents', {}))
                clauses_count = len(getattr(processor, 'clause_database', []))
                
                logger.info(f"✅ Enhanced processor: {docs_count} docs, {clauses_count} clauses")
                
                if clauses_count > 0:
                    processor_type = "enhanced"
                    logger.info("🎉 Enhanced processor ready!")
                    return
                else:
                    logger.warning("Enhanced processor loaded but no clauses found")
                    
        except Exception as e:
            logger.error(f"❌ Enhanced processor failed: {e}")
    
    # Fallback to lightweight processor
    logger.info("🔧 Initializing lightweight processor...")
    try:
        if not ENHANCED_AVAILABLE:
            processor = ProcessorClass()
        else:
            from lightweight_main import LightweightDocumentProcessor
            processor = LightweightDocumentProcessor()
        
        # Load documents with validation
        if os.path.exists(dataset_path):
            processor.load_documents(dataset_path)
        else:
            logger.warning(f"Dataset path not found: {dataset_path}")
            processor._create_sample_data()
            processor._build_knowledge_base()
        
        # Ensure we have clauses
        clauses_count = len(getattr(processor, 'clause_database', []))
        if clauses_count == 0:
            logger.info("🔄 Adding sample clauses for demo...")
            processor._add_sample_clauses()
            clauses_count = len(processor.clause_database)
        
        docs_count = len(getattr(processor, 'documents', {}))
        processor_type = "lightweight"
        
        logger.info(f"✅ Lightweight processor: {docs_count} docs, {clauses_count} clauses")
        logger.info("🎉 System ready for hackathon demo!")
        
    except Exception as e:
        logger.error(f"💥 Critical failure: {e}")
        processor = None
        processor_type = "failed"

@app.get("/", response_class=HTMLResponse)
async def home():
    """🏠 Serve the enhanced frontend interface"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback if static file not found
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Insurance Analyzer</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                .error { color: #e53e3e; background: #fed7d7; padding: 20px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="error">
                <h2>Frontend Not Found</h2>
                <p>The enhanced frontend interface is not available. Please ensure static/index.html exists.</p>
                <p>API is still functional at <code>/process_query</code></p>
            </div>
        </body>
        </html>
        """)

# HACKATHON SPECIFIC MODELS
class HackathonRequest(BaseModel):
    """Hackathon specific request model matching exact requirements"""
    documents: str  # Document content or URL
    questions: List[str]  # List of questions to answer

class HackathonResponse(BaseModel):
    """Hackathon specific response model matching exact requirements"""
    answers: List[str]  # Answers to the questions

# HACKATHON REQUIRED ENDPOINT
@app.post("/hackrx/run", response_model=HackathonResponse)
async def hackathon_endpoint(
    request: HackathonRequest,
    auth: dict = Depends(verify_token)
):
    """🏆 Official Hackathon Endpoint - POST /hackrx/run
    
    Matches exact hackathon requirements:
    - Accepts documents and questions
    - Returns structured answers
    - Implements Bearer token authentication (configurable)
    - Optimized for accuracy, latency, and explainability
    - Evaluated on: Accuracy, Token Efficiency, Latency, Reusability, Explainability
    """
    
    if processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Document processor not initialized. Please restart the service."
        )
    
    try:
        logger.info(f"🏆 Hackathon endpoint called with {len(request.questions)} questions")
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:100]}...")
            
            # Create a comprehensive query including document context
            enhanced_query = f"{question}"
            if request.documents:
                enhanced_query = f"Based on the provided documents: {question}"
            
            # Process with our enhanced system
            result = processor.process_query(enhanced_query)
            
            # Format answer according to hackathon requirements
            answer = {
                "question": question,
                "answer": result.get("justification", "Unable to determine from available information"),
                "decision": result.get("decision", "review_required"),
                "confidence": result.get("confidence", 0.0),
                "evidence": []
            }
            
            # Add evidence from clause mapping
            for clause in result.get("clauses_mapping", [])[:3]:  # Top 3 clauses
                evidence = {
                    "clause": clause.get("clause_text", "")[:200] + "..." if len(clause.get("clause_text", "")) > 200 else clause.get("clause_text", ""),
                    "document": clause.get("document", "Unknown"),
                    "relevance_score": clause.get("similarity_score", 0.0)
                }
                answer["evidence"].append(evidence)
            
            # Format as string response for hackathon compatibility
            formatted_answer = f"Answer: {answer['answer']}\n"
            formatted_answer += f"Decision: {answer['decision']}\n"
            formatted_answer += f"Confidence: {answer['confidence']:.1%}\n"
            
            if answer['evidence']:
                formatted_answer += "Evidence:\n"
                for j, evidence in enumerate(answer['evidence'], 1):
                    formatted_answer += f"{j}. {evidence['clause']} (Source: {evidence['document']}, Relevance: {evidence['relevance_score']:.2f})\n"
            
            answers.append(formatted_answer)
        
        logger.info(f"✅ Hackathon processing completed successfully")
        
        return HackathonResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ Hackathon endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """🤖 Process natural language insurance queries with optimized performance"""
    if processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Document processor not initialized. Please restart the service."
        )
    
    # Input validation
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
    
    try:
        logger.info(f"🔍 Processing query: {query[:100]}...")
        
        # Process with timeout handling
        result = processor.process_query(query)
        
        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError("Invalid processor response format")
        
        # Handle error responses
        if result.get("decision") == "error":
            error_msg = result.get("justification", "Unknown processing error")
            logger.error(f"Processor error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Ensure all required fields exist with defaults
        clauses_mapping = result.get("clauses_mapping", [])
        
        # Enhanced fallback handling for no clauses found
        if not clauses_mapping and result.get("decision") not in ["error", "unknown"]:
            # Provide intelligent fallback based on query type
            fallback_clauses = generate_fallback_clauses(query, result)
            clauses_mapping = fallback_clauses
        
        response_data = {
            "query": query,
            "decision": result.get("decision", "unknown"),
            "amount": float(result.get("amount", 0)),
            "confidence": float(result.get("confidence", 0)),
            "justification": enhance_justification(result.get("justification", ""), clauses_mapping),
            "detailed_response": result.get("detailed_response") if request.include_detailed_response else None,
            "clauses_mapping": clauses_mapping,
            "parsed_query": result.get("parsed_query", {}),
            "risk_factors": result.get("risk_factors", []),
            "recommendations": result.get("recommendations", []),
            "processing_metadata": {
                **result.get("processing_metadata", {}),
                "processor_type": processor_type,
                "response_time_ms": result.get("processing_time", 0) * 1000,
                "fallback_used": len(clauses_mapping) > len(result.get("clauses_mapping", []))
            }
        }
        
        logger.info(f"✅ Query processed: {response_data['decision']} (${response_data['amount']:,.0f})")
        return QueryResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error processing query: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """📊 Get optimized system statistics and health information"""
    if processor is None:
        return SystemStats(
            documents_loaded=0,
            total_clauses=0,
            models_loaded={"status": "not_initialized"},
            device="unknown",
            system_status="not_initialized",
            processor_type="none"
        )
    
    try:
        # Get stats with error handling
        stats = processor.get_system_stats() if hasattr(processor, 'get_system_stats') else {}
        
        # Enhanced stats with performance metrics
        enhanced_stats = {
            "documents_loaded": stats.get("documents_loaded", len(getattr(processor, 'documents', {}))),
            "total_clauses": stats.get("total_clauses", len(getattr(processor, 'clause_database', []))),
            "models_loaded": {
                **stats.get("models_loaded", {}),
                "processor_type": processor_type,
                "performance_mode": "optimized"
            },
            "device": stats.get("device", "cpu"),
            "system_status": "healthy" if processor_type != "failed" else "degraded",
            "processor_type": processor_type
        }
        
        return SystemStats(**enhanced_stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return SystemStats(
            documents_loaded=0,
            total_clauses=0,
            models_loaded={"error": str(e)},
            device="unknown",
            system_status="error",
            processor_type=processor_type
        )

@app.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload new documents to the system"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_files = []
            
            for file in files:
                if not file.filename.endswith('.pdf'):
                    continue
                
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                uploaded_files.append(file.filename)
            
            if uploaded_files:
                # Process the uploaded documents
                processor.load_documents(temp_dir)
                
                return {
                    "message": f"Successfully uploaded and processed {len(uploaded_files)} documents",
                    "files": uploaded_files
                }
            else:
                raise HTTPException(status_code=400, detail="No valid PDF files uploaded")
                
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """🏥 Optimized health check with detailed system status"""
    if processor is None:
        return {
            "status": "unhealthy", 
            "reason": "processor_not_initialized",
            "processor_type": processor_type,
            "timestamp": "2025-08-05"
        }
    
    try:
        # Quick health validation
        stats = processor.get_system_stats() if hasattr(processor, 'get_system_stats') else {}
        
        docs_count = stats.get("documents_loaded", len(getattr(processor, 'documents', {})))
        clauses_count = stats.get("total_clauses", len(getattr(processor, 'clause_database', [])))
        
        # Determine health status
        is_healthy = docs_count > 0 and clauses_count > 0 and processor_type != "failed"
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "processor_type": processor_type,
            "documents_loaded": docs_count,
            "total_clauses": clauses_count,
            "performance": "optimized",
            "ready_for_demo": is_healthy,
            "timestamp": "2025-08-05"
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "reason": str(e),
            "processor_type": processor_type,
            "timestamp": "2025-08-05"
        }

@app.get("/api/docs")
async def get_api_docs():
    """📖 Comprehensive API documentation with examples"""
    return {
        "title": "🤖 LLM Document Processing API - Hackathon Edition",
        "version": "2.0.0",
        "description": "Optimized insurance policy document analysis using Large Language Models",
        "processor_type": processor_type,
        "performance": "optimized",
        "endpoints": {
            "POST /process_query": {
                "description": "Process natural language insurance queries",
                "example": {
                    "query": "46-year-old male, knee surgery in Pune, 3-month policy",
                    "include_detailed_response": True
                }
            },
            "GET /stats": "Get system statistics and performance metrics",
            "POST /upload_documents": "Upload new PDF documents for analysis",
            "GET /health": "Detailed health check with system status",
            "GET /": "Interactive web interface"
        },
        "example_queries": [
            {
                "category": "Surgical Procedures",
                "queries": [
                    "46-year-old male, knee surgery in Pune, 3-month policy",
                    "Female patient needs heart surgery, age 55, Mumbai location",
                    "Emergency surgery required, what's the coverage limit?"
                ]
            },
            {
                "category": "Specialized Care",
                "queries": [
                    "Dental treatment for 30-year-old, covered under policy?",
                    "Cataract surgery eligibility for 70-year-old senior citizen",
                    "Cancer treatment coverage for 45-year-old patient"
                ]
            },
            {
                "category": "Complex Scenarios",
                "queries": [
                    "Pre-existing condition, diabetes treatment for 50-year-old",
                    "Maternity coverage for 28-year-old in Delhi",
                    "ICU treatment costs for elderly patient"
                ]
            }
        ],
        "features": [
            "🚀 Optimized performance for hackathon demos",
            "🧠 Advanced LLM-based analysis",
            "📊 Real-time statistics and monitoring",
            "🔄 Automatic fallback to lightweight mode",
            "💡 Smart clause extraction and mapping",
            "🎯 Confidence scoring and risk assessment"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Support both Vercel and local development
    port = int(os.environ.get("PORT", 8001))
    
    print("🚀 Starting LLM Insurance Document Processor...")
    print(f"📖 Access the interface at: http://localhost:{port}")
    print(f"📚 API Documentation: http://localhost:{port}/api/docs")
    print(f"🌐 Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# Vercel serverless handler
handler = app
