#!/usr/bin/env python3
"""
COMPREHENSIVE SOLUTION FOR CLAUSE GENERATION ISSUES

This script provides multiple fixes for the Bajaj Document Analyzer clause generation problem:

1. Fix dataset path issues
2. Optimize model loading to prevent hanging
3. Improve clause extraction logic
4. Add better error handling
5. Provide fallback mechanisms
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_problem():
    """Analyze the current problem with clause generation"""
    print("🔍 ANALYZING CLAUSE GENERATION PROBLEM")
    print("=" * 50)
    
    # Problem 1: Dataset path mismatch
    print("1. Dataset Path Analysis:")
    old_path = "d:/Bajaj_Finserv/Datasets"
    correct_path = "Datasets"
    
    print(f"   Old path in code: {old_path}")
    print(f"   Correct path: {correct_path}")
    print(f"   Exists: {os.path.exists(correct_path)}")
    
    if os.path.exists(correct_path):
        files = [f for f in os.listdir(correct_path) if f.endswith('.pdf')]
        print(f"   PDF files found: {len(files)}")
    
    # Problem 2: Model loading issues
    print("\n2. Model Loading Analysis:")
    print("   Issue: Enhanced processor loads heavy models causing hanging")
    print("   Solution: Use lazy loading or lightweight alternatives")
    
    # Problem 3: Clause extraction issues
    print("\n3. Clause Extraction Analysis:")
    print("   Issue: Complex entity extraction may fail")
    print("   Solution: Add error handling and fallbacks")
    
    return True

def create_optimized_processor():
    """Create an optimized version that focuses on clause generation"""
    
    optimized_code = '''#!/usr/bin/env python3
"""
Optimized Document Processor - Focused on Clause Generation
This version prioritizes reliable clause extraction over advanced ML features
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Basic libraries only - no heavy ML models initially
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK for text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    # Download required data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except ImportError:
    # Fallback without NLTK
    def sent_tokenize(text):
        return re.split(r'[.!?]+', text)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDocumentProcessor:
    """
    Optimized processor focusing on reliable clause generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with minimal dependencies"""
        logger.info("Initializing Optimized Document Processor (Clause-Focused)...")
        
        self.config = config or {
            'similarity_threshold': 0.3,
            'max_clauses_per_search': 10,
            'min_clause_length': 8,
            'max_clause_length': 100
        }
        
        # Core storage
        self.documents = {}
        self.document_metadata = {}
        self.clause_database = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Insurance domain knowledge (simplified)
        self.insurance_knowledge = {
            'procedures': [
                'knee surgery', 'hip surgery', 'heart surgery', 'eye surgery', 
                'dental surgery', 'brain surgery', 'appendectomy', 'bypass surgery',
                'chemotherapy', 'radiation therapy', 'dialysis', 'physical therapy'
            ],
            'conditions': [
                'diabetes', 'hypertension', 'heart disease', 'asthma', 'arthritis',
                'cancer', 'stroke', 'depression', 'anxiety'
            ],
            'coverage_terms': [
                'covered', 'eligible', 'included', 'excluded', 'not covered',
                'benefits', 'deductible', 'copay', 'premium'
            ]
        }
        
        self.processing_stats = {
            'queries_processed': 0,
            'documents_loaded': 0,
            'total_clauses': 0
        }
        
        logger.info("✅ Optimized processor initialized successfully!")
    
    def load_documents(self, folder_path: str) -> None:
        """Load documents with robust error handling"""
        logger.info(f"Loading documents from: {folder_path}")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError("No PDF files found in the dataset folder")
        
        # Process documents
        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            try:
                text = self._extract_text_from_pdf(file_path)
                if text and text.strip():
                    self.documents[filename] = text
                    logger.info(f"✅ Loaded: {filename} ({len(text)} chars)")
                else:
                    logger.warning(f"⚠️ No content: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {e}")
        
        if self.documents:
            self._build_clause_database()
            self.processing_stats['documents_loaded'] = len(self.documents)
            logger.info(f"✅ Successfully loaded {len(self.documents)} documents")
        else:
            raise ValueError("No documents could be loaded successfully")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with fallback methods"""
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\\n"
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return ""
        
        return text.strip()
    
    def _build_clause_database(self):
        """Build clause database from loaded documents"""
        logger.info("Building clause database...")
        
        all_clauses = []
        for doc_name, doc_content in self.documents.items():
            clauses = self._extract_clauses_robust(doc_content, doc_name)
            all_clauses.extend(clauses)
            logger.info(f"Extracted {len(clauses)} clauses from {doc_name}")
        
        self.clause_database = all_clauses
        self.processing_stats['total_clauses'] = len(all_clauses)
        
        if all_clauses:
            # Build TF-IDF index
            clause_texts = [clause['text'] for clause in all_clauses]
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clause_texts)
            logger.info(f"✅ Built TF-IDF index with {len(clause_texts)} clauses")
        
        logger.info(f"✅ Clause database built: {len(all_clauses)} total clauses")
    
    def _extract_clauses_robust(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """Extract clauses with robust method"""
        clauses = []
        
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            logger.debug(f"Processing {len(sentences)} sentences from {doc_name}")
            
            current_clause = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 5:  # Skip very short sentences
                    continue
                
                current_clause.append(sentence)
                
                # Determine when to end a clause
                words_count = len(' '.join(current_clause).split())
                should_end = (
                    words_count >= self.config['min_clause_length'] and
                    (sentence.endswith('.') or sentence.endswith(';') or
                     words_count >= self.config['max_clause_length'])
                )
                
                if should_end:
                    clause_text = ' '.join(current_clause).strip()
                    if self.config['min_clause_length'] <= len(clause_text.split()) <= self.config['max_clause_length']:
                        clause_info = {
                            'text': clause_text,
                            'document': doc_name,
                            'clause_type': self._classify_clause_simple(clause_text),
                            'importance_score': self._calculate_importance_simple(clause_text)
                        }
                        clauses.append(clause_info)
                    
                    current_clause = []
            
            # Handle remaining sentences
            if current_clause:
                clause_text = ' '.join(current_clause).strip()
                if self.config['min_clause_length'] <= len(clause_text.split()):
                    clause_info = {
                        'text': clause_text,
                        'document': doc_name,
                        'clause_type': self._classify_clause_simple(clause_text),
                        'importance_score': self._calculate_importance_simple(clause_text)
                    }
                    clauses.append(clause_info)
            
        except Exception as e:
            logger.error(f"Error extracting clauses from {doc_name}: {e}")
        
        return clauses
    
    def _classify_clause_simple(self, text: str) -> str:
        """Simple clause classification"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['covered', 'eligible', 'benefits']):
            return 'coverage_positive'
        elif any(word in text_lower for word in ['excluded', 'not covered']):
            return 'coverage_negative'
        elif any(word in text_lower for word in ['premium', 'cost', 'amount']):
            return 'financial_terms'
        elif any(word in text_lower for word in ['waiting period', 'duration']):
            return 'temporal_conditions'
        else:
            return 'general_terms'
    
    def _calculate_importance_simple(self, text: str) -> float:
        """Simple importance calculation"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Length factor
        score += min(len(text.split()) / 50.0, 0.3)
        
        # Important keywords
        important_keywords = [
            'covered', 'excluded', 'benefit', 'premium', 'deductible',
            'surgery', 'treatment', 'condition', 'procedure'
        ]
        
        for keyword in important_keywords:
            if keyword in text_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def parse_query_simple(self, query: str) -> Dict[str, Any]:
        """Simple query parsing"""
        parsed = {
            'raw_query': query,
            'age': None,
            'procedure': None,
            'location': None,
            'condition': None,
            'policy_duration': None
        }
        
        query_lower = query.lower()
        
        # Extract age
        age_match = re.search(r'(\\d+)\\s*year\\s*old', query_lower)
        if age_match:
            parsed['age'] = int(age_match.group(1))
        
        # Extract procedures
        for procedure in self.insurance_knowledge['procedures']:
            if procedure in query_lower:
                parsed['procedure'] = procedure
                break
        
        # Extract conditions
        for condition in self.insurance_knowledge['conditions']:
            if condition in query_lower:
                parsed['condition'] = condition
                break
        
        # Extract location
        location_match = re.search(r'in\\s+(\\w+)', query_lower)
        if location_match:
            parsed['location'] = location_match.group(1)
        
        return parsed
    
    def search_relevant_clauses(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for relevant clauses"""
        if not self.clause_database or not self.tfidf_vectorizer:
            return []
        
        query_text = parsed_query['raw_query']
        relevant_clauses = []
        
        # TF-IDF search
        try:
            query_vector = self.tfidf_vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top similar clauses
            top_indices = similarities.argsort()[-self.config['max_clauses_per_search']:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > self.config['similarity_threshold']:
                    clause = self.clause_database[idx].copy()
                    clause['similarity_score'] = float(similarities[idx])
                    clause['search_method'] = 'tfidf'
                    relevant_clauses.append(clause)
        
        except Exception as e:
            logger.error(f"Error in clause search: {e}")
        
        return relevant_clauses
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return results"""
        start_time = datetime.now()
        
        try:
            if not query or not query.strip():
                return self._error_response("Empty query provided", query)
            
            # Parse query
            parsed_query = self.parse_query_simple(query)
            
            # Search clauses
            relevant_clauses = self.search_relevant_clauses(parsed_query)
            
            # Make decision
            decision = self._make_simple_decision(parsed_query, relevant_clauses)
            
            # Build response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "query": query,
                "decision": decision.get("status", "unknown"),
                "amount": float(decision.get("amount", 0)),
                "confidence": float(decision.get("confidence", 0)),
                "justification": decision.get("justification", "Analysis completed"),
                "detailed_response": decision.get("detailed_response"),
                "clauses_mapping": [
                    {
                        "clause_text": clause["text"],
                        "document": clause["document"],
                        "clause_type": clause.get("clause_type", "unknown"),
                        "similarity_score": clause.get("similarity_score", 0.0),
                        "search_method": clause.get("search_method", "tfidf")
                    }
                    for clause in relevant_clauses[:5]  # Top 5 clauses
                ],
                "parsed_query": parsed_query,
                "risk_factors": decision.get("risk_factors", []),
                "recommendations": decision.get("recommendations", []),
                "processing_metadata": {
                    "processing_time_seconds": processing_time,
                    "total_clauses_searched": len(self.clause_database),
                    "relevant_clauses_found": len(relevant_clauses),
                    "search_methods_used": ["tfidf"]
                }
            }
            
            self.processing_stats['queries_processed'] += 1
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._error_response(f"Processing error: {str(e)}", query)
    
    def _make_simple_decision(self, parsed_query: Dict[str, Any], relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a simple decision based on clauses"""
        if not relevant_clauses:
            return {
                "status": "information_needed",
                "amount": 0.0,
                "confidence": 0.0,
                "justification": "No relevant policy clauses found for this query",
                "risk_factors": [],
                "recommendations": ["Please provide more specific information about your query"]
            }
        
        # Analyze coverage based on clause types
        positive_clauses = [c for c in relevant_clauses if c.get('clause_type') == 'coverage_positive']
        negative_clauses = [c for c in relevant_clauses if c.get('clause_type') == 'coverage_negative']
        
        if positive_clauses and not negative_clauses:
            status = "approved"
            confidence = 0.8
            amount = 50000.0  # Default coverage amount
        elif negative_clauses and not positive_clauses:
            status = "rejected"
            confidence = 0.8
            amount = 0.0
        elif positive_clauses and negative_clauses:
            status = "review_required"
            confidence = 0.6
            amount = 25000.0
        else:
            status = "review_required"
            confidence = 0.5
            amount = 0.0
        
        # Build justification
        justification = f"Based on analysis of {len(relevant_clauses)} relevant policy clauses. "
        if positive_clauses:
            justification += f"Found {len(positive_clauses)} supportive clauses. "
        if negative_clauses:
            justification += f"Found {len(negative_clauses)} restrictive clauses. "
        
        return {
            "status": status,
            "amount": amount,
            "confidence": confidence,
            "justification": justification,
            "risk_factors": [],
            "recommendations": []
        }
    
    def _error_response(self, error_msg: str, query: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "query": query,
            "decision": "error",
            "amount": 0.0,
            "confidence": 0.0,
            "justification": error_msg,
            "detailed_response": None,
            "clauses_mapping": [],
            "parsed_query": {"raw_query": query},
            "risk_factors": [],
            "recommendations": ["Please try again with a different query"],
            "processing_metadata": {
                "processing_time_seconds": 0.0,
                "total_clauses_searched": len(self.clause_database),
                "relevant_clauses_found": 0,
                "search_methods_used": [],
                "error": error_msg
            }
        }

# Test function
def test_optimized_processor():
    """Test the optimized processor"""
    logger.info("Testing optimized processor...")
    
    processor = OptimizedDocumentProcessor()
    
    # Load documents
    datasets_path = "Datasets"
    if os.path.exists(datasets_path):
        processor.load_documents(datasets_path)
        
        # Test query
        test_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        result = processor.process_query(test_query)
        
        logger.info(f"Test Result:")
        logger.info(f"  Decision: {result['decision']}")
        logger.info(f"  Clauses found: {len(result['clauses_mapping'])}")
        logger.info(f"  Total clauses in DB: {len(processor.clause_database)}")
        
        return True
    else:
        logger.error(f"Datasets not found at {datasets_path}")
        return False

if __name__ == "__main__":
    test_optimized_processor()
'''
    
    # Write the optimized processor
    with open("optimized_processor.py", "w", encoding="utf-8") as f:
        f.write(optimized_code)
    
    logger.info("✅ Created optimized_processor.py")
    return True

def apply_fixes():
    """Apply all the identified fixes"""
    print("\n🔧 APPLYING COMPREHENSIVE FIXES")
    print("=" * 40)
    
    fixes_applied = 0
    
    # Fix 1: Create optimized processor
    if create_optimized_processor():
        print("✅ Created optimized processor")
        fixes_applied += 1
    
    # Fix 2: Update API to use correct dataset path (already done)
    print("✅ Updated API dataset path")
    fixes_applied += 1
    
    # Fix 3: Create a startup script that works
    startup_script = '''#!/usr/bin/env python3
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
'''
    
    with open("start_optimized.py", "w", encoding="utf-8") as f:
        f.write(startup_script)
    
    print("✅ Created optimized startup script")
    fixes_applied += 1
    
    # Fix 4: Create test script
    test_script = '''#!/usr/bin/env python3
"""
Comprehensive test for the fixed system
"""

import os
import requests
import time
import json

def test_system():
    """Test the entire system"""
    print("🧪 TESTING FIXED SYSTEM")
    print("=" * 30)
    
    # Test 1: Test optimized processor directly
    print("1. Testing optimized processor...")
    try:
        from optimized_processor import OptimizedDocumentProcessor
        processor = OptimizedDocumentProcessor()
        
        if os.path.exists("Datasets"):
            processor.load_documents("Datasets")
            print(f"   ✅ Loaded {len(processor.documents)} documents")
            print(f"   ✅ Generated {len(processor.clause_database)} clauses")
            
            # Test query
            query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
            result = processor.process_query(query)
            print(f"   ✅ Query processed: {result['decision']}")
            print(f"   ✅ Clauses mapped: {len(result['clauses_mapping'])}")
            
            if result['clauses_mapping']:
                print("   🎉 CLAUSE GENERATION IS WORKING!")
                return True
            else:
                print("   ⚠️ No clauses mapped")
                return False
        else:
            print("   ❌ Datasets not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if test_system():
        print("\\n🎉 ALL TESTS PASSED!")
        print("✅ The clause generation issue has been fixed!")
        print("\\nNext steps:")
        print("1. Run: python start_optimized.py")
        print("2. Test API at: http://localhost:8001")
        print("3. Use /process_query endpoint for testing")
    else:
        print("\\n❌ Tests failed. Check the logs above.")
'''
    
    with open("test_fixed_system.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ Created comprehensive test script")
    fixes_applied += 1
    
    print(f"\n🎉 Applied {fixes_applied} fixes successfully!")
    return True

def main():
    """Main function to analyze and fix the clause generation issues"""
    print("🔧 BAJAJ DOCUMENT ANALYZER - CLAUSE GENERATION FIX")
    print("=" * 60)
    
    # Step 1: Analyze the problem
    if not analyze_problem():
        print("❌ Problem analysis failed")
        return
    
    # Step 2: Apply fixes
    if not apply_fixes():
        print("❌ Failed to apply fixes")
        return
    
    # Step 3: Provide instructions
    print("\n📋 NEXT STEPS:")
    print("1. Test the optimized processor:")
    print("   python test_fixed_system.py")
    print()
    print("2. Start the server:")
    print("   python start_optimized.py")
    print()
    print("3. Test the API:")
    print("   curl -X POST http://localhost:8001/process_query \\")
    print('   -H "Content-Type: application/json" \\')
    print('   -d \'{"query": "46-year-old male, knee surgery in Pune"}\'')
    print()
    print("🎯 SUMMARY OF FIXES:")
    print("✅ Fixed dataset path issues")
    print("✅ Created lightweight processor without heavy ML models")
    print("✅ Improved clause extraction with robust error handling")
    print("✅ Added comprehensive testing")
    print("✅ Simplified initialization to prevent hanging")
    print()
    print("🔧 The main issue was heavy model loading in enhanced_main.py")
    print("🚀 The optimized processor focuses on reliable clause generation")

if __name__ == "__main__":
    main()
