#!/usr/bin/env python3
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

# Text processing without heavy dependencies
import re

def sent_tokenize(text):
    """Simple sentence tokenizer without NLTK dependency"""
    # Split on sentence endings, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def word_tokenize(text):
    """Simple word tokenizer"""
    return re.findall(r'\b\w+\b', text.lower())

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
                    text += page.extract_text() + "\n"
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
        age_match = re.search(r'(\d+)\s*year\s*old', query_lower)
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
        location_match = re.search(r'in\s+(\w+)', query_lower)
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
