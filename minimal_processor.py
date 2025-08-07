#!/usr/bin/env python3
"""
Minimal Working Document Processor for Bajaj Document Analyzer
This version focuses on reliability and working clause generation
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalDocumentProcessor:
    """
    Minimal working processor that guarantees clause generation
    """
    
    def __init__(self):
        """Initialize with minimal dependencies"""
        logger.info("Initializing Minimal Document Processor...")
        
        self.documents = {}
        self.clause_database = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Simple insurance knowledge
        self.insurance_terms = {
            'procedures': ['surgery', 'treatment', 'procedure', 'operation', 'therapy'],
            'conditions': ['disease', 'condition', 'illness', 'medical', 'health'],
            'coverage': ['covered', 'eligible', 'benefit', 'included', 'excluded'],
            'financial': ['premium', 'deductible', 'amount', 'cost', 'payment']
        }
        
        self.stats = {'queries_processed': 0, 'documents_loaded': 0, 'total_clauses': 0}
        logger.info("✅ Minimal processor initialized")
    
    def load_documents(self, folder_path: str) -> None:
        """Load documents with simple extraction"""
        logger.info(f"Loading documents from: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            self._create_sample_clauses()
            return
        
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning("No PDF files found, creating sample data")
            self._create_sample_clauses()
            return
        
        # Process PDFs with simple extraction
        for filename in pdf_files[:2]:  # Limit to 2 files for speed
            file_path = os.path.join(folder_path, filename)
            try:
                text = self._extract_text_simple(file_path)
                if text:
                    self.documents[filename] = text
                    logger.info(f"✅ Loaded: {filename}")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
        
        # Build clause database
        self._build_clause_database()
        self.stats['documents_loaded'] = len(self.documents)
        logger.info(f"✅ Loaded {len(self.documents)} documents, {len(self.clause_database)} clauses")
    
    def _extract_text_simple(self, file_path: str) -> str:
        """Simple PDF text extraction"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Extract from first 5 pages only for speed
                for i, page in enumerate(pdf_reader.pages[:5]):
                    text += page.extract_text() + " "
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
        return text.strip()
    
    def _build_clause_database(self):
        """Build clause database from documents"""
        all_clauses = []
        
        for doc_name, doc_content in self.documents.items():
            clauses = self._extract_clauses_simple(doc_content, doc_name)
            all_clauses.extend(clauses)
        
        # Add sample clauses if none found
        if not all_clauses:
            all_clauses = self._create_sample_clauses()
        
        self.clause_database = all_clauses
        self.stats['total_clauses'] = len(all_clauses)
        
        # Build TF-IDF index
        if all_clauses:
            clause_texts = [clause['text'] for clause in all_clauses]
            self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clause_texts)
        
        logger.info(f"✅ Built clause database: {len(all_clauses)} clauses")
    
    def _extract_clauses_simple(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """Simple clause extraction"""
        clauses = []
        
        # Split into sentences using simple regex
        sentences = re.split(r'[.!?]+', text)
        
        current_clause = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            current_clause.append(sentence)
            
            # Create clause every 2-3 sentences
            if len(current_clause) >= 2:
                clause_text = '. '.join(current_clause).strip()
                
                if 10 <= len(clause_text.split()) <= 50:  # Reasonable clause length
                    clause_info = {
                        'text': clause_text,
                        'document': doc_name,
                        'clause_type': self._classify_clause_simple(clause_text),
                        'importance_score': self._calculate_importance(clause_text)
                    }
                    clauses.append(clause_info)
                
                current_clause = []
        
        return clauses
    
    def _classify_clause_simple(self, text: str) -> str:
        """Simple clause classification"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['covered', 'eligible', 'benefit']):
            return 'coverage_positive'
        elif any(term in text_lower for term in ['excluded', 'not covered']):
            return 'coverage_negative'
        elif any(term in text_lower for term in ['premium', 'cost', 'amount']):
            return 'financial_terms'
        else:
            return 'general_terms'
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score"""
        score = 0.0
        text_lower = text.lower()
        
        # Important keywords
        important_words = ['covered', 'excluded', 'benefit', 'premium', 'surgery', 'treatment']
        for word in important_words:
            if word in text_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def _create_sample_clauses(self) -> List[Dict[str, Any]]:
        """Create sample clauses for demonstration"""
        sample_clauses = [
            {
                'text': 'Knee surgery is covered under this policy subject to pre-authorization and waiting period conditions.',
                'document': 'sample_policy.pdf',
                'clause_type': 'coverage_positive',
                'importance_score': 0.8
            },
            {
                'text': 'The policy covers medical expenses for surgery performed in network hospitals with valid medical necessity.',
                'document': 'sample_policy.pdf',
                'clause_type': 'coverage_positive',
                'importance_score': 0.7
            },
            {
                'text': 'Pre-existing diseases are covered after 2 years of waiting period from policy inception date.',
                'document': 'sample_policy.pdf',
                'clause_type': 'coverage_positive',
                'importance_score': 0.6
            },
            {
                'text': 'Premium payment grace period is 30 days from due date, after which policy may lapse.',
                'document': 'sample_policy.pdf',
                'clause_type': 'financial_terms',
                'importance_score': 0.5
            },
            {
                'text': 'Emergency treatments are covered without pre-authorization within 24 hours of admission.',
                'document': 'sample_policy.pdf',
                'clause_type': 'coverage_positive',
                'importance_score': 0.9
            }
        ]
        
        self.clause_database = sample_clauses
        logger.info(f"✅ Created {len(sample_clauses)} sample clauses")
        return sample_clauses
    
    def parse_query_simple(self, query: str) -> Dict[str, Any]:
        """Simple query parsing"""
        query_lower = query.lower()
        
        parsed = {
            'raw_query': query,
            'age': None,
            'procedure': None,
            'location': None,
            'keywords': []
        }
        
        # Extract age
        age_match = re.search(r'(\d+)[\s-]*year', query_lower)
        if age_match:
            parsed['age'] = int(age_match.group(1))
        
        # Extract procedure
        procedures = ['surgery', 'treatment', 'knee', 'hip', 'heart', 'eye']
        for proc in procedures:
            if proc in query_lower:
                parsed['procedure'] = proc
                break
        
        # Extract location
        location_match = re.search(r'in\s+(\w+)', query_lower)
        if location_match:
            parsed['location'] = location_match.group(1)
        
        # Extract keywords
        for category, terms in self.insurance_terms.items():
            for term in terms:
                if term in query_lower:
                    parsed['keywords'].append(term)
        
        return parsed
    
    def search_relevant_clauses(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for relevant clauses"""
        if not self.clause_database:
            return []
        
        query_text = parsed_query['raw_query']
        relevant_clauses = []
        
        # TF-IDF search if available
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                query_vector = self.tfidf_vectorizer.transform([query_text])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                
                # Get top 5 similar clauses
                top_indices = similarities.argsort()[-5:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Minimum similarity
                        clause = self.clause_database[idx].copy()
                        clause['similarity_score'] = float(similarities[idx])
                        clause['search_method'] = 'tfidf'
                        relevant_clauses.append(clause)
            except Exception as e:
                logger.warning(f"TF-IDF search failed: {e}")
        
        # Keyword-based fallback
        if not relevant_clauses:
            for clause in self.clause_database:
                clause_text_lower = clause['text'].lower()
                query_lower = query_text.lower()
                
                # Simple keyword matching
                matches = 0
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 3 and word in clause_text_lower:
                        matches += 1
                
                if matches > 0:
                    clause_copy = clause.copy()
                    clause_copy['similarity_score'] = matches / len(query_words)
                    clause_copy['search_method'] = 'keyword'
                    relevant_clauses.append(clause_copy)
        
        # Sort by similarity score
        relevant_clauses.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return relevant_clauses[:5]  # Top 5
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return results"""
        start_time = datetime.now()
        
        try:
            if not query or not query.strip():
                return self._error_response("Empty query", query)
            
            # Parse query
            parsed_query = self.parse_query_simple(query)
            
            # Search clauses
            relevant_clauses = self.search_relevant_clauses(parsed_query)
            
            # Make decision
            decision = self._make_decision(parsed_query, relevant_clauses)
            
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
                        "search_method": clause.get("search_method", "keyword")
                    }
                    for clause in relevant_clauses
                ],
                "parsed_query": parsed_query,
                "risk_factors": decision.get("risk_factors", []),
                "recommendations": decision.get("recommendations", []),
                "processing_metadata": {
                    "processing_time_seconds": processing_time,
                    "total_clauses_searched": len(self.clause_database),
                    "relevant_clauses_found": len(relevant_clauses),
                    "processor_type": "minimal"
                }
            }
            
            self.stats['queries_processed'] += 1
            return response
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return self._error_response(f"Processing error: {str(e)}", query)
    
    def _make_decision(self, parsed_query: Dict[str, Any], relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a simple decision"""
        if not relevant_clauses:
            return {
                "status": "information_needed",
                "amount": 0.0,
                "confidence": 0.0,
                "justification": "No relevant policy clauses found",
                "risk_factors": [],
                "recommendations": ["Please provide more specific information"]
            }
        
        # Simple decision logic
        positive_clauses = [c for c in relevant_clauses if c.get('clause_type') == 'coverage_positive']
        negative_clauses = [c for c in relevant_clauses if c.get('clause_type') == 'coverage_negative']
        
        if positive_clauses and not negative_clauses:
            status = "approved"
            confidence = 0.8
            amount = 50000.0
        elif negative_clauses:
            status = "rejected"
            confidence = 0.7
            amount = 0.0
        else:
            status = "review_required"
            confidence = 0.6
            amount = 25000.0
        
        justification = f"Decision based on {len(relevant_clauses)} relevant policy clauses. "
        if positive_clauses:
            justification += f"Found {len(positive_clauses)} supportive clauses. "
        if negative_clauses:
            justification += f"Found {len(negative_clauses)} restrictive clauses."
        
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
            "recommendations": [],
            "processing_metadata": {
                "processing_time_seconds": 0.0,
                "total_clauses_searched": len(self.clause_database),
                "relevant_clauses_found": 0,
                "error": error_msg
            }
        }

# Test function
def test_minimal_processor():
    """Test the minimal processor"""
    print("🧪 TESTING MINIMAL PROCESSOR")
    print("=" * 35)
    
    processor = MinimalDocumentProcessor()
    
    # Load documents
    processor.load_documents("Datasets")
    
    # Test queries
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "Does this policy cover maternity expenses?",
        "What is the waiting period for pre-existing diseases?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = processor.process_query(query)
        print(f"   Decision: {result['decision']}")
        print(f"   Amount: ₹{result['amount']:,.0f}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Clauses: {len(result['clauses_mapping'])}")
        
        if result['clauses_mapping']:
            print("   Sample clause:", result['clauses_mapping'][0]['clause_text'][:80] + "...")
    
    print(f"\n✅ Test complete! Processed {processor.stats['queries_processed']} queries")
    return True

if __name__ == "__main__":
    test_minimal_processor()
