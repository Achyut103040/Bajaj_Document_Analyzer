#!/usr/bin/env python3
"""
Lightweight demonstration version of the LLM Document Processing System
This version loads quickly for hackathon demonstrations
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Basic libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightDocumentProcessor:
    """
    Lightweight version for quick demonstration
    Focuses on core functionality without heavy model loading
    """
    
    def __init__(self):
        """Initialize the lightweight processor"""
        logger.info("Initializing Lightweight Document Processor...")
        
        # Document storage
        self.documents = {}
        self.clause_database = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Insurance domain knowledge
        self.insurance_knowledge = {
            'procedures': [
                'knee surgery', 'hip surgery', 'heart surgery', 'cardiac surgery', 
                'eye surgery', 'cataract surgery', 'dental surgery', 'brain surgery',
                'appendectomy', 'gallbladder surgery', 'hernia surgery', 'bypass surgery',
                'angioplasty', 'transplant surgery', 'spine surgery', 'chemotherapy',
                'radiation therapy', 'dialysis', 'physical therapy'
            ],
            'conditions': [
                'diabetes', 'hypertension', 'heart disease', 'asthma', 'arthritis',
                'chronic kidney disease', 'cancer', 'stroke', 'depression', 'anxiety'
            ],
            'coverage_terms': [
                'covered', 'eligible', 'included', 'excluded', 'not covered', 
                'benefits', 'deductible', 'copay', 'premium', 'limitation'
            ]
        }
        
        # Processing statistics
        self.stats = {
            'queries_processed': 0,
            'documents_loaded': 0,
            'total_clauses': 0
        }
        
        logger.info("Lightweight processor initialized successfully!")
    
    def load_documents(self, folder_path: str) -> None:
        """Load and process documents quickly"""
        logger.info(f"Loading documents from: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.warning(f"Dataset folder not found: {folder_path}")
            # Create sample data for demonstration
            self._create_sample_data()
            return
        
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning("No PDF files found. Creating sample data for demonstration.")
            self._create_sample_data()
            return
        
        # Process PDFs
        for filename in pdf_files:  # Process all PDF files
            file_path = os.path.join(folder_path, filename)
            try:
                text = self._extract_text_from_pdf(file_path)
                if text.strip():
                    self.documents[filename] = text
                    logger.info(f"Loaded: {filename}")
            except Exception as e:
                logger.warning(f"Could not load {filename}: {e}")
        
        # If no documents loaded, create sample data
        if not self.documents:
            self._create_sample_data()
        
        # Build knowledge base
        self._build_knowledge_base()
        self.stats['documents_loaded'] = len(self.documents)
        logger.info(f"Loaded {len(self.documents)} documents with {len(self.clause_database)} clauses")
        
        # If no clauses were extracted, create sample data
        if len(self.clause_database) == 0:
            logger.warning("No clauses extracted, adding sample insurance data")
            self._add_sample_clauses()
    
    def _create_sample_data(self):
        """Create sample insurance policy data for demonstration"""
        sample_policies = {
            "Sample_Health_Policy.pdf": """
            COMPREHENSIVE HEALTH INSURANCE POLICY
            
            COVERAGE DETAILS:
            
            Surgical Procedures:
            - Knee surgery: Covered up to Rs. 1,00,000
            - Heart surgery: Covered up to Rs. 3,00,000
            - Eye surgery including cataract: Covered up to Rs. 50,000
            - Dental surgery: Covered up to Rs. 25,000
            - Brain surgery: Covered up to Rs. 5,00,000
            
            Age Limits:
            - Children (0-17 years): Full coverage
            - Adults (18-64 years): Full coverage  
            - Senior citizens (65+ years): 80% coverage
            
            Geographic Coverage:
            - Treatment in Mumbai, Delhi, Pune, Bangalore: 100% coverage
            - Treatment in other cities: 90% coverage
            
            Exclusions:
            - Pre-existing conditions for first 2 years
            - Cosmetic surgery not covered
            - Treatment due to alcohol or drug abuse not covered
            
            Emergency Services:
            - Emergency surgery: Immediate coverage up to Rs. 2,00,000
            - Ambulance services: Covered up to Rs. 5,000
            
            Waiting Periods:
            - General surgery: 30 days waiting period
            - Specific diseases: 2 years waiting period
            - Maternity: 10 months waiting period
            """,
            
            "Sample_Dental_Policy.pdf": """
            DENTAL CARE INSURANCE POLICY
            
            DENTAL COVERAGE:
            
            Preventive Care:
            - Regular checkups: Covered 100%
            - Teeth cleaning: Covered twice per year
            - X-rays: Covered 100%
            
            Basic Procedures:
            - Fillings: Covered up to Rs. 5,000 per tooth
            - Root canal: Covered up to Rs. 15,000
            - Tooth extraction: Covered up to Rs. 3,000
            
            Major Procedures:
            - Dental surgery: Covered up to Rs. 25,000
            - Implants: Covered up to Rs. 50,000
            - Orthodontics: Covered up to Rs. 75,000
            
            Age-based Benefits:
            - Children under 18: 100% coverage
            - Adults 18-60: 80% coverage
            - Seniors 60+: 70% coverage
            
            Annual Limits:
            - Maximum benefit per year: Rs. 1,00,000
            - Maximum per procedure varies by type
            """,
            
            "Sample_Emergency_Policy.pdf": """
            EMERGENCY MEDICAL INSURANCE
            
            EMERGENCY COVERAGE:
            
            Immediate Coverage:
            - Life-threatening conditions: 100% coverage
            - Emergency surgery: Up to Rs. 5,00,000
            - ICU treatment: Up to Rs. 50,000 per day
            - Emergency room visits: Covered 100%
            
            Ambulance Services:
            - Ground ambulance: Up to Rs. 5,000
            - Air ambulance: Up to Rs. 1,00,000
            
            Geographic Coverage:
            - Nationwide coverage
            - International emergency: Up to Rs. 10,00,000
            
            Age Groups:
            - All ages covered
            - Newborns covered from birth
            - No upper age limit
            
            Pre-authorization:
            - Not required for emergencies
            - Documentation required within 24 hours
            """
        }
        
        self.documents = sample_policies
        logger.info("Created sample policy data for demonstration")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF files"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
        return text
    
    def _build_knowledge_base(self):
        """Build knowledge base with TF-IDF"""
        logger.info("Building knowledge base...")
        
        # Extract clauses
        all_clauses = []
        for doc_name, doc_content in self.documents.items():
            clauses = self._extract_clauses(doc_content, doc_name)
            all_clauses.extend(clauses)
        
        self.clause_database = all_clauses
        self.stats['total_clauses'] = len(all_clauses)
        
        if all_clauses:
            # Build TF-IDF index
            clause_texts = [clause['text'] for clause in all_clauses]
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clause_texts)
            
            logger.info(f"Knowledge base built with {len(all_clauses)} clauses")
    
    def _extract_clauses(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """Extract clauses from document text"""
        clauses = []
        
        # Split by lines and group into meaningful clauses
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_clause = []
        for line in lines:
            if line.startswith('---') or len(line) < 10:
                continue
            
            current_clause.append(line)
            
            # End clause on certain patterns
            if (len(current_clause) >= 2 and 
                (line.endswith('.') or line.endswith(':') or 
                 any(term in line.lower() for term in ['covered', 'excluded', 'limit']))):
                
                clause_text = ' '.join(current_clause)
                if len(clause_text.split()) >= 5:  # Minimum clause length
                    clause_info = {
                        'text': clause_text,
                        'document': doc_name,
                        'clause_type': self._classify_clause(clause_text)
                    }
                    clauses.append(clause_info)
                
                current_clause = []
        
        return clauses
    
    def _classify_clause(self, text: str) -> str:
        """Classify clause type"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['covered', 'eligible', 'included']):
            return 'coverage_positive'
        elif any(word in text_lower for word in ['excluded', 'not covered', 'limitation']):
            return 'coverage_negative'
        elif any(word in text_lower for word in ['premium', 'cost', 'amount', 'limit']):
            return 'financial'
        elif any(word in text_lower for word in ['waiting', 'period', 'time']):
            return 'temporal'
        else:
            return 'general'
    
    def parse_query_advanced(self, query: str) -> Dict[str, Any]:
        """Parse natural language query"""
        logger.info(f"Parsing query: {query}")
        
        parsed_data = {
            "age": None,
            "gender": None,
            "procedure": None,
            "location": None,
            "policy_duration": None,
            "urgency": "normal",
            "intent": "coverage_inquiry",
            "confidence": 0.0,
            "raw_query": query
        }
        
        query_lower = query.lower()
        
        # Extract age
        age_patterns = [
            r'(\d+)[-\s]?year[-\s]?old',
            r'(\d+)[-\s]?yrs?[-\s]?old',
            r'age\s*:?\s*(\d+)',
            r'(\d+)\s*years?\s*of\s*age'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parsed_data["age"] = int(match.group(1))
                break
        
        # Extract gender
        if re.search(r'\bmale\b', query_lower) and not re.search(r'\bfemale\b', query_lower):
            parsed_data["gender"] = "male"
        elif re.search(r'\bfemale\b|\bwoman\b', query_lower):
            parsed_data["gender"] = "female"
        
        # Extract procedures
        for procedure in self.insurance_knowledge['procedures']:
            if procedure in query_lower:
                parsed_data["procedure"] = procedure
                break
        
        # Extract location
        cities = ['mumbai', 'delhi', 'pune', 'bangalore', 'hyderabad', 'chennai']
        for city in cities:
            if city in query_lower:
                parsed_data["location"] = city
                break
        
        # Extract urgency
        if any(word in query_lower for word in ['emergency', 'urgent', 'immediate']):
            parsed_data["urgency"] = "high"
        
        # Calculate confidence based on multiple factors
        confidence_factors = []
        
        # Data completeness (30% weight)
        data_completeness = [
            parsed_data["age"] is not None,
            parsed_data["gender"] is not None,
            parsed_data["procedure"] is not None,
            parsed_data["location"] is not None
        ]
        completeness_score = sum(data_completeness) / len(data_completeness)
        confidence_factors.append(completeness_score * 0.3)
        
        # Query clarity (40% weight) 
        query_length = len(parsed_data["raw_query"].split())
        clarity_score = min(query_length / 10, 1.0)  # Normalize to 1.0
        confidence_factors.append(clarity_score * 0.4)
        
        # Medical procedure detection (30% weight)
        procedure_score = 1.0 if parsed_data["procedure"] else 0.3
        confidence_factors.append(procedure_score * 0.3)
        
        # Final confidence (minimum 20% for any valid query)
        parsed_data["confidence"] = max(sum(confidence_factors), 0.2)
        
        return parsed_data
    
    def search_relevant_clauses_hybrid(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for relevant clauses using TF-IDF"""
        if not self.clause_database or self.tfidf_vectorizer is None:
            return []
        
        query_text = parsed_query["raw_query"]
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top similar clauses
        top_indices = similarities.argsort()[-10:][::-1]  # Top 10
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                clause = self.clause_database[idx].copy()
                clause['similarity_score'] = float(similarities[idx])
                clause['search_method'] = 'tfidf'
                results.append(clause)
        
        return results
    
    def evaluate_decision_advanced(self, parsed_query: Dict[str, Any], relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate decision based on clauses"""
        decision = {
            "status": "pending",
            "amount": 0.0,
            "confidence": 0.0,
            "justification": "Analyzing coverage...",
            "risk_factors": [],
            "recommendations": [],
            "clauses_used": []
        }
        
        if not relevant_clauses:
            decision["status"] = "information_needed"
            decision["justification"] = "No relevant policy clauses found for this query"
            decision["recommendations"] = ["Please provide more specific information about the treatment needed"]
            return decision
        
        # Analyze coverage indicators
        coverage_score = 0
        exclusion_score = 0
        amount_found = 0
        
        for clause in relevant_clauses:
            clause_text = clause.get('text', '')
            if not clause_text or not isinstance(clause_text, str):
                continue
                
            clause_text = clause_text.lower()
            
            # Check for positive coverage
            if any(term in clause_text for term in ['covered', 'eligible', 'included']):
                coverage_score += clause.get('similarity_score', 0.5)
            
            # Check for exclusions
            if any(term in clause_text for term in ['excluded', 'not covered', 'limitation']):
                exclusion_score += clause.get('similarity_score', 0.5)
            
            # Extract amounts
            amounts = re.findall(r'rs\.?\s*[\d,]+|[\d,]+\s*rupees?', clause_text)
            for amount_str in amounts:
                try:
                    amount_num = int(re.sub(r'[^\d]', '', amount_str))
                    if amount_num > amount_found:
                        amount_found = amount_num
                except:
                    pass
        
        # Calculate risk factors
        risk_factors = []
        age = parsed_query.get('age')
        if age:
            if age < 5 or age > 65:
                risk_factors.append(f"Age {age} may have special coverage considerations")
        
        if parsed_query.get('urgency') == 'high':
            risk_factors.append("Emergency treatment - priority processing")
        
        # Make decision with improved confidence calculation
        base_confidence = max(coverage_score, exclusion_score, 0.1)  # Minimum 10%
        
        if coverage_score > exclusion_score and coverage_score > 0.3:
            if len(risk_factors) == 0:
                decision["status"] = "approved"
                decision["amount"] = max(amount_found, 50000)  # Default minimum
                # High confidence for clear approvals
                decision["confidence"] = min(base_confidence * 1.2, 0.95)
            else:
                decision["status"] = "conditional_approval"
                decision["amount"] = max(amount_found * 0.8, 25000)
                # Moderate confidence for conditional approvals
                decision["confidence"] = min(base_confidence * 0.9, 0.80)
        elif exclusion_score > coverage_score:
            decision["status"] = "rejected"
            decision["amount"] = 0
            # High confidence for clear rejections
            decision["confidence"] = min(base_confidence * 1.1, 0.90)
        else:
            decision["status"] = "review_required"
            decision["amount"] = 0
            decision["confidence"] = 0.6
        
        # Build justification
        if decision["status"] == "approved":
            decision["justification"] = f"Treatment appears to be covered under policy terms. Estimated coverage: ₹{decision['amount']:,.0f}"
        elif decision["status"] == "conditional_approval":
            decision["justification"] = f"Treatment may be covered with conditions. Risk factors: {', '.join(risk_factors)}"
        elif decision["status"] == "rejected":
            decision["justification"] = "Treatment appears to be excluded or not covered under current policy terms"
        else:
            decision["justification"] = "Manual review required - insufficient information for automated decision"
        
        # Add recommendations
        if decision["status"] in ["approved", "conditional_approval"]:
            decision["recommendations"] = [
                "Submit required medical documentation",
                "Ensure treatment is from network providers",
                "Verify coverage limits before proceeding"
            ]
        else:
            decision["recommendations"] = [
                "Review policy terms and conditions",
                "Consider contacting customer service for clarification",
                "Explore supplementary coverage options"
            ]
        
        decision["risk_factors"] = risk_factors
        decision["clauses_used"] = relevant_clauses[:3]  # Top 3 clauses
        
        return decision
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process queries"""
        start_time = datetime.now()
        
        try:
            # Input validation
            if not query or not isinstance(query, str):
                return {
                    "query": str(query) if query else "",
                    "decision": "error",
                    "amount": 0.0,
                    "confidence": 0.0,
                    "justification": "Invalid query: Query must be a non-empty string",
                    "detailed_response": "Error: Please provide a valid query string",
                    "risk_factors": [],
                    "recommendations": ["Please provide a valid query"],
                    "clauses_mapping": [],
                    "parsed_query": {},
                    "processing_metadata": {
                        "total_clauses_searched": len(self.clause_database),
                        "relevant_clauses_found": 0,
                        "processing_time": 0.0,
                        "error": "Invalid input"
                    }
                }
            
            query = query.strip()
            if not query:
                return {
                    "query": "",
                    "decision": "error",
                    "amount": 0.0,
                    "confidence": 0.0,
                    "justification": "Empty query provided",
                    "detailed_response": "Error: Query cannot be empty",
                    "risk_factors": [],
                    "recommendations": ["Please provide a valid query"],
                    "clauses_mapping": [],
                    "parsed_query": {},
                    "processing_metadata": {
                        "total_clauses_searched": len(self.clause_database),
                        "relevant_clauses_found": 0,
                        "processing_time": 0.0,
                        "error": "Empty query"
                    }
                }
            
            # Parse query
            parsed_query = self.parse_query_advanced(query)
            
            # Search clauses
            relevant_clauses = self.search_relevant_clauses_hybrid(parsed_query)
            
            # Evaluate decision
            decision = self.evaluate_decision_advanced(parsed_query, relevant_clauses)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats
            self.stats['queries_processed'] += 1
            
            # Format response
            response = {
                "query": query,
                "decision": decision["status"],
                "amount": decision["amount"],
                "confidence": decision["confidence"],
                "justification": decision["justification"],
                "detailed_response": f"Analysis completed for: {parsed_query.get('procedure', 'general treatment')}",
                "risk_factors": decision["risk_factors"],
                "recommendations": decision["recommendations"],
                "clauses_mapping": [
                    {
                        "clause_text": clause["text"][:200] + "...",
                        "document": clause["document"],
                        "clause_type": clause.get("clause_type", "unknown"),
                        "similarity_score": clause.get("similarity_score", 0.0),
                        "search_method": clause.get("search_method", "tfidf")
                    }
                    for clause in decision["clauses_used"]
                ],
                "parsed_query": parsed_query,
                "processing_metadata": {
                    "processing_time_seconds": processing_time,
                    "total_clauses_searched": len(self.clause_database),
                    "relevant_clauses_found": len(relevant_clauses),
                    "search_methods_used": ["tfidf"],
                    "model_confidence": parsed_query.get("confidence", 0.0)
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "decision": "error",
                "amount": 0.0,
                "confidence": 0.0,
                "justification": f"Error processing query: {str(e)}",
                "detailed_response": None,
                "risk_factors": [],
                "recommendations": ["Please try again with a different query"],
                "clauses_mapping": [],
                "parsed_query": {"raw_query": query},
                "processing_metadata": {
                    "processing_time_seconds": 0.0,
                    "total_clauses_searched": 0,
                    "relevant_clauses_found": 0,
                    "search_methods_used": [],
                    "error": str(e)
                }
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "documents_loaded": len(self.documents),
            "total_clauses": len(self.clause_database),
            "queries_processed": self.stats['queries_processed'],
            "models_loaded": {
                "tfidf_vectorizer": self.tfidf_vectorizer is not None,
                "lightweight_mode": True
            },
            "device": "cpu",
            "system_status": "healthy" if self.documents else "no_documents_loaded"
        }
    
    def _add_sample_clauses(self):
        """Add sample insurance clauses for demonstration"""
        sample_clauses = [
            {
                'text': 'Surgical procedures including knee surgery, hip surgery, and cardiac surgery are covered under this policy up to the sum insured amount.',
                'document': 'Sample_Policy.pdf',
                'clause_type': 'coverage_positive'
            },
            {
                'text': 'All emergency medical treatments including ambulance charges are covered without any waiting period restrictions.',
                'document': 'Sample_Policy.pdf', 
                'clause_type': 'coverage_positive'
            },
            {
                'text': 'Dental treatments including dental surgery and root canal procedures are covered up to Rs. 25,000 per policy year.',
                'document': 'Sample_Policy.pdf',
                'clause_type': 'coverage_positive'
            },
            {
                'text': 'Eye surgery including cataract surgery and LASIK procedures are covered under the policy benefits.',
                'document': 'Sample_Policy.pdf',
                'clause_type': 'coverage_positive'
            },
            {
                'text': 'Heart surgery and cardiac procedures including bypass surgery and angioplasty are covered up to Rs. 3,00,000.',
                'document': 'Sample_Policy.pdf',
                'clause_type': 'coverage_positive'
            },
            {
                'text': 'Pre-existing medical conditions are covered after a waiting period of 2 years from policy inception.',
                'document': 'Sample_Policy.pdf',
                'clause_type': 'general_terms'
            },
            {
                'text': 'Cosmetic surgery and treatments undertaken for aesthetic purposes are excluded from coverage.',
                'document': 'Sample_Policy.pdf',
                'clause_type': 'coverage_negative'
            },
            {
                'text': 'Maximum coverage amount for surgical procedures is Rs. 1,00,000 per incident unless specified otherwise.',
                'document': 'Sample_Policy.pdf',
                'clause_type': 'financial_terms'
            }
        ]
        
        self.clause_database.extend(sample_clauses)
        
        # Rebuild TF-IDF with sample data
        if self.clause_database:
            clause_texts = [clause['text'] for clause in self.clause_database]
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clause_texts)
            logger.info(f"Added {len(sample_clauses)} sample clauses and rebuilt search index")

# For compatibility with existing API
OptimizedDocumentProcessor = LightweightDocumentProcessor

# Example usage
if __name__ == "__main__":
    processor = LightweightDocumentProcessor()
    
    # Load documents
    dataset_path = "d:/Bajaj_Finserv/Datasets"
    processor.load_documents(dataset_path)
    
    # Test queries
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month policy",
        "Female patient needs heart surgery, age 55, Mumbai location",
        "Dental treatment for 30-year-old, covered under policy?",
        "Emergency surgery required, what's the coverage limit?",
        "Cataract surgery eligibility for senior citizen"
    ]
    
    print("=" * 80)
    print("LIGHTWEIGHT LLM DOCUMENT PROCESSOR - TEST RESULTS")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        
        result = processor.process_query(query)
        
        print(f"Decision: {result['decision'].upper()}")
        print(f"Amount: ₹{result['amount']:,.2f}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Justification: {result['justification']}")
        
        if result.get('recommendations'):
            print(f"Recommendations: {', '.join(result['recommendations'][:2])}")
        
        print(f"Processing Time: {result['processing_metadata']['processing_time_seconds']:.3f}s")
        print("-" * 60)
    
    # Print system stats
    stats = processor.get_system_stats()
    print(f"\nSystem Statistics:")
    print(json.dumps(stats, indent=2))
