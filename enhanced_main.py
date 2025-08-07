import os
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

# Core libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import PyPDF2
try:
    import fitz  # PyMuPDF for better PDF extraction
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Setup logging EARLY
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now we can use logger
if not HAS_PYMUPDF:
    logger.warning("PyMuPDF not found. Using PyPDF2 only for PDF extraction.")

from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Transformers and LLM
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForQuestionAnswering,
    pipeline, BertTokenizer, BertModel, AutoModelForSequenceClassification
)

# OpenAI GPT-4 (Hackathon recommended)
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI not found. Install: pip install openai")

# Pinecone Vector DB (Hackathon recommended)
try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    logger.warning("Pinecone not found. Install: pip install pinecone-client")

import torch
import torch
import torch.nn.functional as F

class OptimizedDocumentProcessor:
    """
    Optimized LLM-based document processing system for insurance policy analysis
    Features:
    - Advanced PDF text extraction
    - Multi-model ensemble approach
    - Semantic search with FAISS
    - Risk assessment
    - Confidence scoring
    - Real-time processing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the optimized document processor"""
        logger.info("Initializing Optimized Document Processor...")
        
        # Configuration
        self.config = config or self._default_config()
        
        # Download required NLTK data
        self._setup_nltk()
        
        # Initialize device with detailed GPU info
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            logger.info(f"🔥 CUDA Version: {torch.version.cuda}")
        else:
            self.device = torch.device('cpu')
            logger.warning("⚠️ GPU not available, using CPU")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Initialize document storage
        self.documents = {}
        self.document_metadata = {}
        self.clause_database = []
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Insurance domain knowledge
        self._initialize_domain_knowledge()
        
        # Processing statistics
        self.processing_stats = {
            'queries_processed': 0,
            'documents_loaded': 0,
            'total_clauses': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info("Document processor initialized successfully!")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration settings for hackathon demo"""
        return {
            # GPT-4 Configuration (Hackathon Recommended)
            'llm_model': 'gpt-4',
            'llm_api_key': os.getenv('OPENAI_API_KEY', ''),
            'use_gpt4': True,
            
            # Fallback models for demo resilience
            'qa_model': 'deepset/roberta-base-squad2',
            'sentence_model': 'all-MiniLM-L6-v2',
            'classification_model': 'microsoft/DialoGPT-medium',
            
            # Pinecone Configuration (Hackathon Recommended)
            'pinecone_api_key': os.getenv('PINECONE_API_KEY', ''),
            'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws'),
            'pinecone_index_name': 'insurance-clauses',
            'use_pinecone': True,
            
            # Processing settings
            'max_sequence_length': 4096,  # GPT-4 supports longer sequences
            'batch_size': 8,  # Smaller batches for GPT-4 API calls
            'similarity_threshold': 0.8,  # Higher threshold for GPT-4
            'confidence_threshold': 0.7,
            'max_clauses_per_search': 15,
            'enable_gpu': True,
            'cache_embeddings': True,
            
            # API settings
            'max_tokens': 1500,
            'temperature': 0.1,  # Low temperature for consistent responses
            'top_p': 0.9
        }
    
    def _setup_nltk(self):
        """Setup NLTK dependencies"""
        try:
            required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']
            for data in required_data:
                nltk.download(data, quiet=True)
        except Exception as e:
            logger.warning(f"NLTK setup warning: {e}")
    
    def _initialize_models(self):
        """Initialize models with GPT-4 as primary LLM (Hackathon Spec)"""
        try:
            # Initialize GPT-4 (Primary LLM for Hackathon)
            if HAS_OPENAI and self.config['llm_api_key']:
                logger.info("Initializing GPT-4 (Primary LLM)...")
                self.openai_client = OpenAI(api_key=self.config['llm_api_key'])
                self.use_gpt4 = True
                logger.info("✅ GPT-4 initialized successfully!")
            else:
                logger.warning("⚠️ GPT-4 not available. Using fallback models.")
                self.use_gpt4 = False
            
            # Initialize Pinecone (Primary Vector DB for Hackathon)
            if HAS_PINECONE and self.config['pinecone_api_key']:
                logger.info("Initializing Pinecone Vector Database...")
                self.pc = Pinecone(api_key=self.config['pinecone_api_key'])
                
                # Create index if it doesn't exist
                index_name = self.config['pinecone_index_name']
                if index_name not in self.pc.list_indexes().names():
                    self.pc.create_index(
                        name=index_name,
                        dimension=384,  # all-MiniLM-L6-v2 dimension
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                
                self.pinecone_index = self.pc.Index(index_name)
                self.use_pinecone = True
                logger.info("✅ Pinecone initialized successfully!")
            else:
                logger.warning("⚠️ Pinecone not available. Using FAISS fallback.")
                self.use_pinecone = False
            
            # Fallback Models (for demo resilience)
            logger.info("Loading fallback models for demo resilience...")
            
            # Question Answering pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.config['qa_model'],
                tokenizer=self.config['qa_model'],
                device=0 if torch.cuda.is_available() and self.config['enable_gpu'] else -1,
                max_answer_len=256,
                max_question_len=64
            )
            
            # Sentence transformer for embeddings
            self.sentence_model = SentenceTransformer(
                self.config['sentence_model'],
                device=self.device
            )
            
            # NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            logger.info("🚀 All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            # Continue with fallback models
            self.use_gpt4 = False
            self.use_pinecone = False
    
    def _initialize_domain_knowledge(self):
        """Initialize insurance domain-specific knowledge"""
        self.insurance_knowledge = {
            'procedures': {
                'surgical': [
                    'knee surgery', 'hip surgery', 'heart surgery', 'cardiac surgery', 
                    'eye surgery', 'cataract surgery', 'lasik', 'dental surgery',
                    'orthopedic surgery', 'plastic surgery', 'brain surgery', 'neurosurgery',
                    'appendectomy', 'gallbladder surgery', 'hernia surgery', 'bypass surgery',
                    'angioplasty', 'transplant surgery', 'spine surgery', 'shoulder surgery'
                ],
                'medical': [
                    'chemotherapy', 'radiation therapy', 'dialysis', 'physical therapy',
                    'occupational therapy', 'mental health treatment', 'rehabilitation'
                ],
                'emergency': [
                    'emergency surgery', 'emergency treatment', 'ambulance service',
                    'emergency room visit', 'urgent care', 'trauma care'
                ]
            },
            'conditions': {
                'chronic': [
                    'diabetes', 'hypertension', 'heart disease', 'asthma', 'arthritis',
                    'chronic kidney disease', 'copd', 'epilepsy', 'multiple sclerosis'
                ],
                'acute': [
                    'heart attack', 'stroke', 'appendicitis', 'pneumonia', 'broken bone',
                    'injury', 'accident', 'emergency condition'
                ],
                'mental_health': [
                    'depression', 'anxiety', 'bipolar disorder', 'schizophrenia',
                    'ptsd', 'eating disorder', 'substance abuse'
                ]
            },
            'coverage_terms': {
                'positive': ['covered', 'eligible', 'included', 'benefits', 'reimburse', 'pay'],
                'negative': ['excluded', 'not covered', 'denied', 'rejected', 'limitation'],
                'financial': ['deductible', 'copay', 'coinsurance', 'premium', 'out-of-pocket']
            },
            'demographic_factors': {
                'age_groups': {
                    'child': (0, 17),
                    'adult': (18, 64),
                    'senior': (65, 120)
                },
                'risk_factors': {
                    'high_risk_age': [0, 5, 65, 120],
                    'high_risk_procedures': ['heart surgery', 'brain surgery', 'transplant'],
                    'high_risk_conditions': ['cancer', 'heart disease', 'stroke']
                }
            }
        }
    
    def load_documents(self, folder_path: str) -> None:
        """Load and process documents with enhanced extraction"""
        logger.info(f"Loading documents from: {folder_path}")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError("No PDF files found in the dataset folder")
        
        # Process documents in parallel for better performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for filename in pdf_files:
                file_path = os.path.join(folder_path, filename)
                future = executor.submit(self._process_single_document, file_path, filename)
                futures[future] = filename
            
            for future in futures:
                try:
                    filename = futures[future]
                    document_data = future.result()
                    if document_data:
                        self.documents[filename] = document_data['text']
                        self.document_metadata[filename] = document_data['metadata']
                        logger.info(f"Successfully loaded: {filename}")
                    else:
                        logger.warning(f"No content extracted from: {filename}")
                except Exception as e:
                    logger.error(f"Error loading {futures[future]}: {e}")
        
        if self.documents:
            self._build_enhanced_knowledge_base()
            self.processing_stats['documents_loaded'] = len(self.documents)
            logger.info(f"Loaded {len(self.documents)} documents successfully")
        else:
            raise ValueError("No documents could be loaded successfully")
    
    def _process_single_document(self, file_path: str, filename: str) -> Optional[Dict[str, Any]]:
        """Process a single document with enhanced extraction"""
        logger.info(f"Processing document: {filename}")
        
        try:
            text = ""
            
            # Try PyMuPDF first (better extraction) if available
            if HAS_PYMUPDF:
                try:
                    text = self._extract_text_pymupdf(file_path)
                    logger.info(f"PyMuPDF successfully extracted {len(text)} characters from {filename}")
                except Exception as e:
                    logger.warning(f"PyMuPDF failed for {filename}: {e}")
                    # Fallback to PyPDF2
                    try:
                        text = self._extract_text_pypdf2(file_path)
                        logger.info(f"PyPDF2 fallback extracted {len(text)} characters from {filename}")
                    except Exception as e2:
                        logger.error(f"Both PyMuPDF and PyPDF2 failed for {filename}: PyMuPDF={e}, PyPDF2={e2}")
                        return None
            else:
                # Use PyPDF2 directly
                try:
                    text = self._extract_text_pypdf2(file_path)
                    logger.info(f"PyPDF2 extracted {len(text)} characters from {filename}")
                except Exception as e:
                    logger.error(f"PyPDF2 failed for {filename}: {e}")
                    return None
            
            if not text or not text.strip():
                logger.warning(f"No text content extracted from {filename}")
                return None
            
            # Extract metadata
            metadata = self._extract_document_metadata(text, filename)
            
            logger.info(f"Successfully processed {filename}: {len(text)} chars, {metadata.get('page_count', 0)} pages")
            
            return {
                'text': text,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Unexpected error processing {filename}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better quality)"""
        if not HAS_PYMUPDF:
            raise Exception("PyMuPDF not available")
        
        text = ""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            logger.debug(f"PyMuPDF opened {pdf_path}: {page_count} pages")
            
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text and page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1} from {pdf_path}: {e}")
                    continue
                    
            logger.debug(f"PyMuPDF extracted {len(text)} characters from {page_count} pages")
                    
        except Exception as e:
            logger.error(f"PyMuPDF failed to open {pdf_path}: {e}")
            raise Exception(f"PyMuPDF extraction failed: {e}")
        finally:
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
        return text
    
    def _extract_text_pypdf2(self, pdf_path: str) -> str:
        """Fallback extraction using PyPDF2"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {pdf_path}: {e}")
                        continue
                        
                logger.info(f"PyPDF2 extracted text from {pdf_path}: {len(text)} characters, {total_pages} pages")
                
        except Exception as e:
            logger.error(f"PyPDF2 failed to process {pdf_path}: {e}")
            raise Exception(f"PyPDF2 extraction failed: {e}")
            
        return text
    
    def _extract_document_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from document content"""
        # Input validation
        if not text or not isinstance(text, str):
            text = ""
            
        metadata = {
            'filename': filename,
            'word_count': len(text.split()),
            'char_count': len(text),
            'page_count': text.count('--- Page'),
            'procedures_mentioned': [],
            'conditions_mentioned': [],
            'coverage_terms': [],
            'document_type': self._classify_document_type(text),
            'extracted_at': datetime.now().isoformat()
        }
        
        # Extract mentioned procedures
        text_lower = text.lower()
        for category, procedures in self.insurance_knowledge['procedures'].items():
            for procedure in procedures:
                if procedure in text_lower:
                    metadata['procedures_mentioned'].append({
                        'procedure': procedure,
                        'category': category
                    })
        
        # Extract mentioned conditions
        for category, conditions in self.insurance_knowledge['conditions'].items():
            for condition in conditions:
                if condition in text_lower:
                    metadata['conditions_mentioned'].append({
                        'condition': condition,
                        'category': category
                    })
        
        return metadata
    
    def _classify_document_type(self, text: str) -> str:
        """Classify the type of insurance document"""
        if not text or not isinstance(text, str):
            return 'unknown_document'
            
        text_lower = text.lower()
        
        if 'policy' in text_lower and 'terms' in text_lower:
            return 'policy_document'
        elif 'claim' in text_lower:
            return 'claims_document'
        elif 'benefit' in text_lower and 'schedule' in text_lower:
            return 'benefits_schedule'
        elif 'premium' in text_lower:
            return 'premium_document'
        else:
            return 'general_insurance_document'
    
    def _build_enhanced_knowledge_base(self):
        """Build enhanced knowledge base with multiple indexing strategies"""
        logger.info("Building enhanced knowledge base...")
        
        # Extract clauses from all documents
        all_clauses = []
        for doc_name, doc_content in self.documents.items():
            clauses = self._extract_clauses_advanced(doc_content, doc_name)
            all_clauses.extend(clauses)
        
        self.clause_database = all_clauses
        self.processing_stats['total_clauses'] = len(all_clauses)
        
        if all_clauses:
            # Build sentence embeddings for semantic search
            clause_texts = [clause['text'] for clause in all_clauses]
            
            logger.info("Creating sentence embeddings...")
            embeddings = self.sentence_model.encode(
                clause_texts,
                batch_size=self.config['batch_size'],
                show_progress_bar=True
            )
            
            # Build FAISS index for fast similarity search
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Build TF-IDF index for keyword-based search
            logger.info("Building TF-IDF index...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clause_texts)
            
            logger.info(f"Knowledge base built with {len(all_clauses)} clauses")
    
    def _extract_clauses_advanced(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """Extract clauses with advanced segmentation - IMPROVED VERSION"""
        clauses = []
        
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            logger.info(f"Processing {len(sentences)} sentences from {doc_name}")
            
            # Group sentences into meaningful clauses
            current_clause = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 5:  # Skip very short sentences
                    continue
                
                current_clause.append(sentence)
                
                # End clause on certain patterns or when we have enough content
                should_end_clause = (
                    len(current_clause) >= 2 and  # Reduced from 3 to get more clauses
                    (sentence.endswith('.') or sentence.endswith(';') or 
                     sentence.endswith('!') or sentence.endswith('?') or
                     any(term in sentence.lower() for term in ['however', 'furthermore', 'additionally', 'moreover', 'therefore']))
                )
                
                # Also end clause if it gets too long
                if len(' '.join(current_clause).split()) > 100:  # Max clause length
                    should_end_clause = True
                
                if should_end_clause:
                    clause_text = ' '.join(current_clause).strip()
                    if len(clause_text.split()) >= 8:  # Reduced minimum length for more clauses
                        try:
                            clause_info = {
                                'text': clause_text,
                                'document': doc_name,
                                'clause_type': self._classify_clause_advanced(clause_text),
                                'entities': self._extract_entities_advanced(clause_text),
                                'importance_score': self._calculate_importance_score(clause_text),
                                'sentiment': self._analyze_sentiment(clause_text)
                            }
                            clauses.append(clause_info)
                        except Exception as e:
                            logger.warning(f"Error processing clause: {e}")
                    
                    current_clause = []
            
            # Add remaining clause if any
            if current_clause:
                clause_text = ' '.join(current_clause).strip()
                if len(clause_text.split()) >= 8:
                    try:
                        clause_info = {
                            'text': clause_text,
                            'document': doc_name,
                            'clause_type': self._classify_clause_advanced(clause_text),
                            'entities': self._extract_entities_advanced(clause_text),
                            'importance_score': self._calculate_importance_score(clause_text),
                            'sentiment': self._analyze_sentiment(clause_text)
                        }
                        clauses.append(clause_info)
                    except Exception as e:
                        logger.warning(f"Error processing final clause: {e}")
            
            logger.info(f"Extracted {len(clauses)} clauses from {doc_name}")
            
        except Exception as e:
            logger.error(f"Error in clause extraction for {doc_name}: {e}")
        
        return clauses
    
    def _classify_clause_advanced(self, text: str) -> str:
        """Advanced clause classification"""
        if not text or not isinstance(text, str):
            return 'unknown'
            
        text_lower = text.lower()
        
        # Check for specific patterns
        if any(word in text_lower for word in ['covered', 'eligible', 'included', 'benefits']):
            return 'coverage_positive'
        elif any(word in text_lower for word in ['excluded', 'not covered', 'denied', 'limitations']):
            return 'coverage_negative'
        elif any(word in text_lower for word in ['premium', 'cost', 'amount', 'fee', 'payment']):
            return 'financial_terms'
        elif any(word in text_lower for word in ['waiting period', 'effective date', 'duration']):
            return 'temporal_conditions'
        elif any(word in text_lower for word in ['deductible', 'copay', 'coinsurance']):
            return 'cost_sharing'
        elif any(word in text_lower for word in ['pre-existing', 'medical history', 'condition']):
            return 'medical_conditions'
        elif any(word in text_lower for word in ['procedure', 'treatment', 'surgery', 'therapy']):
            return 'medical_procedures'
        else:
            return 'general_terms'
    
    def _extract_entities_advanced(self, text: str) -> Dict[str, List[str]]:
        """Advanced entity extraction - FIXED VERSION"""
        entities = {
            'procedures': [],
            'conditions': [],
            'amounts': [],
            'percentages': [],
            'dates': [],
            'locations': [],
            'age_ranges': [],
            'medical_terms': []
        }
        
        if not text or not isinstance(text, str):
            return entities
            
        text_lower = text.lower()
        
        # Extract procedures - FIX: Handle nested structure correctly with error handling
        try:
            if hasattr(self, 'insurance_knowledge') and 'procedures' in self.insurance_knowledge:
                for category, procedures in self.insurance_knowledge['procedures'].items():
                    for procedure in procedures:
                        if procedure in text_lower:
                            entities['procedures'].append({
                                'term': procedure,
                                'category': category
                            })
        except Exception as e:
            logger.warning(f"Error extracting procedures: {e}")
        
        # Extract conditions - FIX: Handle nested structure correctly with error handling
        try:
            if hasattr(self, 'insurance_knowledge') and 'conditions' in self.insurance_knowledge:
                for category, conditions in self.insurance_knowledge['conditions'].items():
                    for condition in conditions:
                        if condition in text_lower:
                            entities['conditions'].append({
                                'term': condition,
                                'category': category
                            })
        except Exception as e:
            logger.warning(f"Error extracting conditions: {e}")
        
        # Extract monetary amounts
        money_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\s*dollars?',
            r'[\d,]+\s*rupees?',
            r'rs\.?\s*[\d,]+',
            r'inr\s*[\d,]+'
        ]
        for pattern in money_patterns:
            amounts = re.findall(pattern, text, re.IGNORECASE)
            entities['amounts'].extend(amounts)
        
        # Extract percentages
        percentage_pattern = r'\d+\.?\d*\s*%'
        percentages = re.findall(percentage_pattern, text)
        entities['percentages'].extend(percentages)
        
        # Extract age ranges
        age_patterns = [
            r'(\d+)\s*to\s*(\d+)\s*years?',
            r'age\s*(\d+)\s*to\s*(\d+)',
            r'(\d+)-(\d+)\s*years?'
        ]
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['age_ranges'].extend([f"{m[0]}-{m[1]}" for m in matches])
        
        return entities
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score for a clause"""
        if not text or not isinstance(text, str):
            return 0.0
            
        score = 0.0
        text_lower = text.lower()
        
        # Key term weights
        key_terms = {
            'covered': 1.0,
            'excluded': 1.0,
            'eligible': 0.8,
            'benefits': 0.7,
            'premium': 0.6,
            'deductible': 0.6,
            'limitation': 0.8,
            'maximum': 0.7,
            'minimum': 0.5
        }
        
        for term, weight in key_terms.items():
            if term in text_lower:
                score += weight
        
        # Length factor (longer clauses might be more detailed)
        word_count = len(text.split())
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.1
        
        return min(score, 3.0)  # Cap at 3.0
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of clause (positive/negative/neutral)"""
        if not text or not isinstance(text, str):
            return 'neutral'
            
        text_lower = text.lower()
        
        positive_words = ['covered', 'eligible', 'included', 'benefits', 'approved']
        negative_words = ['excluded', 'denied', 'rejected', 'not covered', 'limitations']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def parse_query_advanced(self, query: str) -> Dict[str, Any]:
        """Advanced query parsing with multiple extraction strategies"""
        logger.info(f"Parsing query: {query}")
        
        # Input validation
        if not query or not isinstance(query, str):
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "condition": None,
                "location": None,
                "policy_duration": None,
                "amount_requested": None,
                "urgency": None,
                "intent": None,
                "entities": {},
                "confidence": 0.0,
                "raw_query": str(query) if query else "",
                "error": "Invalid query input"
            }
        
        parsed_data = {
            "age": None,
            "gender": None,
            "procedure": None,
            "condition": None,
            "location": None,
            "policy_duration": None,
            "amount_requested": None,
            "urgency": None,
            "intent": None,
            "entities": {},
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
        elif re.search(r'\bfemale\b', query_lower):
            parsed_data["gender"] = "female"
        elif re.search(r'\bwoman\b|\bgirl\b|\bher\b|\bshe\b', query_lower):
            parsed_data["gender"] = "female"
        elif re.search(r'\bman\b|\bboy\b|\bhis\b|\bhe\b', query_lower):
            parsed_data["gender"] = "male"
        
        # Extract procedures
        for category, procedures in self.insurance_knowledge['procedures'].items():
            for procedure in procedures:
                if procedure and isinstance(procedure, str) and procedure in query_lower:
                    parsed_data["procedure"] = procedure
                    break
            if parsed_data["procedure"]:
                break
        
        # Extract conditions
        for category, conditions in self.insurance_knowledge['conditions'].items():
            for condition in conditions:
                if condition and isinstance(condition, str) and condition in query_lower:
                    parsed_data["condition"] = condition
                    break
            if parsed_data["condition"]:
                break
        
        # Extract location
        indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'pune', 'kolkata',
            'chennai', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur'
        ]
        
        for city in indian_cities:
            if city in query_lower:
                parsed_data["location"] = city
                break
        
        # Extract policy duration
        duration_patterns = [
            r'(\d+)[-\s]?month[s]?',
            r'(\d+)[-\s]?year[s]?',
            r'(\d+)[-\s]?day[s]?'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parsed_data["policy_duration"] = match.group(0)
                break
        
        # Extract monetary amounts
        money_patterns = [
            r'\$[\d,]+\.?\d*',
            r'rs\.?\s*[\d,]+',
            r'[\d,]+\s*rupees?'
        ]
        
        for pattern in money_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parsed_data["amount_requested"] = match.group(0)
                break
        
        # Detect urgency
        urgency_keywords = ['emergency', 'urgent', 'immediate', 'asap', 'critical']
        if any(keyword in query_lower for keyword in urgency_keywords):
            parsed_data["urgency"] = "high"
        else:
            parsed_data["urgency"] = "normal"
        
        # Detect intent
        if any(word in query_lower for word in ['covered', 'eligible', 'cover']):
            parsed_data["intent"] = "coverage_inquiry"
        elif any(word in query_lower for word in ['cost', 'amount', 'fee', 'premium']):
            parsed_data["intent"] = "cost_inquiry"
        elif any(word in query_lower for word in ['claim', 'reimburse', 'refund']):
            parsed_data["intent"] = "claim_inquiry"
        else:
            parsed_data["intent"] = "general_inquiry"
        
        # Extract advanced entities
        parsed_data["entities"] = self._extract_entities_advanced(query)
        
        # Calculate parsing confidence
        confidence_factors = [
            parsed_data["age"] is not None,
            parsed_data["gender"] is not None,
            parsed_data["procedure"] is not None or parsed_data["condition"] is not None,
            parsed_data["intent"] != "general_inquiry"
        ]
        
        parsed_data["confidence"] = sum(confidence_factors) / len(confidence_factors)
        
        return parsed_data
    
    def search_relevant_clauses_hybrid(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword-based approaches"""
        if not self.clause_database:
            return []
        
        query_text = parsed_query["raw_query"]
        relevant_clauses = []
        
        # 1. Semantic search using FAISS
        if self.faiss_index is not None:
            semantic_clauses = self._semantic_search(query_text)
            relevant_clauses.extend(semantic_clauses)
        
        # 2. Keyword-based search using TF-IDF
        if self.tfidf_vectorizer is not None:
            keyword_clauses = self._keyword_search(query_text)
            relevant_clauses.extend(keyword_clauses)
        
        # 3. Entity-based search
        entity_clauses = self._entity_based_search(parsed_query)
        relevant_clauses.extend(entity_clauses)
        
        # Remove duplicates and sort by relevance
        seen_clauses = set()
        unique_clauses = []
        
        for clause in relevant_clauses:
            clause_id = clause['text'][:100]  # Use first 100 chars as ID
            if clause_id not in seen_clauses:
                seen_clauses.add(clause_id)
                unique_clauses.append(clause)
        
        # Sort by similarity score (descending)
        unique_clauses.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Return top clauses
        return unique_clauses[:self.config['max_clauses_per_search']]
    
    def _semantic_search(self, query_text: str) -> List[Dict[str, Any]]:
        """Semantic search using sentence embeddings"""
        if self.faiss_index is None:
            return []
        
        # Encode query
        query_embedding = self.sentence_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        k = min(20, len(self.clause_database))  # Search top 20
        similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.clause_database) and similarity > self.config['similarity_threshold']:
                clause = self.clause_database[idx].copy()
                clause['similarity_score'] = float(similarity)
                clause['search_method'] = 'semantic'
                results.append(clause)
        
        return results
    
    def _keyword_search(self, query_text: str) -> List[Dict[str, Any]]:
        """Keyword-based search using TF-IDF"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top similar clauses
        top_indices = similarities.argsort()[-20:][::-1]  # Top 20
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                clause = self.clause_database[idx].copy()
                clause['similarity_score'] = float(similarities[idx])
                clause['search_method'] = 'keyword'
                results.append(clause)
        
        return results
    
    def _entity_based_search(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Entity-based search for specific medical terms"""
        results = []
        
        search_terms = []
        if parsed_query.get('procedure'):
            search_terms.append(parsed_query['procedure'])
        if parsed_query.get('condition'):
            search_terms.append(parsed_query['condition'])
        
        if not search_terms:
            return results
        
        for clause in self.clause_database:
            clause_text = clause.get('text', '')
            if not clause_text or not isinstance(clause_text, str):
                continue
                
            clause_text_lower = clause_text.lower()
            match_count = 0
            
            for term in search_terms:
                if term and isinstance(term, str) and term.lower() in clause_text_lower:
                    match_count += 1
            
            if match_count > 0:
                clause_copy = clause.copy()
                clause_copy['similarity_score'] = match_count / len(search_terms)
                clause_copy['search_method'] = 'entity'
                results.append(clause_copy)
        
        return results
    
    def evaluate_decision_advanced(self, parsed_query: Dict[str, Any], relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Advanced decision evaluation with multiple factors"""
        decision = {
            "status": "pending",
            "amount": 0.0,
            "confidence": 0.0,
            "justification": "Insufficient information for decision",
            "risk_factors": [],
            "recommendations": [],
            "clauses_used": [],
            "decision_factors": {}
        }
        
        if not relevant_clauses:
            decision["status"] = "information_needed"
            decision["justification"] = "No relevant policy clauses found"
            return decision
        
        # Analyze coverage
        coverage_analysis = self._analyze_coverage(parsed_query, relevant_clauses)
        
        # Assess risk factors
        risk_assessment = self._assess_risk_factors(parsed_query)
        
        # Calculate coverage amount
        amount_calculation = self._calculate_coverage_amount(parsed_query, relevant_clauses)
        
        # Determine final decision
        if coverage_analysis['is_covered']:
            if risk_assessment['risk_level'] == 'low':
                decision["status"] = "approved"
                decision["amount"] = amount_calculation['covered_amount']
                decision["confidence"] = min(coverage_analysis['confidence'] * 0.9, 0.95)
            elif risk_assessment['risk_level'] == 'medium':
                decision["status"] = "conditional_approval"
                decision["amount"] = amount_calculation['covered_amount'] * 0.8
                decision["confidence"] = min(coverage_analysis['confidence'] * 0.7, 0.85)
            else:  # high risk
                decision["status"] = "review_required"
                decision["amount"] = 0.0
                decision["confidence"] = 0.6
        else:
            decision["status"] = "rejected"
            decision["amount"] = 0.0
            decision["confidence"] = coverage_analysis['confidence']
        
        # Build justification
        decision["justification"] = self._build_justification(
            parsed_query, coverage_analysis, risk_assessment, amount_calculation
        )
        
        # Set additional fields
        decision["risk_factors"] = risk_assessment['factors']
        decision["recommendations"] = self._generate_recommendations(parsed_query, decision["status"])
        decision["clauses_used"] = relevant_clauses[:5]  # Top 5 clauses
        decision["decision_factors"] = {
            "coverage_analysis": coverage_analysis,
            "risk_assessment": risk_assessment,
            "amount_calculation": amount_calculation
        }
        
        return decision
    
    def _analyze_coverage(self, parsed_query: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if the query is covered based on clauses"""
        coverage_indicators = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        confidence_score = 0.0
        relevant_clauses = 0
        
        for clause in clauses:
            clause_text = clause.get('text', '')
            if not clause_text or not isinstance(clause_text, str):
                continue
                
            clause_text = clause_text.lower()
            
            # Check for coverage keywords
            positive_terms = ['covered', 'eligible', 'included', 'benefits', 'reimburse']
            negative_terms = ['excluded', 'not covered', 'denied', 'limitation', 'except']
            
            pos_count = sum(1 for term in positive_terms if term in clause_text)
            neg_count = sum(1 for term in negative_terms if term in clause_text)
            
            if pos_count > neg_count:
                coverage_indicators['positive'] += 1
                confidence_score += clause.get('similarity_score', 0.5)
            elif neg_count > pos_count:
                coverage_indicators['negative'] += 1
                confidence_score += clause.get('similarity_score', 0.5)
            else:
                coverage_indicators['neutral'] += 1
            
            relevant_clauses += 1
        
        # Determine coverage
        is_covered = coverage_indicators['positive'] > coverage_indicators['negative']
        
        # Calculate confidence with multiple factors
        total_indicators = sum(coverage_indicators.values())
        if total_indicators > 0 and relevant_clauses > 0:
            # Base confidence from similarity scores
            avg_similarity = confidence_score / relevant_clauses
            
            # Coverage strength factor
            coverage_strength = coverage_indicators['positive'] / total_indicators if total_indicators > 0 else 0
            
            # Clause relevance factor  
            relevance_factor = min(relevant_clauses / 3, 1.0)  # Normalize to max 3 clauses
            
            # Combined confidence (minimum 15% for any analysis)
            confidence = max(
                (avg_similarity * 0.4 + coverage_strength * 0.4 + relevance_factor * 0.2),
                0.15
            )
        else:
            confidence = 0.1  # Low but not zero confidence
        
        return {
            'is_covered': is_covered,
            'confidence': min(confidence, 1.0),
            'indicators': coverage_indicators,
            'reasoning': f"Found {coverage_indicators['positive']} positive and {coverage_indicators['negative']} negative coverage indicators"
        }
    
    def _assess_risk_factors(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk factors for the query"""
        risk_factors = []
        risk_score = 0.0
        
        # Age-based risk
        age = parsed_query.get('age')
        if age is not None:
            if age < 5 or age > 65:
                risk_factors.append(f"Age {age} is in higher risk category")
                risk_score += 0.3
            elif age < 18:
                risk_factors.append("Pediatric case requires special consideration")
                risk_score += 0.1
        
        # Procedure-based risk
        procedure = parsed_query.get('procedure')
        if procedure and isinstance(procedure, str):
            high_risk_procedures = self.insurance_knowledge['demographic_factors']['risk_factors']['high_risk_procedures']
            procedure_lower = procedure.lower()
            if any(proc and isinstance(proc, str) and proc in procedure_lower for proc in high_risk_procedures):
                risk_factors.append(f"High-risk procedure: {procedure}")
                risk_score += 0.4
        
        # Condition-based risk
        condition = parsed_query.get('condition')
        if condition and isinstance(condition, str):
            high_risk_conditions = self.insurance_knowledge['demographic_factors']['risk_factors']['high_risk_conditions']
            condition_lower = condition.lower()
            if any(cond and isinstance(cond, str) and cond in condition_lower for cond in high_risk_conditions):
                risk_factors.append(f"High-risk condition: {condition}")
                risk_score += 0.3
        
        # Urgency risk
        if parsed_query.get('urgency') == 'high':
            risk_factors.append("Emergency/urgent treatment")
            risk_score += 0.2
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'risk_score': min(risk_score, 1.0),
            'factors': risk_factors
        }
    
    def _calculate_coverage_amount(self, parsed_query: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate coverage amount based on clauses"""
        amounts_found = []
        default_amounts = {
            'surgery': 100000,
            'therapy': 50000,
            'emergency': 150000,
            'dental': 25000,
            'general': 75000
        }
        
        # Extract amounts from clauses
        for clause in clauses:
            amounts = clause.get('entities', {}).get('amounts', [])
            for amount in amounts:
                # Parse amount
                amount_str = re.sub(r'[^\d.]', '', str(amount))
                try:
                    amount_value = float(amount_str)
                    amounts_found.append(amount_value)
                except:
                    continue
        
        # Determine coverage amount
        if amounts_found:
            # Use maximum amount found
            covered_amount = max(amounts_found)
        else:
            # Use default based on procedure type
            procedure = parsed_query.get('procedure') or ''
            procedure_lower = procedure.lower() if isinstance(procedure, str) else ''
            if 'surgery' in procedure_lower:
                covered_amount = default_amounts['surgery']
            elif 'therapy' in procedure_lower:
                covered_amount = default_amounts['therapy']
            elif parsed_query.get('urgency') == 'high':
                covered_amount = default_amounts['emergency']
            elif 'dental' in procedure_lower:
                covered_amount = default_amounts['dental']
            else:
                covered_amount = default_amounts['general']
        
        return {
            'covered_amount': covered_amount,
            'amounts_found': amounts_found,
            'calculation_method': 'extracted' if amounts_found else 'default'
        }
    
    def _build_justification(self, parsed_query: Dict[str, Any], coverage_analysis: Dict[str, Any], 
                            risk_assessment: Dict[str, Any], amount_calculation: Dict[str, Any]) -> str:
        """Build detailed justification for the decision"""
        justification_parts = []
        
        # Coverage justification
        if coverage_analysis['is_covered']:
            justification_parts.append(
                f"Coverage analysis indicates this is covered under the policy "
                f"({coverage_analysis['reasoning']})"
            )
        else:
            justification_parts.append(
                f"Coverage analysis indicates this may not be covered "
                f"({coverage_analysis['reasoning']})"
            )
        
        # Risk assessment justification
        risk_level = risk_assessment['risk_level']
        if risk_level == 'low':
            justification_parts.append("Risk assessment shows low risk factors")
        elif risk_level == 'medium':
            justification_parts.append(
                f"Risk assessment shows medium risk factors: {', '.join(risk_assessment['factors'])}"
            )
        else:
            justification_parts.append(
                f"Risk assessment shows high risk factors: {', '.join(risk_assessment['factors'])}"
            )
        
        # Amount justification
        if amount_calculation['amounts_found']:
            justification_parts.append(
                f"Coverage amount based on policy limits found in documents"
            )
        else:
            justification_parts.append(
                f"Coverage amount based on standard limits for this type of treatment"
            )
        
        return ". ".join(justification_parts) + "."
    
    def _generate_recommendations(self, parsed_query: Dict[str, Any], status: str) -> List[str]:
        """Generate recommendations based on the decision"""
        recommendations = []
        
        if status == "approved":
            recommendations.append("Submit required documentation for claim processing")
            recommendations.append("Ensure treatment is from network providers for maximum coverage")
        
        elif status == "conditional_approval":
            recommendations.append("Additional medical documentation may be required")
            recommendations.append("Consider pre-authorization for better coverage")
        
        elif status == "review_required":
            recommendations.append("Manual review required due to high risk factors")
            recommendations.append("Provide comprehensive medical history and treatment plan")
        
        elif status == "rejected":
            recommendations.append("Review policy terms for coverage details")
            recommendations.append("Consider supplementary insurance for this type of treatment")
        
        else:  # information_needed
            recommendations.append("Provide more specific information about the treatment needed")
            recommendations.append("Include age, procedure details, and location for better assessment")
        
        # Age-specific recommendations
        age = parsed_query.get('age')
        if age and age > 65:
            recommendations.append("Senior citizen benefits may be available")
        elif age and age < 18:
            recommendations.append("Pediatric coverage terms may apply")
        
        return recommendations
    
    def _process_with_gpt4(self, query: str, relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process query using GPT-4 (Hackathon Primary Method)"""
        try:
            if not self.use_gpt4 or not HAS_OPENAI:
                raise Exception("GPT-4 not available")
            
            # Prepare context from relevant clauses
            context = "\n\n".join([
                f"Document: {clause.get('document', 'Unknown')}\n"
                f"Clause: {clause.get('text', '')}"
                for clause in relevant_clauses[:10]  # Top 10 clauses
            ])
            
            # Create GPT-4 prompt
            prompt = f"""
You are an expert insurance policy analyst. Analyze the following query against the provided policy clauses and provide a structured decision.

QUERY: "{query}"

RELEVANT POLICY CLAUSES:
{context}

Please analyze and respond with a JSON structure containing:
1. decision: "approved", "rejected", "conditional_approval", or "review_required"
2. amount: estimated coverage amount in INR (number)
3. confidence: confidence score 0.0 to 1.0
4. justification: detailed explanation of the decision
5. risk_factors: array of identified risk factors
6. recommendations: array of recommendations for the patient

Respond ONLY with valid JSON. Be thorough but concise.
"""

            # Call GPT-4 API
            response = self.openai_client.chat.completions.create(
                model=self.config['llm_model'],
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                top_p=self.config['top_p']
            )
            
            # Parse response
            gpt_response = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(gpt_response)
                
                # Validate required fields
                required_fields = ['decision', 'amount', 'confidence', 'justification']
                for field in required_fields:
                    if field not in result:
                        result[field] = self._get_default_value(field)
                
                # Ensure proper types
                result['amount'] = float(result.get('amount', 0))
                result['confidence'] = min(max(float(result.get('confidence', 0)), 0.0), 1.0)
                result['risk_factors'] = result.get('risk_factors', [])
                result['recommendations'] = result.get('recommendations', [])
                
                logger.info("✅ GPT-4 processing successful")
                return result
                
            except json.JSONDecodeError:
                logger.warning("⚠️ GPT-4 returned invalid JSON, using fallback parsing")
                return self._parse_gpt_text_response(gpt_response)
                
        except Exception as e:
            logger.error(f"❌ GPT-4 processing failed: {e}")
            raise Exception(f"GPT-4 processing error: {e}")
    
    def _get_default_value(self, field: str):
        """Get default values for missing fields"""
        defaults = {
            'decision': 'review_required',
            'amount': 0.0,
            'confidence': 0.5,
            'justification': 'Analysis completed with limited information',
            'risk_factors': [],
            'recommendations': []
        }
        return defaults.get(field, None)
    
    def _parse_gpt_text_response(self, text: str) -> Dict[str, Any]:
        """Parse GPT-4 text response as fallback"""
        result = {
            'decision': 'review_required',
            'amount': 0.0,
            'confidence': 0.6,
            'justification': text[:500] if text else 'GPT-4 analysis completed',
            'risk_factors': [],
            'recommendations': ['Manual review recommended due to parsing issues']
        }
        
        # Try to extract decision keywords
        text_lower = text.lower() if text else ''
        if 'approved' in text_lower:
            result['decision'] = 'approved'
        elif 'rejected' in text_lower or 'denied' in text_lower:
            result['decision'] = 'rejected'
        elif 'conditional' in text_lower:
            result['decision'] = 'conditional_approval'
        
        # Try to extract amounts
        amounts = re.findall(r'[\d,]+', text)
        if amounts:
            try:
                result['amount'] = float(amounts[0].replace(',', ''))
            except:
                pass
        
        return result

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process a natural language query"""
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
                    "clauses_mapping": [],
                    "parsed_query": {},
                    "risk_factors": [],
                    "recommendations": ["Please provide a valid query"],
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
                    "clauses_mapping": [],
                    "parsed_query": {},
                    "risk_factors": [],
                    "recommendations": ["Please provide a valid query"],
                    "processing_metadata": {
                        "total_clauses_searched": len(self.clause_database),
                        "relevant_clauses_found": 0,
                        "processing_time": 0.0,
                        "error": "Empty query"
                    }
                }
            
            # Parse the query
            parsed_query = self.parse_query_advanced(query)
            
            # Search for relevant clauses
            relevant_clauses = self.search_relevant_clauses_hybrid(parsed_query)
            
            # PRIMARY: Use GPT-4 for decision making (Hackathon Requirement)
            try:
                if self.use_gpt4:
                    logger.info("🤖 Processing with GPT-4 (Primary Method)")
                    gpt4_result = self._process_with_gpt4(query, relevant_clauses)
                    
                    # Convert GPT-4 result to our standard format
                    decision = {
                        "status": gpt4_result.get("decision", "review_required"),
                        "amount": gpt4_result.get("amount", 0.0),
                        "confidence": gpt4_result.get("confidence", 0.6),
                        "justification": gpt4_result.get("justification", "GPT-4 analysis completed"),
                        "risk_factors": gpt4_result.get("risk_factors", []),
                        "recommendations": gpt4_result.get("recommendations", []),
                        "clauses_used": relevant_clauses[:5],  # Top 5 clauses
                        "method_used": "gpt4_primary"
                    }
                    
                    # Generate detailed response using GPT-4 context
                    detailed_response = f"🤖 **GPT-4 Analysis Result**\n\n"
                    detailed_response += f"**Decision**: {decision['status'].upper()}\n"
                    detailed_response += f"**Coverage Amount**: ₹{decision['amount']:,.0f}\n"
                    detailed_response += f"**Confidence**: {decision['confidence']:.1%}\n\n"
                    detailed_response += f"**Analysis**: {decision['justification']}\n\n"
                    
                    if decision['risk_factors']:
                        detailed_response += f"**Risk Factors**: {', '.join(decision['risk_factors'])}\n\n"
                    
                    if decision['recommendations']:
                        detailed_response += f"**Recommendations**:\n"
                        for rec in decision['recommendations']:
                            detailed_response += f"• {rec}\n"
                    
                else:
                    raise Exception("GPT-4 not available")
                    
            except Exception as e:
                # FALLBACK: Use traditional ML models
                logger.warning(f"⚠️ GPT-4 failed ({e}), using fallback models...")
                decision = self.evaluate_decision_advanced(parsed_query, relevant_clauses)
                decision["method_used"] = "fallback_ml"
                
                # Generate detailed response using QA pipeline
                detailed_response = self._generate_detailed_response(query, relevant_clauses)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.processing_stats['queries_processed'] += 1
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * (self.processing_stats['queries_processed'] - 1) + 
                 processing_time) / self.processing_stats['queries_processed']
            )
            
            # Format response
            response = {
                "query": query,
                "decision": decision["status"],
                "amount": decision["amount"],
                "confidence": decision["confidence"],
                "justification": decision["justification"],
                "detailed_response": detailed_response,
                "risk_factors": decision["risk_factors"],
                "recommendations": decision["recommendations"],
                "clauses_mapping": [
                    {
                        "clause_text": clause["text"],
                        "document": clause["document"],
                        "clause_type": clause.get("clause_type", "unknown"),
                        "similarity_score": clause.get("similarity_score", 0.0),
                        "search_method": clause.get("search_method", "unknown")
                    }
                    for clause in decision["clauses_used"]
                ],
                "parsed_query": parsed_query,
                "processing_metadata": {
                    "processing_time_seconds": processing_time,
                    "total_clauses_searched": len(self.clause_database),
                    "relevant_clauses_found": len(relevant_clauses),
                    "search_methods_used": list(set(clause.get("search_method", "unknown") for clause in relevant_clauses)),
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
    
    def _generate_detailed_response(self, query: str, relevant_clauses: List[Dict[str, Any]]) -> Optional[str]:
        """Generate detailed response using QA pipeline"""
        if not relevant_clauses or not self.qa_pipeline:
            return None
        
        try:
            # Combine top clauses as context
            context = " ".join([clause["text"] for clause in relevant_clauses[:3]])
            
            # Limit context length to avoid token limits
            max_context_length = 2000
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            # Use QA pipeline
            result = self.qa_pipeline(question=query, context=context)
            
            return result.get('answer', None)
            
        except Exception as e:
            logger.warning(f"Error generating detailed response: {e}")
            return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "documents_loaded": len(self.documents),
            "total_clauses": len(self.clause_database),
            "queries_processed": self.processing_stats['queries_processed'],
            "avg_processing_time": self.processing_stats['avg_processing_time'],
            "models_loaded": {
                "qa_pipeline": self.qa_pipeline is not None,
                "sentence_transformer": self.sentence_model is not None,
                "gpt4_available": getattr(self, 'use_gpt4', False),
                "pinecone_available": getattr(self, 'use_pinecone', False),
                "faiss_index": self.faiss_index is not None,
                "tfidf_vectorizer": self.tfidf_vectorizer is not None
            },
            "device": str(self.device),
            "config": self.config,
            "system_status": "healthy" if self.documents else "no_documents_loaded",
            "memory_usage": {
                "documents_size_mb": sum(len(doc.encode('utf-8')) for doc in self.documents.values()) / (1024 * 1024),
                "clauses_count": len(self.clause_database),
                "embeddings_loaded": self.faiss_index is not None
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = OptimizedDocumentProcessor()
    
    # Load documents
    dataset_path = "d:/Bajaj_Finserv/Datasets"
    
    try:
        processor.load_documents(dataset_path)
        
        # Test queries
        test_queries = [
            "46-year-old male, knee surgery in Pune, 3-month policy",
            "Female patient needs heart surgery, age 55, Mumbai location",
            "Dental treatment for 30-year-old, covered under policy?",
            "Emergency surgery required, what's the coverage limit?",
            "Cataract surgery eligibility for senior citizen",
            "Cancer treatment coverage for 45-year-old patient in Delhi",
            "Dialysis treatment for kidney disease, what are the benefits?",
            "Pre-existing diabetes condition, surgery needed urgently"
        ]
        
        print("=" * 100)
        print("OPTIMIZED LLM DOCUMENT PROCESSOR - COMPREHENSIVE TEST RESULTS")
        print("=" * 100)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Query: {query}")
            
            result = processor.process_query(query)
            
            print(f"Decision: {result['decision'].upper()}")
            print(f"Amount: ₹{result['amount']:,.2f}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Justification: {result['justification']}")
            
            if result.get('detailed_response'):
                print(f"\nDetailed Response: {result['detailed_response']}")
            
            if result.get('risk_factors'):
                print(f"\nRisk Factors: {', '.join(result['risk_factors'])}")
            
            if result.get('recommendations'):
                print(f"\nRecommendations:")
                for rec in result['recommendations']:
                    print(f"  • {rec}")
            
            print(f"\nProcessing Time: {result['processing_metadata']['processing_time_seconds']:.3f}s")
            print(f"Clauses Analyzed: {result['processing_metadata']['relevant_clauses_found']}")
            
            print("-" * 80)
        
        # Print comprehensive system stats
        stats = processor.get_system_stats()
        print(f"\n{'='*50} SYSTEM STATISTICS {'='*50}")
        print(json.dumps(stats, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
