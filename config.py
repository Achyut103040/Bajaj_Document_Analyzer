import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create directories if they don't exist
directories = ['logs', 'temp', 'models', 'data']
for directory in directories:
    Path(directory).mkdir(exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration settings
class Config:
    """Hackathon-optimized configuration with GPT-4 and Pinecone support"""
    
    # HACKATHON PRIMARY STACK
    # GPT-4 Configuration (Primary LLM)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    USE_GPT4 = bool(OPENAI_API_KEY)  # Auto-enable if API key present
    
    # Pinecone Configuration (Primary Vector DB)
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
    USE_PINECONE = bool(PINECONE_API_KEY)  # Auto-enable if API key present
    
    # Server settings
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8001))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Demo mode (fallback when APIs not available)
    DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'
    
    # Legacy settings (maintained for compatibility)
    DATASETS_FOLDER = "Datasets"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS = ['.pdf']
    MODEL_CACHE_DIR = './models'
    TEMP_DIR = './temp'
    
    # Processing settings
    MAX_CLAUSES_PER_SEARCH = 15
    SIMILARITY_THRESHOLD = 0.8 if USE_GPT4 else 0.7
    CONFIDENCE_THRESHOLD = 0.7 if USE_GPT4 else 0.6
    
    @classmethod
    def get_status(cls):
        """Get configuration status for debugging"""
        return {
            "gpt4_available": cls.USE_GPT4,
            "pinecone_available": cls.USE_PINECONE, 
            "demo_mode": cls.DEMO_MODE,
            "api_port": cls.API_PORT,
            "datasets_folder": cls.DATASETS_FOLDER
        }
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models')
    TEMP_DIR = os.getenv('TEMP_DIR', './temp')
    MAX_FILE_SIZE = os.getenv('MAX_FILE_SIZE', '50MB')
    DEVICE = os.getenv('DEVICE', 'auto')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8001'))
    
    # Dataset configuration
    DATASET_PATH = os.getenv('DATASET_PATH', 'Datasets')
    
    # Model configuration
    QA_MODEL = os.getenv('QA_MODEL', 'distilbert-base-cased-distilled-squad')
    SENTENCE_MODEL = os.getenv('SENTENCE_MODEL', 'all-MiniLM-L6-v2')
    SPACY_MODEL = os.getenv('SPACY_MODEL', 'en_core_web_sm')

config = Config()
