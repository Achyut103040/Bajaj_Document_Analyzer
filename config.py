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
