#!/usr/bin/env python3
"""Check PDF libraries"""

try:
    import fitz
    print('✅ PyMuPDF (fitz) is available')
    print(f'   Version: {fitz.version}')
except ImportError as e:
    print('❌ PyMuPDF not available:', e)
    
try:
    import PyPDF2
    print('✅ PyPDF2 is available')
    print(f'   Version: {PyPDF2.__version__}')
except ImportError as e:
    print('❌ PyPDF2 not available:', e)
