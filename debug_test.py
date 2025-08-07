#!/usr/bin/env python3
"""
Debug test to check the clause generation issue
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

print("=== Debugging Clause Generation Issue ===")
print()

# Check if datasets exist
datasets_path = "d:/Bajaj_Document_Analyzer/Datasets"
print(f"1. Checking datasets at: {datasets_path}")
if os.path.exists(datasets_path):
    files = [f for f in os.listdir(datasets_path) if f.endswith('.pdf')]
    print(f"   ✅ Found {len(files)} PDF files:")
    for f in files:
        print(f"      - {f}")
else:
    print("   ❌ Datasets directory not found!")

print()

# Test enhanced processor initialization
try:
    print("2. Testing Enhanced Processor...")
    from enhanced_main import OptimizedDocumentProcessor
    processor = OptimizedDocumentProcessor()
    print("   ✅ Enhanced processor initialized")
    
    print(f"   Documents loaded: {len(processor.documents)}")
    print(f"   Clause database size: {len(processor.clause_database)}")
    
    if os.path.exists(datasets_path):
        print("   Loading documents...")
        processor.load_documents(datasets_path)
        print(f"   After loading - Documents: {len(processor.documents)}")
        print(f"   After loading - Clauses: {len(processor.clause_database)}")
        
        if processor.clause_database:
            print("   ✅ Clauses generated successfully!")
            print(f"   Sample clause: {processor.clause_database[0]['text'][:100]}...")
        else:
            print("   ❌ No clauses generated!")
    
    # Test a simple query
    print()
    print("3. Testing Query Processing...")
    test_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    print(f"   Query: {test_query}")
    
    result = processor.process_query(test_query)
    print(f"   Decision: {result.get('decision')}")
    print(f"   Confidence: {result.get('confidence')}")
    print(f"   Clauses found: {len(result.get('clauses_mapping', []))}")
    
    if result.get('clauses_mapping'):
        print("   ✅ Clauses mapped successfully!")
        for i, clause in enumerate(result.get('clauses_mapping', [])[:2]):
            print(f"      Clause {i+1}: {clause.get('text', '')[:80]}...")
    else:
        print("   ❌ No clauses mapped!")
        
except Exception as e:
    print(f"   ❌ Enhanced processor failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test lightweight processor as fallback
try:
    print("4. Testing Lightweight Processor...")
    from lightweight_main import LightweightDocumentProcessor
    light_processor = LightweightDocumentProcessor()
    print("   ✅ Lightweight processor initialized")
    
    print(f"   Documents loaded: {len(light_processor.documents)}")
    print(f"   Clause database size: {len(light_processor.clause_database)}")
    
    if os.path.exists(datasets_path):
        light_processor.load_documents(datasets_path)
        print(f"   After loading - Documents: {len(light_processor.documents)}")
        print(f"   After loading - Clauses: {len(light_processor.clause_database)}")
    
    # Test simple query
    test_query = "knee surgery coverage"
    result = light_processor.process_query(test_query)
    print(f"   Query result decision: {result.get('decision')}")
    print(f"   Clauses found: {len(result.get('clauses_mapping', []))}")
    
except Exception as e:
    print(f"   ❌ Lightweight processor failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== Debug Complete ===")
