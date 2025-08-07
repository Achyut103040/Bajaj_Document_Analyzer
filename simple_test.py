#!/usr/bin/env python3

import sys
import os

print("🧪 SIMPLE CLAUSE TEST")
print("=" * 25)

try:
    print("1. Importing processor...")
    from optimized_processor import OptimizedDocumentProcessor
    print("   ✅ Import successful")
    
    print("2. Creating processor...")
    processor = OptimizedDocumentProcessor()
    print("   ✅ Processor created")
    
    print("3. Loading documents...")
    processor.load_documents("Datasets")
    print(f"   ✅ Documents: {len(processor.documents)}")
    print(f"   ✅ Clauses: {len(processor.clause_database)}")
    
    if processor.clause_database:
        print("4. Testing query...")
        result = processor.process_query("knee surgery coverage")
        print(f"   ✅ Decision: {result['decision']}")
        print(f"   ✅ Clauses: {len(result['clauses_mapping'])}")
        
        if result['clauses_mapping']:
            print("🎉 SUCCESS: Clause generation is working!")
        else:
            print("⚠️ No clauses mapped to query")
    else:
        print("❌ No clauses generated")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
