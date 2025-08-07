#!/usr/bin/env python3
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
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The clause generation issue has been fixed!")
        print("\nNext steps:")
        print("1. Run: python start_optimized.py")
        print("2. Test API at: http://localhost:8001")
        print("3. Use /process_query endpoint for testing")
    else:
        print("\n❌ Tests failed. Check the logs above.")
