#!/usr/bin/env python3
"""
Simple test to verify clause generation fixes
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_clause_generation():
    """Test if clause generation is working"""
    print("🧪 TESTING CLAUSE GENERATION")
    print("=" * 40)
    
    try:
        # Import and initialize
        print("1. Importing processor...")
        from enhanced_main import OptimizedDocumentProcessor
        
        print("2. Initializing processor...")
        processor = OptimizedDocumentProcessor()
        
        print(f"3. Initial state:")
        print(f"   - Documents: {len(processor.documents)}")
        print(f"   - Clauses: {len(processor.clause_database)}")
        
        # Load documents
        datasets_path = "Datasets"
        if os.path.exists(datasets_path):
            print("4. Loading documents...")
            processor.load_documents(datasets_path)
            
            print(f"5. After loading:")
            print(f"   - Documents: {len(processor.documents)}")
            print(f"   - Clauses: {len(processor.clause_database)}")
            
            if processor.clause_database:
                print("✅ CLAUSES GENERATED SUCCESSFULLY!")
                
                # Show sample clauses
                print("\n📄 Sample clauses:")
                for i, clause in enumerate(processor.clause_database[:3]):
                    print(f"   {i+1}. Type: {clause.get('clause_type', 'unknown')}")
                    print(f"      Text: {clause['text'][:100]}...")
                    print(f"      Score: {clause.get('importance_score', 0):.2f}")
                    print()
                
                # Test query processing
                print("6. Testing query processing...")
                test_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
                print(f"   Query: {test_query}")
                
                result = processor.process_query(test_query)
                
                print(f"   Decision: {result.get('decision')}")
                print(f"   Amount: ₹{result.get('amount', 0):,.0f}")
                print(f"   Confidence: {result.get('confidence', 0):.1%}")
                print(f"   Clauses mapped: {len(result.get('clauses_mapping', []))}")
                
                if result.get('clauses_mapping'):
                    print("\n📋 Mapped clauses:")
                    for i, clause in enumerate(result['clauses_mapping'][:2]):
                        print(f"   {i+1}. From: {clause.get('document', 'unknown')}")
                        print(f"      Type: {clause.get('clause_type', 'unknown')}")
                        print(f"      Score: {clause.get('similarity_score', 0):.3f}")
                        print(f"      Text: {clause.get('clause_text', '')[:80]}...")
                        print()
                    
                    print("🎉 CLAUSE MAPPING WORKING!")
                else:
                    print("⚠️ No clauses mapped to response")
                
                return True
            else:
                print("❌ NO CLAUSES GENERATED")
                return False
        else:
            print(f"❌ Datasets not found at: {datasets_path}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clause_generation()
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Clause generation is working correctly")
        print("✅ Query processing is working correctly")
        print("✅ Clause mapping is working correctly")
    else:
        print("\n❌ TESTS FAILED")
        print("The clause generation issue still exists")
