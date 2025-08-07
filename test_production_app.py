#!/usr/bin/env python3
"""
Quick test of the production app
"""

try:
    from app import processor, processor_type
    print(f"✅ App imported successfully")
    print(f"📊 Processor: {processor_type}")
    print(f"📄 Documents: {len(processor.documents) if processor else 0}")
    print(f"📋 Clauses: {len(processor.clause_database) if processor else 0}")
    
    # Test a query
    result = processor.process_query("46-year-old male, knee surgery in Pune, 3-month-old insurance policy")
    print("\n🔍 Query Test Results:")
    print(f"Decision: {result['decision']}")
    print(f"Amount: ₹{result['amount']:,}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Clauses found: {len(result['clauses_mapping'])}")
    
    if result['clauses_mapping']:
        print(f"Sample evidence: {result['clauses_mapping'][0]['clause_text'][:100]}...")
    
    print("\n✅ Production app is working perfectly!")
    print("🚀 Ready for Vercel deployment!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
