#!/usr/bin/env python3
"""
Optimized Hackathon Readiness Test - Problem Statement Compliance
"""
import requests
import json
import time
from datetime import datetime

def test_server_health():
    """Quick server health check"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_problem_statement():
    """Test exact problem statement compliance"""
    print("🎯 PROBLEM STATEMENT TEST")
    print("=" * 40)
    
    query = "46-year-old male, knee surgery in Pune, 3-month policy"
    print(f"Query: {query}")
    
    try:
        response = requests.post(
            "http://localhost:8001/process_query",
            json={"query": query, "include_detailed_response": True},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return False
        
        result = response.json()
        
        # Check all requirements
        checks = {
            'Decision': result.get('decision', 'N/A'),
            'Amount': f"${result.get('amount', 0):,}",
            'Confidence': f"{result.get('confidence', 0)*100:.1f}%",
            'Clauses Found': len(result.get('clauses_mapping', [])),
            'Age Parsed': result.get('parsed_query', {}).get('age'),
            'Procedure Parsed': result.get('parsed_query', {}).get('procedure'),
            'Location Parsed': result.get('parsed_query', {}).get('location')
        }
        
        print("\n✅ RESULTS:")
        for key, value in checks.items():
            print(f"   {key}: {value}")
        
        # Verify compliance
        compliance = all([
            result.get('decision'),
            result.get('amount', 0) > 0,
            result.get('confidence', 0) > 0,
            len(result.get('clauses_mapping', [])) > 0,
            result.get('parsed_query', {}).get('age') is not None
        ])
        
        print(f"\n🎯 COMPLIANCE: {'✅ PASSED' if compliance else '❌ FAILED'}")
        return compliance
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def main():
    """Run optimized test"""
    print("🏆 HACKATHON READINESS - OPTIMIZED TEST")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check server
    if not test_server_health():
        print("❌ Server not running. Start with: python api.py")
        return False
    
    print("✅ Server running on http://localhost:8001")
    
    # Test compliance
    success = test_problem_statement()
    
    print("\n" + "="*50)
    if success:
        print("🎉 SYSTEM READY FOR HACKATHON!")
        print("🚀 Demo: http://localhost:8001")
    else:
        print("❌ Issues detected - check output above")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
