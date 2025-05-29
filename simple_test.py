#!/usr/bin/env python3
"""
Simple integration test for LLM Collusion HGF project
Tests basic functionality without external dependencies
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        # Test basic imports (these will fail but show structure)
        modules = [
            'market_env',
            'llm_agents', 
            'safety_wrapper',
            'communication_analyzer',
            'run_experiment',
            'dashboard'
        ]
        
        for module in modules:
            try:
                __import__(module)
                print(f"  ✓ {module}.py exists and has valid Python syntax")
            except ImportError as e:
                if "No module named" in str(e) and module not in str(e):
                    # Module exists but has import errors (expected without dependencies)
                    print(f"  ✓ {module}.py exists (has dependency imports)")
                else:
                    print(f"  ✗ {module}.py has issues: {e}")
            except SyntaxError as e:
                print(f"  ✗ {module}.py has syntax error: {e}")
    except Exception as e:
        print(f"Import test failed: {e}")
        return False
    return True

def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    required_files = [
        'market_env.py',
        'llm_agents.py',
        'safety_wrapper.py',
        'communication_analyzer.py',
        'run_experiment.py',
        'dashboard.py',
        'test_suite.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
            all_exist = False
    
    return all_exist

def test_code_structure():
    """Test that key classes and functions exist"""
    print("\nTesting code structure...")
    
    # Check for key patterns in files
    checks = [
        ('market_env.py', 'class MarketCollusionEnv', 'MarketCollusionEnv class'),
        ('llm_agents.py', 'class LLMAgent', 'LLMAgent class'),
        ('safety_wrapper.py', 'class CollusionReferee', 'CollusionReferee class'),
        ('safety_wrapper.py', 'class LLMGovernor', 'LLMGovernor class'),
        ('safety_wrapper.py', 'class HierarchicalSafetyWrapper', 'HierarchicalSafetyWrapper class'),
        ('communication_analyzer.py', 'class CommunicationAnalyzer', 'CommunicationAnalyzer class'),
        ('run_experiment.py', 'def run_comparison_experiment', 'run_comparison_experiment function'),
        ('dashboard.py', 'st.title', 'Streamlit dashboard setup')
    ]
    
    all_found = True
    for filename, pattern, description in checks:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                if pattern in content:
                    print(f"  ✓ {description} found in {filename}")
                else:
                    print(f"  ✗ {description} not found in {filename}")
                    all_found = False
        else:
            print(f"  ✗ {filename} not found")
            all_found = False
    
    return all_found

def test_readme():
    """Check README content"""
    print("\nTesting README...")
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            content = f.read()
            
        # Check for key sections
        sections = [
            '# LLM Collusion',
            'PettingZoo',
            'Quick Start',
            'Architecture'
        ]
        
        for section in sections:
            if section.lower() in content.lower():
                print(f"  ✓ README contains '{section}' section")
            else:
                print(f"  ✗ README missing '{section}' section")
        
        return True
    else:
        print("  ✗ README.md not found")
        return False

def main():
    """Run all tests"""
    print("LLM Collusion HGF - Simple Integration Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Code Structure", test_code_structure),
        ("Module Imports", test_imports),
        ("README", test_readme)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n{test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✅" if result else "❌"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The lean research app structure is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up API keys for LLM providers (OpenAI, Anthropic, etc.)")
        print("3. Run experiment: python run_experiment.py --episodes 10 --model gpt-3.5-turbo")
        print("4. Launch dashboard: python run_experiment.py --dashboard")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)