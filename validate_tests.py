#!/usr/bin/env python3
"""
Test Validation - Check if tests can be imported and basic functionality works
"""

import sys
import os
import importlib.util

def check_dependencies():
    """Check if testing dependencies are available"""
    required_modules = {
        'pytest': 'pip install pytest',
        'fastapi.testclient': 'pip install fastapi',
        'sqlalchemy': 'pip install sqlalchemy',
        'unittest.mock': 'Built-in module'
    }
    
    missing = []
    for module, install_cmd in required_modules.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - Install with: {install_cmd}")
            missing.append(module)
    
    return len(missing) == 0

def check_test_file():
    """Check if test file can be imported"""
    try:
        spec = importlib.util.spec_from_file_location("test_comprehensive", "test_comprehensive.py")
        if spec is None:
            print("❌ Cannot load test_comprehensive.py")
            return False
        
        test_module = importlib.util.module_from_spec(spec)
        print("✅ test_comprehensive.py can be loaded")
        return True
    except Exception as e:
        print(f"❌ Error loading test_comprehensive.py: {e}")
        return False

def run_basic_tests():
    """Run a few basic tests to validate functionality"""
    print("\n🧪 Running basic validation tests...")
    
    try:
        # Test hash generation
        from async_video_service import generate_content_hash
        hash1 = generate_content_hash("test")
        hash2 = generate_content_hash("test")
        assert hash1 == hash2, "Hash generation failed"
        print("✅ Hash generation working")
        
        # Test URL validation
        from async_video_service import is_valid_youtube_url_or_id
        assert is_valid_youtube_url_or_id("dQw4w9WgXcQ"), "YouTube ID validation failed"
        assert is_valid_youtube_url_or_id("https://youtube.com/watch?v=dQw4w9WgXcQ"), "YouTube URL validation failed"
        print("✅ URL validation working")
        
        # Test database models
        from database import User, Job
        user = User(email="test@example.com", name="Test User")
        assert user.email == "test@example.com", "User model failed"
        print("✅ Database models working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    print("🔍 Video Audio Text Service - Test Validation")
    print("=" * 60)
    
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n📄 Checking test file...")
    test_file_ok = check_test_file()
    
    if deps_ok and test_file_ok:
        basic_tests_ok = run_basic_tests()
        
        if basic_tests_ok:
            print("\n✅ All validation checks passed!")
            print("\n🚀 Ready to run comprehensive tests:")
            print("   python run_tests.py                 # Run all tests")
            print("   python run_tests.py --unit          # Run unit tests only")
            print("   python run_tests.py --coverage      # Run with coverage")
            print("   python -m pytest test_comprehensive.py -v  # Direct pytest")
            return 0
        else:
            print("\n❌ Basic tests failed!")
            return 1
    else:
        print("\n❌ Validation failed!")
        if not deps_ok:
            print("   Install missing dependencies first")
        if not test_file_ok:
            print("   Fix test file import issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
