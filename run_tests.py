#!/usr/bin/env python3
"""
Test Runner for Video Audio Text Service
Provides easy commands to run different types of tests
"""

import sys
import subprocess
import argparse
import os

def run_command(command, description):
    """Run a command and handle the output"""
    print(f"\n🧪 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Exit code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for Video Audio Text Service")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--database", action="store_true", help="Run database tests only")
    parser.add_argument("--processing", action="store_true", help="Run processing tests only")
    parser.add_argument("--auth", action="store_true", help="Run authentication tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument("--function", help="Run specific test function")
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Install test dependencies:")
        print("   pip install -r requirements-test.txt")
        return 1
    
    # Build pytest command
    cmd_parts = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd_parts.append("-v")
    else:
        cmd_parts.append("-q")
    
    # Add coverage
    if args.coverage:
        cmd_parts.extend(["--cov=async_video_service", "--cov=database", "--cov-report=html", "--cov-report=term"])
    
    # Add markers for specific test types
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.api:
        markers.append("api")
    if args.database:
        markers.append("database")
    if args.processing:
        markers.append("processing")
    if args.auth:
        markers.append("auth")
    
    if markers:
        cmd_parts.extend(["-m", " or ".join(markers)])
    
    # Exclude slow tests unless explicitly requested
    if not args.slow and not any([args.unit, args.integration, args.api, args.database, args.processing, args.auth]):
        cmd_parts.extend(["-m", "'not slow'"])
    
    # Add specific file or function
    if args.file:
        cmd_parts.append(args.file)
        if args.function:
            cmd_parts[-1] += f"::{args.function}"
    elif args.function:
        cmd_parts.append(f"test_comprehensive.py::{args.function}")
    
    # If no specific tests selected, run comprehensive tests
    if not any([args.unit, args.integration, args.api, args.database, args.processing, args.auth, args.file]):
        cmd_parts.append("test_comprehensive.py")
    
    command = " ".join(cmd_parts)
    
    print("🚀 Video Audio Text Service - Test Runner")
    print("=" * 60)
    print(f"Running command: {command}")
    
    success = run_command(command, "Running Tests")
    
    if success:
        print("\n✅ All tests completed successfully!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
