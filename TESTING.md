# 🧪 Comprehensive Unit Tests for Video Audio Text Service

This directory contains a comprehensive test suite for the Video Audio Text Service, covering all major functionality including database operations, utility functions, audio/video processing, API endpoints, authentication, and background tasks.

## 📋 Test Coverage

### Core Test Categories

- **Database Tests** (`TestDatabaseModels`, `TestDatabaseFunctions`)
  - User, Job, Subscription, UserPreferences models
  - Database utility functions (subscription checks, usage limits, etc.)
  - Duplicate detection and content reuse

- **Utility Function Tests** (`TestUtilityFunctions`, `TestURLValidation`)
  - Hash generation (content, file, media)
  - URL validation for multiple video platforms
  - YouTube ID extraction and normalization
  - Markdown filtering

- **Audio/Video Processing Tests** (`TestAudioVideoProcessing`)
  - Audio extraction from video files
  - Audio format conversion
  - Speech transcription with Whisper
  - Text-to-speech with Edge-TTS
  - MLX summarization with dynamic token allocation

- **API Endpoint Tests** (`TestAPIEndpoints`)
  - Authentication and authorization
  - File upload validation
  - Usage limit enforcement
  - Error handling

- **Authentication Tests** (`TestAuthentication`)
  - JWT token validation
  - User session management
  - Access control

- **Background Processing Tests** (`TestBackgroundProcessing`)
  - Video download from multiple platforms
  - YouTube transcript extraction
  - Async task processing

- **Error Handling Tests** (`TestErrorHandling`)
  - Edge cases and invalid inputs
  - Graceful failure modes
  - Exception handling

- **Integration Tests** (`TestIntegration`)
  - Full workflow testing
  - Multi-component interactions
  - End-to-end scenarios

- **Performance Tests** (`TestPerformance`)
  - Hash generation performance
  - Database query performance
  - Load testing scenarios

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
# Install core testing framework
pip install pytest pytest-asyncio pytest-mock

# Or install all testing dependencies
pip install -r requirements-test.txt
```

### 2. Validate Test Setup

```bash
# Check if tests can run
python validate_tests.py
```

### 3. Run Tests

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit           # Unit tests only
python run_tests.py --integration    # Integration tests only
python run_tests.py --api           # API endpoint tests
python run_tests.py --database      # Database tests only
python run_tests.py --processing    # Audio/video processing tests
python run_tests.py --auth          # Authentication tests

# Run with coverage report
python run_tests.py --coverage

# Run specific test file or function
python run_tests.py --file test_comprehensive.py
python run_tests.py --function test_generate_content_hash
```

### 4. Direct Pytest Usage

```bash
# Run all tests with verbose output
python -m pytest test_comprehensive.py -v

# Run specific test class
python -m pytest test_comprehensive.py::TestDatabaseModels -v

# Run with coverage
python -m pytest test_comprehensive.py --cov=async_video_service --cov=database --cov-report=html

# Run with specific markers
python -m pytest -m "unit" -v              # Unit tests only
python -m pytest -m "not slow" -v          # Exclude slow tests
python -m pytest -m "database or api" -v   # Database OR API tests
```

## 📊 Test Structure

### Test Fixtures

- `test_db`: In-memory SQLite database for testing
- `sample_user`: Standard free-tier user
- `premium_user`: Premium subscription user
- `sample_job`: Completed job with sample data
- `temp_file`: Temporary file for testing file operations
- `temp_audio_file`: Temporary audio file for audio processing tests

### Mocking Strategy

The tests use extensive mocking to isolate components:

- **External APIs**: YouTube, video download services
- **File I/O**: Audio/video file operations
- **ML Models**: Whisper transcription, MLX summarization
- **TTS Services**: Edge-TTS, system TTS
- **Database**: Isolated test database per test

### Test Data

- Uses factory patterns for creating test data
- Realistic but safe test content
- Predictable test scenarios

## 🔧 Configuration

### pytest.ini

The test suite is configured with:
- Automatic test discovery
- Strict marker enforcement
- Warning suppression for cleaner output
- Custom markers for test categorization

### Test Markers

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Multi-component integration tests
- `@pytest.mark.slow`: Tests that may take several seconds
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.database`: Database-related tests
- `@pytest.mark.auth`: Authentication tests
- `@pytest.mark.processing`: Audio/video processing tests

## 📈 Coverage Goals

Target coverage areas:

- **Core Functions**: 95%+ coverage
- **API Endpoints**: 90%+ coverage
- **Database Operations**: 95%+ coverage
- **Error Handling**: 85%+ coverage
- **Edge Cases**: 80%+ coverage

## 🐛 Debugging Tests

### Running Individual Tests

```bash
# Debug specific test with full output
python -m pytest test_comprehensive.py::TestDatabaseModels::test_user_creation -v -s

# Run with pdb debugger
python -m pytest test_comprehensive.py::test_specific_function --pdb

# Show local variables on failure
python -m pytest test_comprehensive.py -v --tb=long
```

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Errors**: Each test uses isolated database
3. **File Permission Errors**: Tests clean up temporary files
4. **Async Test Issues**: Use `@pytest.mark.asyncio` for async tests

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    - name: Validate tests
      run: python validate_tests.py
    - name: Run tests
      run: python run_tests.py --coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 📝 Adding New Tests

### Test File Structure

```python
class TestNewFeature:
    """Test new feature functionality"""
    
    def test_basic_functionality(self, test_db, sample_user):
        """Test basic functionality"""
        # Arrange
        # Act
        # Assert
        pass
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality"""
        # Use appropriate mocks
        with patch('module.function') as mock_func:
            # Test async code
            pass
    
    @pytest.mark.slow
    def test_performance_feature(self):
        """Test that might be slow"""
        # Performance or integration test
        pass
```

### Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies
3. **Clear Names**: Test names should describe what they test
4. **AAA Pattern**: Arrange, Act, Assert
5. **Edge Cases**: Test boundary conditions
6. **Error Cases**: Test error handling
7. **Documentation**: Document complex test scenarios

## 🛠 Maintenance

### Updating Tests

- Update tests when adding new features
- Maintain mocks when external APIs change
- Update fixtures when database schema changes
- Review test performance regularly

### Test Review Checklist

- [ ] Tests cover new functionality
- [ ] Error cases are tested
- [ ] Mocks are appropriate and realistic
- [ ] Tests are properly categorized with markers
- [ ] Documentation is updated
- [ ] Performance impact is considered

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites)
- [Async Testing Best Practices](https://pytest-asyncio.readthedocs.io/)

---

## 🎯 Example Usage

```bash
# Quick validation
python validate_tests.py

# Development workflow
python run_tests.py --unit --coverage  # Fast feedback during development
python run_tests.py --integration     # Before committing
python run_tests.py --slow           # Before releases

# Debugging specific issues
python -m pytest test_comprehensive.py::TestDatabaseModels -v -s
```

This comprehensive test suite ensures the Video Audio Text Service is robust, reliable, and maintainable. The tests provide confidence when making changes and help catch issues early in the development process.
