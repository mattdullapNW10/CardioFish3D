# Tests

Unit and integration tests for the project.

## Organization

- `test_preprocessing.py` — Tests for image preprocessing functions
- `test_models.py` — Tests for model architectures
- `test_utils.py` — Tests for utility functions
- `fixtures/` — Test data and fixtures

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run specific test
pytest tests/test_preprocessing.py::test_normalize_image
```

## Best Practices

- Use descriptive test names
- Include docstrings explaining what is being tested
- Use fixtures for reusable test data
- Aim for >80% code coverage
