# CascadeGuard Tests

## Overview

This directory contains unit tests for the CascadeGuard library components.

## Test Files

- `test_preranker.py`: Tests for Preranker class (scanner initialization, configuration)
- `test_fineranker.py`: Tests for Fineranker class (LLM-based ranking, boolean parsing)
- `test_guardrail.py`: Tests for CascadeGuard class (cascade logic, data conversion)

## Running Tests

### Run all tests
```bash
uv run pytest src/cascadeguard/tests/ -v
```

### Run specific test file
```bash
uv run pytest src/cascadeguard/tests/test_preranker.py -v
uv run pytest src/cascadeguard/tests/test_fineranker.py -v
uv run pytest src/cascadeguard/tests/test_guardrail.py -v
```

### Run with coverage
```bash
uv run pytest src/cascadeguard/tests/ --cov=cascadeguard --cov-report=html
```

## Test Structure

Tests are organized by class and functionality:

- **Initialization tests**: Verify proper object creation and configuration
- **Method tests**: Test individual methods and their behavior
- **Error handling tests**: Ensure proper error handling for invalid inputs
- **Integration tests**: Test interaction between components

## Fixtures

Test fixtures are defined in `conftest.py`:

- `mock_llm_response`: Mock LLM completion for Fineranker testing
- `sample_prompt_output_pairs`: Sample data for testing guardrail pipelines

## Notes

- Tests use mocking to avoid downloading actual models during testing
- All tests run quickly without external dependencies
- Tests cover core functionality without being overly verbose
