"""
Pytest configuration and fixtures for cascadeguard tests.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for Fineranker testing."""
    mock = MagicMock()

    def mock_completion(messages, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "True"
        return response

    mock.side_effect = mock_completion
    return mock


@pytest.fixture
def sample_prompt_output_pairs():
    """Sample test data for guardrail testing."""
    return [
        ("What is AI?", "AI stands for Artificial Intelligence."),
        ("Explain Python", "Python is a programming language."),
        ("Hello", "Hi there!"),
    ]
