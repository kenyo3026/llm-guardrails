"""
Unit tests for Preranker class.

Tests cover:
- Relevance scanner initialization
- Scanner functionality
- Error handling

Usage:
    pytest src/cascadeguard/tests/test_preranker.py -v
"""

import sys

import numpy as np
import pytest
from cascadeguard.preranker import Preranker, ScannerType


class TestPrerankerRelevance:
    """Tests for Preranker with Relevance scanner."""

    def test_init_relevance_scanner(self):
        """Test initializing with Relevance scanner (actual model from config)."""
        preranker = Preranker(
            scanner="Relevance",
            scanner_type=ScannerType.OUTPUT,
            scanner_kwargs={
                "model": {
                    "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                },
                "threshold": 0.5
            }
        )
        assert preranker.scanner is not None
        assert hasattr(preranker.scanner, "scan")

    def test_relevance_scan_valid_pair(self):
        """Test Relevance scanner with valid prompt-output pair."""
        preranker = Preranker(
            scanner="Relevance",
            scanner_kwargs={
                "model": {
                    "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                },
                "threshold": 0.5
            }
        )

        prompt = "What is artificial intelligence?"
        output = "Artificial intelligence is the simulation of human intelligence by machines."

        sanitized_output, is_valid, risk_score = preranker.scanner.scan(prompt, output)

        assert isinstance(sanitized_output, str)
        assert isinstance(is_valid, bool)
        assert isinstance(risk_score, np.floating)

    def test_relevance_scan_irrelevant_pair(self):
        """Test Relevance scanner with irrelevant prompt-output pair."""
        preranker = Preranker(
            scanner="Relevance",
            scanner_kwargs={
                "model": {
                    "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                },
                "threshold": 0.5
            }
        )

        prompt = "What is the weather today?"
        output = "Python is a programming language used for software development."

        sanitized_output, is_valid, risk_score = preranker.scanner.scan(prompt, output)

        assert isinstance(is_valid, bool)
        assert is_valid == False


class TestPrerankerErrors:
    """Tests for error handling."""

    def test_invalid_scanner_name(self):
        """Test error handling for invalid scanner name."""
        with pytest.raises(ValueError, match="Scanner .* not found"):
            Preranker(scanner="NonexistentScanner")

    def test_invalid_scanner_type(self):
        """Test error handling for invalid scanner type."""
        with pytest.raises(ValueError, match="Invalid scanner type"):
            Preranker(scanner="Relevance", scanner_type="invalid")


if __name__ == "__main__":
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from base import run_tests_with_report
    sys.exit(run_tests_with_report(__file__, "preranker"))
