"""
Unit tests for Fineranker class.

Tests cover:
- Instruction setup
- Boolean parsing
- Response dataclass

Usage:
    pytest src/cascadeguard/tests/test_fineranker.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

from cascadeguard.fineranker import (
    Fineranker,
    FinerankResponse,
    FinerankResponseStatus,
    FinerankResponseMetadata,
)


class TestFinerankerInitialization:
    """Tests for Fineranker initialization."""

    def test_init_with_builtin_instruction(self):
        """Test initialization with built-in instruction."""
        fineranker = Fineranker(
            instruction="Relevance",
            model="gpt-4o-mini"
        )
        assert fineranker.instruction is not None
        assert "Reject if" in fineranker.instruction

    def test_init_with_custom_instruction(self):
        """Test initialization with custom instruction."""
        custom = "Reject if output contains profanity."
        fineranker = Fineranker(
            instruction=custom,
            model="gpt-4o-mini"
        )
        assert fineranker.instruction == custom


class TestFinerankerBoolParsing:
    """Tests for boolean parsing logic."""

    def test_parse_bool_true_variants(self):
        """Test parsing various 'true' formats."""
        fineranker = Fineranker(instruction="Relevance", model="gpt-4o-mini")

        assert fineranker._parse_bool("True") == True
        assert fineranker._parse_bool("true") == True
        assert fineranker._parse_bool("  True  ") == True
        assert fineranker._parse_bool("1") == True
        assert fineranker._parse_bool("yes") == True

    def test_parse_bool_false_variants(self):
        """Test parsing various 'false' formats."""
        fineranker = Fineranker(instruction="Relevance", model="gpt-4o-mini")

        assert fineranker._parse_bool("False") == False
        assert fineranker._parse_bool("false") == False
        assert fineranker._parse_bool("  False  ") == False
        assert fineranker._parse_bool("0") == False
        assert fineranker._parse_bool("no") == False

    def test_parse_bool_invalid(self):
        """Test error handling for invalid boolean strings."""
        fineranker = Fineranker(instruction="Relevance", model="gpt-4o-mini")

        with pytest.raises(ValueError):
            fineranker._parse_bool("maybe")
        with pytest.raises(ValueError):
            fineranker._parse_bool("invalid")


class TestFinerankerResponse:
    """Tests for FinerankResponse dataclass."""

    def test_response_creation(self):
        """Test creating FinerankResponse."""
        response = FinerankResponse(
            is_valid=True,
            status=FinerankResponseStatus.SUCCESS
        )
        assert response.is_valid == True
        assert response.status == "success"

    def test_response_with_metadata(self):
        """Test FinerankResponse with metadata."""
        metadata = FinerankResponseMetadata(
            raw_response="True",
            error=""
        )
        response = FinerankResponse(
            is_valid=True,
            metadata=metadata
        )
        assert response.metadata.raw_response == "True"


class TestFinerankerMocked:
    """Tests for Fineranker with mocked LLM calls."""

    @patch('cascadeguard.fineranker.acompletion')
    def test_rank_success(self, mock_acompletion):
        """Test successful ranking with mocked LLM."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "True"
        mock_acompletion.return_value = mock_response

        fineranker = Fineranker(instruction="Relevance", model="gpt-4o-mini")
        result = fineranker.rank("Prompt: Test\nOutput: Response")

        assert isinstance(result, FinerankResponse)
        assert result.is_valid == True
        assert result.status == "success"

    @patch('cascadeguard.fineranker.acompletion')
    def test_rank_return_dict(self, mock_acompletion):
        """Test ranking with dict return format."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "False"
        mock_acompletion.return_value = mock_response

        fineranker = Fineranker(instruction="Relevance", model="gpt-4o-mini")
        result = fineranker.rank("Test context", return_as_dict=True)

        assert isinstance(result, dict)
        assert "is_valid" in result
        assert result["is_valid"] == False


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from base import run_tests_with_report
    sys.exit(run_tests_with_report(__file__, "fineranker"))
