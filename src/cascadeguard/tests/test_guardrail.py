"""
Unit tests for CascadeGuard class.

Tests cover:
- Initialization with preranker/fineranker
- RankData creation
- Data conversion

Usage:
    pytest src/cascadeguard/tests/test_guardrail.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from cascadeguard.guardrail import CascadeGuard, RankData
from cascadeguard.preranker import Preranker
from cascadeguard.fineranker import Fineranker


class TestCascadeGuardInitialization:
    """Tests for CascadeGuard initialization."""

    @patch('cascadeguard.preranker.importlib.import_module')
    def test_init_with_preranker_only(self, mock_import):
        """Test initialization with preranker only."""
        mock_scanner_class = MagicMock()
        mock_module = MagicMock()
        mock_module.Gibberish = mock_scanner_class
        mock_import.return_value = mock_module

        guard = CascadeGuard(
            preranker={"scanner": "Gibberish"}
        )
        assert guard.preranker is not None
        assert guard.fineranker is None

    @patch('cascadeguard.fineranker.acompletion')
    def test_init_with_fineranker_only(self, mock_acompletion):
        """Test initialization with fineranker only."""
        guard = CascadeGuard(
            fineranker={"instruction": "Relevance", "model": "gpt-4o-mini"}
        )
        assert guard.preranker is None
        assert guard.fineranker is not None

    def test_init_with_none_raises_error(self):
        """Test that initialization without rankers raises error."""
        with pytest.raises(ValueError, match="At least one of preranker or fineranker"):
            CascadeGuard()


class TestRankData:
    """Tests for RankData dataclass."""

    def test_rankdata_creation(self):
        """Test creating RankData."""
        data = RankData(
            idx=0,
            prompt="Test prompt",
            output="Test output"
        )
        assert data.idx == 0
        assert data.prompt == "Test prompt"
        assert data.output == "Test output"

    def test_get_pair(self):
        """Test get_pair method."""
        data = RankData(idx=0, prompt="Q", output="A")
        pair = data.get_pair()
        assert pair == {"prompt": "Q", "output": "A"}

    def test_get_pair_prettystr(self):
        """Test get_pair_prettystr method."""
        data = RankData(idx=0, prompt="Q", output="A")
        pretty = data.get_pair_prettystr()
        assert "Prompt: Q" in pretty
        assert "Output: A" in pretty


class TestCascadeGuardDataConversion:
    """Tests for data conversion methods."""

    @patch('cascadeguard.preranker.importlib.import_module')
    def test_apply_as_datas(self, mock_import, sample_prompt_output_pairs):
        """Test converting pairs to RankData."""
        mock_scanner_class = MagicMock()
        mock_module = MagicMock()
        mock_module.Gibberish = mock_scanner_class
        mock_import.return_value = mock_module

        guard = CascadeGuard(preranker={"scanner": "Gibberish"})
        datas = guard.apply_as_datas(sample_prompt_output_pairs)

        assert len(datas) == 3
        assert all(isinstance(d, RankData) for d in datas)
        assert datas[0].idx == 0
        assert datas[1].idx == 1

    @patch('cascadeguard.preranker.importlib.import_module')
    def test_apply_datas_as_listdict(self, mock_import, sample_prompt_output_pairs):
        """Test converting RankData to list of dicts."""
        mock_scanner_class = MagicMock()
        mock_module = MagicMock()
        mock_module.Gibberish = mock_scanner_class
        mock_import.return_value = mock_module

        guard = CascadeGuard(preranker={"scanner": "Gibberish"})
        datas = guard.apply_as_datas(sample_prompt_output_pairs)
        list_dict = guard.apply_datas_as_listdict(datas)

        assert isinstance(list_dict, list)
        assert all(isinstance(d, dict) for d in list_dict)
        assert "prompt" in list_dict[0]
        assert "output" in list_dict[0]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from base import run_tests_with_report
    sys.exit(run_tests_with_report(__file__, "guardrail"))
