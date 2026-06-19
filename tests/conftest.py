"""Shared test fixtures and helpers."""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
