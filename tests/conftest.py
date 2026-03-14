"""Shared pytest fixtures for CatchUp tests."""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _bypass_parse_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent parse-result caching from leaking between tests.

    parse_ipynb (and parse_pdf) call load_cached_parse / save_cached_parse
    inside the function body.  The cache is keyed by file-content hash, so
    two tests that write the same stub bytes (e.g. ``"{}"``) to a temp file
    share the same cache entry.  This fixture stubs both functions out so
    every test starts from a clean state without touching the filesystem.
    """
    monkeypatch.setattr("utils.cache.load_cached_parse", lambda _path: None)
    monkeypatch.setattr("utils.cache.save_cached_parse", lambda _path, _doc: None)
