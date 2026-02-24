import logging

import pytest

from cellrank._settings import _LOGGER_NAME, CellRankConfig, _setup_logger


@pytest.fixture
def fresh_logger():
    """Provide a clean cellrank logger for each test."""
    log = logging.getLogger(_LOGGER_NAME)
    old_handlers = log.handlers[:]
    old_level = log.level
    yield log
    log.handlers = old_handlers
    log.setLevel(old_level)


class TestSetupLogger:
    def test_adds_handler(self, fresh_logger):
        fresh_logger.handlers.clear()
        _setup_logger()
        assert len(fresh_logger.handlers) == 1

    def test_idempotent(self, fresh_logger):
        n = len(fresh_logger.handlers)
        _setup_logger()
        assert len(fresh_logger.handlers) == n

    def test_default_level(self, fresh_logger):
        fresh_logger.handlers.clear()
        fresh_logger.setLevel(logging.NOTSET)
        _setup_logger()
        assert fresh_logger.level == logging.INFO


class TestCellRankConfig:
    def test_logging_level_default(self):
        cfg = CellRankConfig()
        # Default set by _setup_logger
        assert logging.getLogger(_LOGGER_NAME).level == cfg.logging_level

    def test_logging_level_set(self, fresh_logger):
        cfg = CellRankConfig()
        cfg.logging_level = logging.DEBUG
        assert fresh_logger.level == logging.DEBUG
        cfg.logging_level = "WARNING"
        assert fresh_logger.level == logging.WARNING

    def test_override_logging_level(self, fresh_logger):
        cfg = CellRankConfig()
        original = fresh_logger.level
        with cfg.override_logging_level(logging.DEBUG):
            assert fresh_logger.level == logging.DEBUG
        assert fresh_logger.level == original

    def test_override_logging_level_restores_on_error(self, fresh_logger):
        cfg = CellRankConfig()
        original = fresh_logger.level
        with pytest.raises(RuntimeError), cfg.override_logging_level(logging.DEBUG):
            raise RuntimeError("boom")
        assert fresh_logger.level == original

    def test_figdir_default(self):
        from pathlib import Path

        cfg = CellRankConfig()
        assert cfg.figdir == Path("./figures")

    def test_figdir_set(self, tmp_path):
        cfg = CellRankConfig()
        cfg.figdir = tmp_path
        assert cfg.figdir == tmp_path

    def test_child_logger_inherits(self, fresh_logger):
        cfg = CellRankConfig()
        cfg.logging_level = logging.DEBUG
        child = logging.getLogger(f"{_LOGGER_NAME}.test_child")
        assert child.getEffectiveLevel() == logging.DEBUG
