import logging

from pytorch_pipeline.utils.logger import init_logger


def test_accumulating_logger(tmp_path):
    log_file = tmp_path / "test_app.log"

    # First session
    logger1 = init_logger(log_file, logging.INFO)
    logger1.info("Session 1 log entry")

    assert log_file.exists()
    content1 = log_file.read_text()
    assert "Session 1 log entry" in content1

    # Clear handlers for re-init test
    pkg_logger = logging.getLogger("pytorch_pipeline")
    pkg_logger.handlers.clear()

    # Second session
    logger2 = init_logger(log_file, logging.INFO)
    logger2.info("Session 2 log entry")

    content2 = log_file.read_text()
    # Verify both session logs exist in the file (accumulated)
    assert "Session 1 log entry" in content2
    assert "Session 2 log entry" in content2
