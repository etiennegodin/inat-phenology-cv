from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading

import mlflow
import torch

logger = logging.getLogger(__name__)


class TorchSaver:
    """Manages robust, non-blocking asynchronous checkpoint saving."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    def _execute(
        self,
        payload: dict,
        save_path: str,
        save_file: str,
        to_mlflow: bool = False,
        async_transfer: bool = True,
    ) -> None:
        """Saves torch payload to fast local disk first,
        then copies to target path asynchronously."""

        os.makedirs(save_path, exist_ok=True)

        # 1. Fast synchronous write to local temp file (~50ms)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        temp_path = temp_file.name
        temp_file.close()

        logger.info(f"Saving checkpoint locally to temporary file: {temp_path}")
        torch.save(payload, temp_path)

        # 2. Transfer job to copy from temp_path to final destination
        def _transfer_job(src_path: str, dst_file: str):
            dst_tmp = f"{dst_file}.tmp"
            try:
                shutil.copy2(src_path, dst_tmp)
                if os.path.exists(dst_file):
                    os.replace(dst_tmp, dst_file)
                else:
                    shutil.move(dst_tmp, dst_file)
                logger.info(f"Successfully updated target checkpoint: {dst_file}")
            except Exception as e:
                logger.error(f"Failed to copy checkpoint to {dst_file}: {e}")
                if os.path.exists(dst_tmp):
                    try:
                        os.remove(dst_tmp)
                    except Exception:
                        pass
                raise e
            finally:
                if os.path.exists(src_path):
                    try:
                        os.remove(src_path)
                    except Exception:
                        pass

        # Wait for any prior background transfer to finish before starting a new one
        self.wait()

        if async_transfer:
            self._thread = threading.Thread(
                target=_transfer_job, args=(temp_path, save_file), daemon=True
            )
            self._thread.start()
        else:
            _transfer_job(temp_path, save_file)

        if to_mlflow:
            if mlflow.active_run():
                mlflow.log_artifact(save_file)

    def wait(self) -> None:
        """Wait for any active background checkpoint transfer to complete."""
        if self._thread is not None and self._thread.is_alive():
            logger.info("Waiting for active background checkpoint transfer...")
            self._thread.join()
