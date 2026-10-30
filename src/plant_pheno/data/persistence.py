from __future__ import annotations

import logging
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path

import mlflow
import torch

logger = logging.getLogger(__name__)


class RemoteTorchSaver:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._err: BaseException | None = None

    def execute(
        self, payload, save_path, save_file, to_mlflow=False, async_transfer=True
    ):
        os.makedirs(save_path, exist_ok=True)
        self.wait()  # also re-raises a previous failure

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            temp_path = f.name
        torch.save(payload, temp_path)

        def _transfer_job(src, dst):
            dst_tmp = f"{dst}.tmp"
            try:
                shutil.copy2(src, dst_tmp)
                os.replace(dst_tmp, dst)  # dst_tmp is beside dst, so same filesystem
                logger.info("Successfully updated target file: %s", dst)
                if to_mlflow and mlflow.active_run():
                    mlflow.log_artifact(dst)  # only after the file exists
            except BaseException as e:
                self._err = e
                logger.error("Failed to copy file to %s: %s", dst, e)
                Path(dst_tmp).unlink(missing_ok=True)
            finally:
                Path(src).unlink(missing_ok=True)

        if async_transfer:
            self._thread = threading.Thread(
                target=_transfer_job, args=(temp_path, save_file), daemon=True
            )
            self._thread.start()
        else:
            _transfer_job(temp_path, save_file)
            self._raise()

    def wait(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.info("Waiting for active background file transfer...")
            self._thread.join()
        self._raise()

    def _raise(self) -> None:
        if self._err is not None:
            err, self._err = self._err, None
            raise RuntimeError("background transfer failed") from err


class LocalTorchSaver:
    """Background writer for same-disk saves. Bounded queue gives backpressure."""

    def __init__(self, max_pending: int = 2) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=max_pending)
        self._err: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                payload, path = item
                if self._err is not None:  # a write already failed; drain the queue
                    continue
                tmp = path.with_name(path.name + ".tmp")  # same dir => same filesystem
                try:
                    torch.save(payload, tmp)
                    os.replace(tmp, path)
                    logger.info("Saved %s", path)
                except BaseException as e:
                    self._err = e
                    tmp.unlink(missing_ok=True)
            finally:
                self._q.task_done()

    def save(self, payload, path: str | Path) -> None:
        self._raise()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._q.put((payload, path))  # blocks when full

    def wait(self) -> None:
        self._q.join()
        self._raise()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()
        self._raise()

    def _raise(self) -> None:
        if self._err is not None:
            raise RuntimeError("background write failed") from self._err

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except RuntimeError:
            if exc[0] is None:
                raise
