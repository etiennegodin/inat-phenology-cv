from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def save_log():
    import mlflow

    if mlflow.active_run():
        mlflow.log_artifact(str(Path.cwd() / "log.log"))


def get_mlflow_run_id() -> str | None:
    import mlflow

    return mlflow.active_run().info.run_id if mlflow.active_run() else None


def resolve_uri() -> str:
    uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )
    logger.debug(uri)
    return uri


def patch_mlflow_socks():
    try:
        import requests
    except ImportError:
        return

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(var, None)

    mlflow_hosts = ("100.101.196.27",)
    socks_proxies = {
        "http": "socks5h://localhost:1055",
        "https": "socks5h://localhost:1055",
    }

    original_session_request = requests.Session.request

    def _scoped_request(self, method, url, *args, **kwargs):
        if any(host in url for host in mlflow_hosts):
            kwargs.setdefault("proxies", socks_proxies)
        return original_session_request(self, method, url, *args, **kwargs)

    requests.Session.request = _scoped_request
