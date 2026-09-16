# mlflow_socks_patch.py
import os

import requests

for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(var, None)

MLFLOW_HOSTS = ("100.101.196.27",)
SOCKS_PROXIES = {
    "http": "socks5h://localhost:1055",
    "https": "socks5h://localhost:1055",
}

_original_session_request = requests.Session.request


def _scoped_request(self, method, url, *args, **kwargs):
    if any(host in url for host in MLFLOW_HOSTS):
        kwargs.setdefault("proxies", SOCKS_PROXIES)
    return _original_session_request(self, method, url, *args, **kwargs)


requests.Session.request = _scoped_request
